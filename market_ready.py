"""
market_ready.py

Institutional-style decision layer for EquityLens.

This module converts raw annual statements, quarterly data, market history,
peers, DCF, and Monte Carlo output into a practical analyst dashboard:
- Forward estimates and quarterly/TTM trend checks
- Sector-specific valuation model selection
- Historical valuation bands and momentum overlay
- Earnings revision logic
- Data/source reliability scoring
- Signal explainability and historical price backtest proxy
"""

import re

import numpy as np
import pandas as pd

from ratios import last_valid
from scraper import to_num
from validation_engine import run_historical_validation


def _safe_float(value, default=np.nan):
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _valid(values):
    out = []
    for value in values or []:
        f = _safe_float(value)
        if not np.isnan(f):
            out.append(f)
    return out


def _nanmean_or_nan(values):
    vals = _valid(values)
    return float(np.mean(vals)) if vals else np.nan


def _cagr(start, end, years):
    if start and start > 0 and end and end > 0 and years > 0:
        return (end / start) ** (1 / years) - 1
    return np.nan


def _text_blob(data):
    return " ".join(str(data.get(k, "") or "") for k in (
        "ticker", "company_name", "sector", "industry", "about",
    )).lower()


def _classify_sector(data, assumptions=None):
    biz = (assumptions or {}).get("business_type", "")
    text = _text_blob(data)
    if any(x in text for x in ["bank", "nbfc", "housing finance", "home loan", "finance"]):
        return "Banks/NBFCs"
    if any(x in text for x in ["exchange", "clearing", "depository", "capital market"]):
        return "Exchanges"
    if any(x in text for x in ["information technology", "software", "it services", "digital"]):
        return "IT Services"
    if any(x in text for x in ["fmcg", "consumer", "foods", "personal care"]):
        return "FMCG/Consumer"
    if any(x in text for x in ["metal", "steel", "aluminium", "zinc", "mining", "commodity"]):
        return "Metals/Commodity"
    if any(x in text for x in ["real estate", "developer", "property"]):
        return "Real Estate"
    if any(x in text for x in ["insurance", "asset management", "mutual fund", "amc"]):
        return "Insurance/AMC"
    if biz == "exchange-platform":
        return "Exchanges"
    if biz == "commodity":
        return "Metals/Commodity"
    return "General Corporate"


def _extract_row_series(df, patterns, n=None):
    if df is None or df.empty:
        return []
    first_col = df.iloc[:, 0].astype(str)
    for pattern in patterns:
        matches = df[first_col.str.contains(pattern, case=False, na=False, regex=True)]
        if not matches.empty:
            vals = [to_num(v) for v in matches.iloc[0].iloc[1:]]
            vals = [v for v in vals if not np.isnan(v)]
            return vals[-n:] if n else vals
    return []


def _metric_status(value, good=None, bad=None, higher_is_better=True):
    f = _safe_float(value)
    if np.isnan(f):
        return "Unavailable"
    if higher_is_better:
        if good is not None and f >= good:
            return "Strong"
        if bad is not None and f <= bad:
            return "Weak"
    else:
        if good is not None and f <= good:
            return "Strong"
        if bad is not None and f >= bad:
            return "Weak"
    return "Neutral"


def evaluate_data_quality(data, r, peer_df=None):
    """Score input completeness and identify reliability warnings."""
    core_fields = [
        "revenue", "ebitda", "pat", "eps", "cfo", "total_debt", "equity",
        "operating_margin", "roe", "roce",
    ]
    expected = max(int(r.get("n_years", 5)), 1)
    total_points = len(core_fields) * expected
    present = 0
    missing_fields = []

    for key in core_fields:
        vals = _valid(r.get(key, []))
        present += min(len(vals), expected)
        if not vals:
            missing_fields.append(key)

    top_fields = ["current_price", "market_cap", "pe_ratio", "book_value"]
    top_present = sum(1 for key in top_fields if not np.isnan(_safe_float(data.get(key))))
    peer_points = 1 if peer_df is not None and len(peer_df) > 1 else 0
    quarter_points = 1 if data.get("quarters") is not None and not data.get("quarters").empty else 0
    price_points = 1 if data.get("market_data", {}).get("price_history") is not None else 0
    exchange_events = data.get("exchange_events") or {}
    actions = exchange_events.get("corporate_actions")
    announcements = exchange_events.get("announcements")
    event_points = 1 if (
        (actions is not None and hasattr(actions, "empty") and not actions.empty)
        or (announcements is not None and hasattr(announcements, "empty") and not announcements.empty)
    ) else 0

    score = (
        (present / total_points) * 58
        + (top_present / len(top_fields)) * 13
        + peer_points * 10
        + quarter_points * 8
        + price_points * 7
        + event_points * 4
    )
    warnings = []
    if missing_fields:
        warnings.append("Missing: " + ", ".join(missing_fields[:5]))
    if not peer_points:
        warnings.append("Peer support is weak or unavailable")
    if not quarter_points:
        warnings.append("Quarterly result table unavailable; signal timing is weaker")
    if not price_points:
        warnings.append("Yahoo price history unavailable; momentum/backtest evidence is weaker")
    if not event_points:
        warnings.append("NSE corporate actions/announcements unavailable; event context is weaker")
    if not data.get("url", "").endswith("/consolidated/"):
        warnings.append("May be standalone data if consolidated page was unavailable")

    return {
        "score": round(float(np.clip(score, 0, 100)), 1),
        "missing_fields": missing_fields,
        "warnings": warnings,
        "rating": "High" if score >= 80 else "Medium" if score >= 55 else "Low",
    }


def quarterly_snapshot(data):
    """Extract latest quarterly, TTM, YoY/QoQ and margin trend signals."""
    qdf = data.get("quarters")
    if qdf is None or qdf.empty:
        return {
            "available": False,
            "rating": "Unavailable",
            "note": "Quarterly Screener table unavailable",
            "table": pd.DataFrame(),
        }

    revenue = _extract_row_series(qdf, [r"sales|revenue|income"], 12)
    op_profit = _extract_row_series(qdf, [r"operating profit|ebitda"], 12)
    pat = _extract_row_series(qdf, [r"net profit|profit after tax|pat"], 12)
    eps = _extract_row_series(qdf, [r"eps"], 12)
    opm = _extract_row_series(qdf, [r"opm"], 12)

    def growth(vals, periods):
        if len(vals) <= periods or vals[-periods - 1] == 0:
            return np.nan
        return (vals[-1] / vals[-periods - 1] - 1) * 100

    latest_margin = opm[-1] if opm else (op_profit[-1] / revenue[-1] * 100 if revenue and op_profit and revenue[-1] else np.nan)
    prev4_margin = np.nan
    last4_margin = np.nan
    if len(opm) >= 8:
        prev4_margin = _nanmean_or_nan(opm[-8:-4])
        last4_margin = _nanmean_or_nan(opm[-4:])
    elif len(op_profit) >= 8 and len(revenue) >= 8 and sum(revenue[-4:]) and sum(revenue[-8:-4]):
        last4_margin = sum(op_profit[-4:]) / sum(revenue[-4:]) * 100
        prev4_margin = sum(op_profit[-8:-4]) / sum(revenue[-8:-4]) * 100

    yoy = growth(revenue, 4)
    pat_yoy = growth(pat, 4)
    qoq = growth(revenue, 1)
    pat_qoq = growth(pat, 1)
    margin_change = last4_margin - prev4_margin if not any(np.isnan(x) for x in [last4_margin, prev4_margin]) else np.nan

    score = 0
    score += 1 if not np.isnan(yoy) and yoy > 10 else -1 if not np.isnan(yoy) and yoy < 0 else 0
    score += 1 if not np.isnan(pat_yoy) and pat_yoy > 10 else -1 if not np.isnan(pat_yoy) and pat_yoy < 0 else 0
    score += 1 if not np.isnan(margin_change) and margin_change > 1 else -1 if not np.isnan(margin_change) and margin_change < -1 else 0
    rating = "Improving" if score >= 2 else "Weakening" if score <= -2 else "Mixed"

    summary = {
        "Metric": [
            "Latest Revenue", "Latest PAT", "Latest EPS", "Latest OPM %",
            "Revenue YoY %", "Revenue QoQ %", "PAT YoY %", "PAT QoQ %",
            "TTM Revenue", "TTM PAT", "TTM EPS", "TTM/Last4 Margin Change %",
        ],
        "Value": [
            revenue[-1] if revenue else np.nan,
            pat[-1] if pat else np.nan,
            eps[-1] if eps else np.nan,
            latest_margin,
            yoy, qoq, pat_yoy, pat_qoq,
            sum(revenue[-4:]) if len(revenue) >= 4 else np.nan,
            sum(pat[-4:]) if len(pat) >= 4 else np.nan,
            sum(eps[-4:]) if len(eps) >= 4 else np.nan,
            margin_change,
        ],
    }
    table = pd.DataFrame(summary)
    return {
        "available": True,
        "rating": rating,
        "latest_revenue": summary["Value"][0],
        "latest_pat": summary["Value"][1],
        "ttm_revenue": summary["Value"][8],
        "ttm_pat": summary["Value"][9],
        "ttm_eps": summary["Value"][10],
        "revenue_yoy_pct": round(float(yoy), 1) if not np.isnan(yoy) else np.nan,
        "pat_yoy_pct": round(float(pat_yoy), 1) if not np.isnan(pat_yoy) else np.nan,
        "revenue_qoq_pct": round(float(qoq), 1) if not np.isnan(qoq) else np.nan,
        "pat_qoq_pct": round(float(pat_qoq), 1) if not np.isnan(pat_qoq) else np.nan,
        "margin_change_pct": round(float(margin_change), 1) if not np.isnan(margin_change) else np.nan,
        "table": table,
    }


def forecast_financials(data, r, assumptions, years=3, quarterly=None):
    """Build a compact forward income/cash-flow forecast."""
    base_revenue = _safe_float(last_valid(r.get("revenue", [])), 0)
    if quarterly and not np.isnan(_safe_float(quarterly.get("ttm_revenue"))):
        base_revenue = max(base_revenue, _safe_float(quarterly.get("ttm_revenue"), base_revenue))

    base_margin = _safe_float(assumptions.get("base_ebitda_margin"), 15.0) / 100
    base_eps = _safe_float(last_valid(r.get("eps", [])), np.nan)
    if quarterly and not np.isnan(_safe_float(quarterly.get("ttm_eps"))):
        base_eps = _safe_float(quarterly.get("ttm_eps"), base_eps)

    market_cap = _safe_float(data.get("market_cap"), np.nan)
    price = _safe_float(data.get("current_price"), np.nan)
    shares = market_cap / price if not np.isnan(market_cap) and not np.isnan(price) and price > 0 else np.nan
    if np.isnan(base_eps):
        pat = _safe_float(last_valid(r.get("pat", [])), np.nan)
        base_eps = pat / shares if not np.isnan(pat) and shares else np.nan

    growth = _safe_float(assumptions.get("base_growth"), 0.10)
    if quarterly:
        q_yoy = _safe_float(quarterly.get("revenue_yoy_pct"), np.nan)
        if not np.isnan(q_yoy):
            growth = float(np.clip((growth * 0.65) + (q_yoy / 100 * 0.35), -0.10, 0.35))

    margin_delta = _safe_float(assumptions.get("base_margin_delta"), 0.0)
    tax_rate = _safe_float(assumptions.get("tax_rate"), 0.25)
    capex_pct = _safe_float(assumptions.get("capex_pct"), 0.06)
    wc_pct = _safe_float(assumptions.get("wc_pct"), 0.015)

    rows = []
    revenue = base_revenue
    margin = base_margin
    eps = base_eps
    for year in range(1, years + 1):
        revenue *= 1 + growth
        margin = float(np.clip(margin + margin_delta, 0.03, 0.70))
        ebitda = revenue * margin
        pat = ebitda * (1 - tax_rate)
        fcff = pat - revenue * capex_pct - revenue * wc_pct
        if not np.isnan(eps):
            eps *= 1 + max(growth + margin_delta, -0.20)
        rows.append({
            "Year": f"FY+{year}",
            "Revenue": round(revenue, 1),
            "EBITDA": round(ebitda, 1),
            "PAT": round(pat, 1),
            "EPS": round(eps, 2) if not np.isnan(eps) else np.nan,
            "FCFF": round(fcff, 1),
            "EBITDA Margin %": round(margin * 100, 1),
        })

    return pd.DataFrame(rows)


def sector_specific_valuation(data, r, assumptions, dcf_result=None, forecast=None, quarterly=None):
    """Select a valuation model based on business type and sector economics."""
    sector_model = _classify_sector(data, assumptions)
    price = _safe_float(data.get("current_price"), np.nan)
    eps = _safe_float(quarterly.get("ttm_eps") if quarterly else np.nan, np.nan)
    if np.isnan(eps):
        eps = _safe_float(last_valid(r.get("eps", [])), np.nan)
    book = _safe_float(data.get("book_value"), np.nan)
    roe = _safe_float(last_valid(r.get("roe", [])), np.nan)
    roce = _safe_float(last_valid(r.get("roce", [])), np.nan)
    growth = _safe_float(r.get("eps_cagr_3y"), _safe_float(r.get("pat_cagr_3y"), 10))
    opm = _safe_float(last_valid(r.get("operating_margin", [])), np.nan)

    target = np.nan
    fair_multiple = np.nan
    drivers = []

    if sector_model == "Banks/NBFCs" and book > 0:
        fair_multiple = float(np.clip((roe if not np.isnan(roe) else 12) / 10 + max(growth, 0) / 60, 0.7, 4.5))
        target = book * fair_multiple
        drivers = ["P/B driven by ROE, book value and credit growth proxy", f"Fair P/B {fair_multiple:.1f}x"]
    elif sector_model == "Exchanges":
        platform = (dcf_result or {}).get("platform_valuation", {})
        if platform.get("base_target"):
            target = _safe_float(platform.get("base_target"), np.nan)
            fair_multiple = _safe_float(platform.get("target_multiple"), np.nan)
        elif eps > 0:
            fair_multiple = float(np.clip(18 + max(growth, 0) * 0.8 + max(opm, 0) * 0.12, 20, 65))
            target = eps * fair_multiple
        drivers = ["Forward EPS model for exchange/platform economics", "Uses growth, margin and transaction-volume proxy"]
    elif sector_model in ("IT Services", "FMCG/Consumer") and eps > 0:
        quality_premium = 6 if not np.isnan(roce) and roce > 25 else 2
        fair_multiple = float(np.clip(14 + max(growth, 0) * 0.9 + quality_premium, 15, 50))
        target = eps * fair_multiple
        drivers = ["Forward P/E model for asset-light quality businesses", f"Quality-adjusted fair P/E {fair_multiple:.1f}x"]
    elif sector_model == "Metals/Commodity":
        ev_ebitda = _safe_float(r.get("ev_ebitda"), np.nan)
        ebitda = _safe_float(last_valid(r.get("ebitda", [])), np.nan)
        market_cap = _safe_float(data.get("market_cap"), np.nan)
        if ev_ebitda > 0 and ebitda > 0 and market_cap > 0 and price > 0:
            fair_multiple = float(np.clip(ev_ebitda * 0.9, 3, 8))
            fair_ev = ebitda * fair_multiple
            target = price * (fair_ev / market_cap)
        drivers = ["Cycle-adjusted EV/EBITDA proxy", "Commodity businesses require lower normalized multiples"]
    elif sector_model == "Real Estate" and book > 0:
        fair_multiple = float(np.clip((roe if not np.isnan(roe) else 10) / 12 + 0.7, 0.7, 2.5))
        target = book * fair_multiple
        drivers = ["NAV/P/B proxy until project-level NAV is available", f"Fair P/B {fair_multiple:.1f}x"]
    elif sector_model == "Insurance/AMC" and eps > 0:
        fair_multiple = float(np.clip(18 + max(growth, 0) * 0.7, 16, 45))
        target = eps * fair_multiple
        drivers = ["AUM/VNB data unavailable; using forward earnings proxy", f"Fair P/E {fair_multiple:.1f}x"]
    elif eps > 0:
        fair_multiple = float(np.clip(8 + max(growth, 0) * 1.1, 8, 40))
        target = eps * fair_multiple
        drivers = ["General forward P/E model", f"Fair P/E {fair_multiple:.1f}x"]

    upside = (target - price) / price * 100 if target > 0 and price > 0 else np.nan
    return {
        "sector_model": sector_model,
        "target_price": round(float(target), 2) if not np.isnan(target) else np.nan,
        "upside_pct": round(float(upside), 1) if not np.isnan(upside) else np.nan,
        "fair_multiple": round(float(fair_multiple), 2) if not np.isnan(fair_multiple) else np.nan,
        "drivers": drivers or ["Sector model could not be computed from available data"],
    }


def blended_target_price(data, r, dcf_result, peer_df=None, sector_valuation=None):
    """Blend DCF, relative valuation, and sector/platform models into one target."""
    current_price = _safe_float(data.get("current_price"), np.nan)
    pieces = []

    def add(name, value, weight, note):
        value = _safe_float(value, np.nan)
        if not np.isnan(value) and value > 0 and weight > 0:
            pieces.append({"Model": name, "Target": value, "Weight": weight, "Note": note})

    if sector_valuation:
        add(
            f"Sector Model: {sector_valuation.get('sector_model')}",
            sector_valuation.get("target_price"),
            0.30,
            "; ".join(sector_valuation.get("drivers", [])[:2]),
        )

    if dcf_result:
        add("DCF Base", dcf_result.get("base_iv"), 0.25, "FCFF intrinsic value")
        platform = dcf_result.get("platform_valuation", {})
        if platform:
            add("Platform EPS", platform.get("base_target"), 0.30, "FY+2 EPS multiple model")

    eps = _safe_float(last_valid(r.get("eps", [])), np.nan)
    eps_growth = _safe_float(r.get("eps_cagr_3y"), np.nan)
    if not np.isnan(eps) and eps > 0:
        fair_pe = np.clip((eps_growth if not np.isnan(eps_growth) else 12) * 1.2, 8, 45)
        add("Fair P/E", eps * fair_pe, 0.16, f"EPS x {fair_pe:.1f} fair P/E")

    book_value = _safe_float(data.get("book_value"), np.nan)
    roe = _safe_float(last_valid(r.get("roe", [])), np.nan)
    if not np.isnan(book_value) and book_value > 0:
        fair_pb = np.clip((roe if not np.isnan(roe) else 12) / 8, 0.8, 8.0)
        add("Fair P/B", book_value * fair_pb, 0.09, f"BV x {fair_pb:.1f} fair P/B")

    if peer_df is not None and "P/E" in peer_df.columns and not np.isnan(eps):
        peers = peer_df[peer_df.get("_is_target") == False] if "_is_target" in peer_df.columns else peer_df
        peer_pe = pd.to_numeric(peers.get("P/E"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not peer_pe.empty:
            add("Peer P/E", eps * float(peer_pe.median()), 0.12, "Peer median P/E")

    if not pieces:
        return {"target_price": np.nan, "upside_pct": np.nan, "components": pd.DataFrame()}

    total_weight = sum(p["Weight"] for p in pieces)
    target = sum(p["Target"] * p["Weight"] for p in pieces) / total_weight
    upside = (target - current_price) / current_price * 100 if current_price and not np.isnan(current_price) else np.nan
    comp = pd.DataFrame(pieces)
    comp["Weight %"] = (comp["Weight"] / total_weight * 100).round(1)
    comp["Target"] = comp["Target"].round(2)

    return {
        "target_price": round(float(target), 2),
        "upside_pct": round(float(upside), 1) if not np.isnan(upside) else np.nan,
        "components": comp,
    }


def valuation_bands(data, r):
    """Compute historical price/valuation band context from available market history."""
    hist = (data.get("market_data") or {}).get("price_history")
    price = _safe_float(data.get("current_price"), np.nan)
    pe = _safe_float(data.get("pe_ratio"), np.nan)
    pb = _safe_float(r.get("pb_ratio"), np.nan)
    if hist is None or not hasattr(hist, "empty") or hist.empty or "Close" not in hist:
        return {
            "available": False,
            "rating": "Unavailable",
            "note": "Price history unavailable; valuation percentile cannot be computed",
            "table": pd.DataFrame(),
        }

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if close.empty:
        return {"available": False, "rating": "Unavailable", "table": pd.DataFrame()}

    price_pct = (close.rank(pct=True).iloc[-1] * 100) if len(close) > 1 else np.nan
    high = close.max()
    low = close.min()
    median_price = close.median()
    drawdown = (price / high - 1) * 100 if price > 0 and high > 0 else np.nan
    eps = _safe_float(last_valid(r.get("eps", [])), np.nan)
    bv = _safe_float(data.get("book_value"), np.nan)
    pe_band = close / eps if eps > 0 else pd.Series(dtype=float)
    pb_band = close / bv if bv > 0 else pd.Series(dtype=float)

    table = pd.DataFrame([
        {"Metric": "Price Percentile", "Current": price_pct, "Median": 50, "Low": 0, "High": 100},
        {"Metric": "Price", "Current": price, "Median": median_price, "Low": low, "High": high},
        {"Metric": "Drawdown From 5Y High %", "Current": drawdown, "Median": np.nan, "Low": np.nan, "High": np.nan},
        {"Metric": "Approx P/E Band", "Current": pe, "Median": pe_band.median() if not pe_band.empty else np.nan, "Low": pe_band.quantile(0.1) if not pe_band.empty else np.nan, "High": pe_band.quantile(0.9) if not pe_band.empty else np.nan},
        {"Metric": "Approx P/B Band", "Current": pb, "Median": pb_band.median() if not pb_band.empty else np.nan, "Low": pb_band.quantile(0.1) if not pb_band.empty else np.nan, "High": pb_band.quantile(0.9) if not pb_band.empty else np.nan},
    ])
    if price_pct >= 80:
        rating = "Expensive vs history"
    elif price_pct <= 30:
        rating = "Discounted vs history"
    else:
        rating = "Near historical band"
    return {
        "available": True,
        "rating": rating,
        "price_percentile": round(float(price_pct), 1) if not np.isnan(price_pct) else np.nan,
        "drawdown_from_high_pct": round(float(drawdown), 1) if not np.isnan(drawdown) else np.nan,
        "table": table,
    }


def earnings_revision_signal(r, quarterly=None):
    """Proxy for analyst estimate revisions using growth acceleration and margin movement."""
    rev3 = _safe_float(r.get("revenue_cagr_3y"), np.nan)
    rev5 = _safe_float(r.get("revenue_cagr_5y"), np.nan)
    pat3 = _safe_float(r.get("pat_cagr_3y"), np.nan)
    pat5 = _safe_float(r.get("pat_cagr_5y"), np.nan)
    eps3 = _safe_float(r.get("eps_cagr_3y"), np.nan)
    eps5 = _safe_float(r.get("eps_cagr_5y"), np.nan)
    margin = _safe_float(quarterly.get("margin_change_pct") if quarterly else np.nan, np.nan)
    q_rev = _safe_float(quarterly.get("revenue_yoy_pct") if quarterly else np.nan, np.nan)
    q_pat = _safe_float(quarterly.get("pat_yoy_pct") if quarterly else np.nan, np.nan)

    drivers = []
    score = 0.0
    for label, short, long in [
        ("Revenue growth", rev3, rev5),
        ("PAT growth", pat3, pat5),
        ("EPS growth", eps3, eps5),
    ]:
        if not np.isnan(short) and not np.isnan(long):
            delta = short - long
            score += np.clip(delta / 5, -2, 2)
            drivers.append(f"{label} acceleration {delta:+.1f} ppt")
    for label, val in [("Quarterly revenue YoY", q_rev), ("Quarterly PAT YoY", q_pat)]:
        if not np.isnan(val):
            score += 1 if val > 15 else -1 if val < 0 else 0
            drivers.append(f"{label} {val:.1f}%")
    if not np.isnan(margin):
        score += 1 if margin > 1 else -1 if margin < -1 else 0
        drivers.append(f"TTM margin change {margin:+.1f} ppt")

    rating = "Upgrade Bias" if score >= 2 else "Downgrade Risk" if score <= -2 else "Neutral"
    return {"score": round(float(np.clip(score, -5, 5)), 1), "rating": rating, "drivers": drivers or ["Revision proxy unavailable"]}


def price_momentum_overlay(data):
    """Market behavior overlay: moving averages, relative strength, drawdown and volume."""
    hist = (data.get("market_data") or {}).get("price_history")
    bench = (data.get("market_data") or {}).get("benchmark_history")
    if hist is None or not hasattr(hist, "empty") or hist.empty or "Close" not in hist:
        return {"available": False, "rating": "Unavailable", "score": 0, "table": pd.DataFrame()}
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return {"available": False, "rating": "Insufficient history", "score": 0, "table": pd.DataFrame()}

    price = close.iloc[-1]
    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    ret_6m = close.iloc[-1] / close.iloc[-126] - 1 if len(close) > 126 and close.iloc[-126] else np.nan
    ret_12m = close.iloc[-1] / close.iloc[-252] - 1 if len(close) > 252 and close.iloc[-252] else np.nan
    high_52 = close.tail(252).max() if len(close) >= 252 else close.max()
    drawdown = (price / high_52 - 1) * 100 if high_52 else np.nan
    rs_12m = np.nan
    if bench is not None and hasattr(bench, "empty") and not bench.empty and "Close" in bench and len(close) > 252:
        bclose = pd.to_numeric(bench["Close"], errors="coerce").dropna()
        if len(bclose) > 252 and bclose.iloc[-252]:
            rs_12m = ret_12m * 100 - ((bclose.iloc[-1] / bclose.iloc[-252] - 1) * 100)

    vol_ratio = np.nan
    if "Volume" in hist:
        volume = pd.to_numeric(hist["Volume"], errors="coerce").dropna()
        if len(volume) >= 100 and volume.tail(100).mean():
            vol_ratio = volume.tail(20).mean() / volume.tail(100).mean()

    score = 0
    score += 1 if price > dma50 else -1
    if not np.isnan(dma200):
        score += 1 if price > dma200 else -1
    score += 1 if not np.isnan(rs_12m) and rs_12m > 0 else -1 if not np.isnan(rs_12m) and rs_12m < -5 else 0
    score += 1 if not np.isnan(drawdown) and drawdown > -15 else -1 if not np.isnan(drawdown) and drawdown < -30 else 0
    score += 1 if not np.isnan(vol_ratio) and vol_ratio > 1.2 and price > dma50 else 0
    rating = "Bullish" if score >= 3 else "Bearish" if score <= -2 else "Neutral"

    table = pd.DataFrame([
        {"Metric": "Price", "Value": price, "Status": "Current"},
        {"Metric": "50 DMA", "Value": dma50, "Status": "Above" if price > dma50 else "Below"},
        {"Metric": "200 DMA", "Value": dma200, "Status": "Above" if not np.isnan(dma200) and price > dma200 else "Below/NA"},
        {"Metric": "6M Return %", "Value": ret_6m * 100 if not np.isnan(ret_6m) else np.nan, "Status": _metric_status(ret_6m * 100 if not np.isnan(ret_6m) else np.nan, 10, -10)},
        {"Metric": "Relative Strength vs Nifty 12M %", "Value": rs_12m, "Status": _metric_status(rs_12m, 0, -10)},
        {"Metric": "Drawdown from 52W High %", "Value": drawdown, "Status": _metric_status(drawdown, -15, -30)},
        {"Metric": "20D/100D Volume Ratio", "Value": vol_ratio, "Status": _metric_status(vol_ratio, 1.2, 0.7)},
    ])
    return {"available": True, "rating": rating, "score": score, "table": table}


def risk_score(data, r, flags=None, assumptions=None, data_quality=None, momentum=None, revision=None):
    """Estimate downside/reliability risk on a 0-100 scale where high is riskier."""
    flags = flags or []
    assumptions = assumptions or {}
    score = 20.0
    reasons = []

    high_flags = sum(1 for f in flags if f.get("triggered") and f.get("severity") == "high")
    med_flags = sum(1 for f in flags if f.get("triggered") and f.get("severity") == "medium")
    score += high_flags * 12 + med_flags * 6
    if high_flags or med_flags:
        reasons.append(f"{high_flags} high and {med_flags} medium red flags")

    de = _safe_float(last_valid(r.get("debt_equity", [])), 0)
    if de > 1.5:
        score += 15; reasons.append("High debt/equity")
    elif de > 0.8:
        score += 8; reasons.append("Moderate leverage")

    margins = _valid(r.get("operating_margin", []))
    if len(margins) >= 3 and np.std(margins) > 8:
        score += 10; reasons.append("Volatile margins")

    pe = _safe_float(data.get("pe_ratio"), np.nan)
    eps_growth = _safe_float(r.get("eps_cagr_3y"), np.nan)
    if not np.isnan(pe) and not np.isnan(eps_growth) and pe > max(eps_growth * 1.8, 45):
        score += 12; reasons.append("Valuation is ahead of EPS growth")

    if assumptions.get("business_type") in ("cyclical", "commodity", "exchange-platform"):
        score += 6; reasons.append(f"{assumptions.get('business_type')} business risk")

    if momentum and momentum.get("rating") == "Bearish":
        score += 8; reasons.append("Bearish price momentum overlay")
    if revision and revision.get("rating") == "Downgrade Risk":
        score += 8; reasons.append("Earnings revision proxy is negative")

    dq = (data_quality or {}).get("score", 70)
    if dq < 60:
        score += 12; reasons.append("Low data completeness")

    score = float(np.clip(score, 0, 100))
    return {
        "score": round(score, 1),
        "rating": "High" if score >= 65 else "Medium" if score >= 40 else "Low",
        "reasons": reasons or ["No major model-level risk flags"],
    }


def technical_snapshot(data):
    """Simple price-position overlay for summary cards."""
    market_data = data.get("market_data", {}) or {}
    ret_1y = _safe_float(market_data.get("return_1y_pct"), np.nan)
    vol_1y = _safe_float(market_data.get("volatility_1y_pct"), np.nan)
    beta = _safe_float(market_data.get("beta_vs_nifty"), np.nan)
    price = _safe_float(data.get("current_price"), np.nan)
    high = _safe_float(data.get("high_52w"), np.nan)
    low = _safe_float(data.get("low_52w"), np.nan)
    if np.isnan(price) or np.isnan(high) or np.isnan(low) or high <= low:
        return {
            "rating": "Unavailable",
            "position_52w_pct": np.nan,
            "return_1y_pct": ret_1y,
            "volatility_1y_pct": vol_1y,
            "beta_vs_nifty": beta,
            "note": "52-week data unavailable",
        }

    pos = (price - low) / (high - low) * 100
    rating = "Extended" if pos >= 80 else "Depressed" if pos <= 25 else "Neutral"
    return {
        "rating": rating,
        "position_52w_pct": round(float(pos), 1),
        "return_1y_pct": round(float(ret_1y), 1) if not np.isnan(ret_1y) else np.nan,
        "volatility_1y_pct": round(float(vol_1y), 1) if not np.isnan(vol_1y) else np.nan,
        "beta_vs_nifty": round(float(beta), 2) if not np.isnan(beta) else np.nan,
        "note": f"Price is {pos:.1f}% through its 52-week range",
    }


def peer_ranking(peer_df, data=None, assumptions=None):
    """Rank target and peers by quality, growth, valuation, leverage and similarity."""
    if peer_df is None or peer_df.empty:
        return pd.DataFrame()
    df = peer_df.copy()
    if "_is_target" in df.columns:
        df = df[df["_is_target"].isin([True, False])].copy()
    if df.empty:
        return df

    for col in ["ROE %", "ROCE %", "Revenue CAGR 3Y %", "P/E", "D/E", "Mkt Cap (Cr)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_row = df[df.get("_is_target") == True].head(1) if "_is_target" in df.columns else pd.DataFrame()
    target_mcap = _safe_float(target_row["Mkt Cap (Cr)"].iloc[0], np.nan) if not target_row.empty and "Mkt Cap (Cr)" in target_row else np.nan

    score = pd.Series(50.0, index=df.index)
    for col, weight in [("ROE %", 0.22), ("ROCE %", 0.22), ("Revenue CAGR 3Y %", 0.22)]:
        if col in df.columns and df[col].notna().any():
            score += (df[col].rank(pct=True) - 0.5) * 100 * weight
    for col, weight in [("P/E", 0.13), ("D/E", 0.09)]:
        if col in df.columns and df[col].notna().any():
            score += (df[col].rank(pct=True, ascending=False) - 0.5) * 100 * weight

    sim = pd.Series(50.0, index=df.index)
    if "Mkt Cap (Cr)" in df.columns and not np.isnan(target_mcap) and target_mcap > 0:
        diff = (np.log(df["Mkt Cap (Cr)"].replace(0, np.nan)) - np.log(target_mcap)).abs()
        sim = (100 - diff.fillna(diff.max() if diff.notna().any() else 1) * 25).clip(0, 100)
    df["Similarity Score"] = sim.round(1)
    df["Peer Score"] = (score * 0.8 + sim * 0.2).clip(0, 100).round(1)
    return df.sort_values("Peer Score", ascending=False)


def source_reliability_score(data, r, peer_df=None, assumptions=None, data_quality=None, sector_valuation=None):
    """Translate source completeness and model fit into signal reliability."""
    dq_score = _safe_float((data_quality or {}).get("score"), 50)
    score = dq_score
    notes = []

    source_status = data.get("source_status") or {}
    if source_status.get("Yahoo Finance") == "ok":
        notes.append("Yahoo price history available")
    else:
        score -= 8; notes.append("Yahoo market data unavailable or partial")

    if peer_df is not None and len(peer_df) > 2:
        notes.append("Peer set available")
    else:
        score -= 8; notes.append("Peer set is thin")

    if data.get("quarters") is not None and not data.get("quarters").empty:
        notes.append("Quarterly results integrated")
    else:
        score -= 8; notes.append("Quarterly table missing")

    exchange_status = source_status.get("Exchange events")
    nse_status = source_status.get("NSE India")
    bse_status = source_status.get("BSE India")
    if exchange_status == "ok":
        notes.append("NSE/BSE corporate actions and announcements available")
    elif exchange_status == "partial" or "ok" in (nse_status, bse_status):
        score -= 3; notes.append("Exchange events partially available")
    else:
        score -= 5; notes.append("NSE/BSE exchange events unavailable")

    if sector_valuation and not np.isnan(_safe_float(sector_valuation.get("target_price"))):
        notes.append(f"Sector model selected: {sector_valuation.get('sector_model')}")
    else:
        score -= 6; notes.append("Sector valuation model unavailable")

    if data.get("url", "").endswith("/consolidated/"):
        notes.append("Consolidated Screener page used")
    else:
        score -= 5; notes.append("Standalone data may be used")

    score = float(np.clip(score, 0, 100))
    return {
        "score": round(score, 1),
        "rating": "High" if score >= 80 else "Medium" if score >= 55 else "Low",
        "notes": notes,
    }


def signal_confidence(dcf_result, mc_result, data_quality, risk, reliability=None, momentum=None, revision=None, backtest=None):
    """Confidence score based on model agreement, data quality, risk and validation evidence."""
    score = 50.0
    notes = []
    dq = data_quality.get("score", 60)
    score += (dq - 60) * 0.20

    rel = _safe_float((reliability or {}).get("score"), np.nan)
    if not np.isnan(rel):
        score += (rel - 60) * 0.15

    if dcf_result:
        comp = abs(_safe_float(dcf_result.get("composite_score"), 0))
        score += min(comp * 30, 18)
        notes.append("Composite signal strength included")

    if mc_result and not mc_result.get("error"):
        prob = _safe_float(mc_result.get("prob_undervalued"), 50)
        signal = (dcf_result or {}).get("signal", "")
        if signal in ("BUY", "STRONG BUY") and prob >= 60:
            score += 12; notes.append("Monte Carlo supports upside")
        elif signal == "SELL" and prob <= 35:
            score += 12; notes.append("Monte Carlo supports downside")
        elif signal in ("BUY", "STRONG BUY", "SELL"):
            score -= 10; notes.append("Monte Carlo does not fully agree")

    if momentum and momentum.get("rating") == "Bullish":
        score += 5; notes.append("Momentum overlay is bullish")
    elif momentum and momentum.get("rating") == "Bearish":
        score -= 5; notes.append("Momentum overlay is bearish")

    if revision and revision.get("rating") == "Upgrade Bias":
        score += 5; notes.append("Earnings revision proxy is positive")
    elif revision and revision.get("rating") == "Downgrade Risk":
        score -= 5; notes.append("Earnings revision proxy is negative")

    proxy = (backtest or {}).get("price_proxy", {})
    hit = _safe_float(proxy.get("buy_12m_hit_rate_pct"), np.nan)
    validation = (backtest or {}).get("historical_validation", {})
    v_summary = validation.get("summary")
    if v_summary is not None and hasattr(v_summary, "empty") and not v_summary.empty:
        buy_row = v_summary[v_summary.get("Signal") == "BUY"] if "Signal" in v_summary else pd.DataFrame()
        if not buy_row.empty:
            hit = _safe_float(buy_row["12M Hit Rate %"].iloc[0], hit)
    if not np.isnan(hit):
        score += (hit - 50) * 0.15
        notes.append(f"Historical validation BUY 12M hit rate {hit:.1f}%")

    score -= max(risk.get("score", 50) - 45, 0) * 0.25
    score = float(np.clip(score, 0, 100))
    return {
        "score": round(score, 1),
        "rating": "High" if score >= 70 else "Medium" if score >= 50 else "Low",
        "notes": notes,
    }


def backtest_readiness(data_quality, years_available, price_history_available=False):
    """Report whether enough history exists for reliable signal backtesting."""
    if years_available >= 5 and data_quality.get("score", 0) >= 70 and price_history_available:
        status = "Ready"
        note = "Enough financial and price history for rolling signal tests"
    elif years_available >= 3:
        status = "Partial"
        note = "Enough for a limited backtest, but not a full-cycle test"
    else:
        status = "Weak"
        note = "Insufficient historical depth for reliable backtesting"
    return {"status": status, "note": note}


def historical_fundamental_backtest_proxy(r):
    """Offline proxy: checks whether revenue direction was followed by earnings direction."""
    revenue = _valid(r.get("revenue", []))
    eps = _valid(r.get("eps", []))
    pat = _valid(r.get("pat", []))
    if len(revenue) < 4 or (len(eps) < 4 and len(pat) < 4):
        return {"status": "Unavailable", "hit_rate": np.nan, "note": "Not enough history for even a fundamental proxy test"}

    earnings = eps if len(eps) >= 4 else pat
    hits = []
    for i in range(1, min(len(revenue), len(earnings)) - 1):
        rev_growth = (revenue[i] - revenue[i - 1]) / abs(revenue[i - 1]) if revenue[i - 1] else np.nan
        next_earnings_growth = (earnings[i + 1] - earnings[i]) / abs(earnings[i]) if earnings[i] else np.nan
        if not np.isnan(rev_growth) and not np.isnan(next_earnings_growth):
            hits.append(next_earnings_growth > 0 if rev_growth > 0 else next_earnings_growth <= 0)

    if not hits:
        return {"status": "Unavailable", "hit_rate": np.nan, "note": "Proxy test could not be computed"}
    hit_rate = sum(hits) / len(hits) * 100
    return {
        "status": "Computed",
        "hit_rate": round(float(hit_rate), 1),
        "observations": len(hits),
        "note": "Offline proxy: checks whether revenue direction was followed by next-period earnings direction",
    }


def historical_price_backtest_proxy(data):
    """
    Rolling price-signal proxy using only historical price data.

    This is not a point-in-time fundamental backtest. It validates the momentum
    overlay's historical usefulness by testing BUY/HOLD/SELL proxy states every
    quarter and measuring 3M/6M/12M forward returns.
    """
    hist = (data.get("market_data") or {}).get("price_history")
    if hist is None or not hasattr(hist, "empty") or hist.empty or "Close" not in hist:
        return {"status": "Unavailable", "summary": pd.DataFrame(), "trades": pd.DataFrame(), "note": "Price history unavailable"}

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 300:
        return {"status": "Weak", "summary": pd.DataFrame(), "trades": pd.DataFrame(), "note": "Need about 300 trading days for 12M forward tests"}

    rows = []
    for i in range(200, len(close) - 252, 63):
        price = close.iloc[i]
        dma50 = close.iloc[:i + 1].rolling(50).mean().iloc[-1]
        dma200 = close.iloc[:i + 1].rolling(200).mean().iloc[-1]
        mom6 = close.iloc[i] / close.iloc[i - 126] - 1 if i >= 126 and close.iloc[i - 126] else np.nan
        if price > dma50 and price > dma200 and mom6 > 0.05:
            signal = "BUY"
        elif price < dma200 and mom6 < -0.05:
            signal = "SELL"
        else:
            signal = "HOLD"
        rows.append({
            "Index": i,
            "Signal": signal,
            "3M Return %": (close.iloc[i + 63] / price - 1) * 100 if i + 63 < len(close) else np.nan,
            "6M Return %": (close.iloc[i + 126] / price - 1) * 100 if i + 126 < len(close) else np.nan,
            "12M Return %": (close.iloc[i + 252] / price - 1) * 100 if i + 252 < len(close) else np.nan,
        })

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {"status": "Unavailable", "summary": pd.DataFrame(), "trades": trades, "note": "No test windows available"}

    summary_rows = []
    for sig, group in trades.groupby("Signal"):
        summary_rows.append({
            "Signal": sig,
            "Observations": len(group),
            "Median 3M Return %": group["3M Return %"].median(),
            "Median 6M Return %": group["6M Return %"].median(),
            "Median 12M Return %": group["12M Return %"].median(),
            "12M Hit Rate %": (group["12M Return %"] > 0).mean() * 100,
        })
    summary = pd.DataFrame(summary_rows)
    buy_hit = np.nan
    if not summary.empty and (summary["Signal"] == "BUY").any():
        buy_hit = float(summary.loc[summary["Signal"] == "BUY", "12M Hit Rate %"].iloc[0])
    return {
        "status": "Computed",
        "summary": summary.round(1),
        "trades": trades.round(1),
        "buy_12m_hit_rate_pct": round(buy_hit, 1) if not np.isnan(buy_hit) else np.nan,
        "note": "Proxy test uses historical momentum states; full audited backtesting still requires point-in-time fundamentals",
    }


def explain_signal(data, r, dcf_result, mc_result, target, risk, quality, quarterly, valuation, revision, momentum, reliability, sector_valuation):
    """Generate top reasons and top risks for the final signal."""
    reasons = []
    risks = []

    roce = _safe_float(last_valid(r.get("roce", [])), np.nan)
    roe = _safe_float(last_valid(r.get("roe", [])), np.nan)
    de = _safe_float(last_valid(r.get("debt_equity", [])), np.nan)
    eps_growth = _safe_float(r.get("eps_cagr_3y"), np.nan)
    upside = _safe_float(target.get("upside_pct"), np.nan)
    mc_prob = _safe_float((mc_result or {}).get("prob_undervalued"), np.nan)

    if not np.isnan(upside):
        (reasons if upside > 15 else risks).append(f"Blended target implies {upside:.1f}% upside" if upside > 15 else f"Blended target implies only {upside:.1f}% upside")
    if not np.isnan(roce) and roce > 25:
        reasons.append(f"ROCE is strong at {roce:.1f}%")
    elif not np.isnan(roce) and roce < 10:
        risks.append(f"ROCE is weak at {roce:.1f}%")
    if not np.isnan(roe) and roe > 18:
        reasons.append(f"ROE is healthy at {roe:.1f}%")
    if not np.isnan(eps_growth):
        (reasons if eps_growth > 15 else risks if eps_growth < 0 else reasons).append(f"3Y EPS CAGR is {eps_growth:.1f}%")
    if not np.isnan(de):
        (reasons if de < 0.5 else risks if de > 1.2 else reasons).append(f"Debt/equity is {de:.2f}x")
    if not np.isnan(mc_prob):
        (reasons if mc_prob >= 60 else risks if mc_prob <= 40 else reasons).append(f"Monte Carlo undervaluation probability is {mc_prob:.1f}%")
    if quarterly and quarterly.get("rating") == "Improving":
        reasons.append("Latest quarterly trend is improving")
    elif quarterly and quarterly.get("rating") == "Weakening":
        risks.append("Latest quarterly trend is weakening")
    if valuation and valuation.get("rating") == "Expensive vs history":
        risks.append("Current price is high versus its historical band")
    elif valuation and valuation.get("rating") == "Discounted vs history":
        reasons.append("Current price is discounted versus its historical band")
    if revision and revision.get("rating") == "Upgrade Bias":
        reasons.append("Earnings revision proxy shows upgrade bias")
    elif revision and revision.get("rating") == "Downgrade Risk":
        risks.append("Earnings revision proxy shows downgrade risk")
    if momentum and momentum.get("rating") == "Bullish":
        reasons.append("Momentum overlay is bullish")
    elif momentum and momentum.get("rating") == "Bearish":
        risks.append("Momentum overlay is bearish")
    if reliability and reliability.get("rating") == "Low":
        risks.append("Low source/model reliability reduces confidence")
    if sector_valuation and sector_valuation.get("sector_model"):
        reasons.append(f"Sector-specific model used: {sector_valuation.get('sector_model')}")

    for reason in risk.get("reasons", []) or []:
        if reason not in risks:
            risks.append(reason)

    return {
        "top_reasons": reasons[:5] or ["No strong positive thesis drivers from available data"],
        "top_risks": risks[:5] or ["No major negative thesis drivers from available data"],
    }


def build_market_ready_report(data, r, flags, peer_df, dcf_result, mc_result, assumptions):
    """One-call report builder used by the Streamlit app."""
    dq = evaluate_data_quality(data, r, peer_df)
    quarterly = quarterly_snapshot(data)
    forecast = forecast_financials(data, r, assumptions, quarterly=quarterly)
    sector_val = sector_specific_valuation(data, r, assumptions, dcf_result, forecast, quarterly)
    target = blended_target_price(data, r, dcf_result, peer_df, sector_val)
    valuation = valuation_bands(data, r)
    revision = earnings_revision_signal(r, quarterly)
    momentum = price_momentum_overlay(data)
    tech = technical_snapshot(data)
    ranking = peer_ranking(peer_df, data, assumptions)
    reliability = source_reliability_score(data, r, peer_df, assumptions, dq, sector_val)
    risk = risk_score(data, r, flags, assumptions, dq, momentum, revision)
    price_bt = historical_price_backtest_proxy(data)
    validation = run_historical_validation(data, r)
    backtest = backtest_readiness(
        dq,
        len(_valid(r.get("revenue", []))),
        price_history_available=validation.get("status") == "Computed",
    )
    backtest["fundamental_proxy"] = historical_fundamental_backtest_proxy(r)
    backtest["price_proxy"] = price_bt
    backtest["historical_validation"] = validation
    confidence = signal_confidence(dcf_result, mc_result, dq, risk, reliability, momentum, revision, backtest)
    explain = explain_signal(data, r, dcf_result, mc_result, target, risk, dq, quarterly, valuation, revision, momentum, reliability, sector_val)

    return {
        "data_quality": dq,
        "quarterly": quarterly,
        "forecast": forecast,
        "sector_valuation": sector_val,
        "target": target,
        "valuation_bands": valuation,
        "earnings_revision": revision,
        "momentum": momentum,
        "technical": tech,
        "peer_ranking": ranking,
        "source_reliability": reliability,
        "confidence": confidence,
        "risk": risk,
        "backtest": backtest,
        "explainability": explain,
    }
