"""
dcf.py — DCF Valuation Engine with:
  - Auto-derived assumptions from historical financial data
  - 5-year FCFF projection (Bull / Base / Bear scenarios)
  - WACC calculation (CAPM + after-tax cost of debt, using actual D/E)
  - Terminal value (Gordon Growth Model)
  - Monte Carlo simulation (10,000 iterations)
  - Sensitivity analysis (tornado chart data)
  - Composite Buy / Hold / Sell signal (DCF + P/E reversion + EV/EBITDA)
"""

import numpy as np
import pandas as pd
from ratios import last_valid


# ─────────────────────────────────────────────
# HISTORY-DERIVED ASSUMPTION ENGINE
# ─────────────────────────────────────────────

def derive_assumptions_from_history(data: dict, r: dict) -> dict:
    """
    Analyze 5 years of historical data to auto-derive DCF assumptions.

    Logic:
      - Revenue growth  → weighted avg of 3Y and 5Y CAGR (3Y gets 60% weight,
                          more recent trend matters more), then mean-revert toward
                          long-run GDP+inflation (~10%) so outlier years don't
                          produce absurd projections.
      - EBITDA margin   → median of last 5 years of operating_margin (robust to
                          one-off spikes).
      - Capex % rev     → median of (capex / revenue) over available years.
      - WACC            → CAPM using actual D/E ratio from latest BS.  Beta is
                          estimated from D/E via Hamada equation (unlevered β=0.8
                          is a reasonable mid-market Indian equity prior).
      - Terminal growth → clipped to [3%, 6%]; set as min(base_growth/2, 5%).
      - Margin delta    → (latest_margin - oldest_margin) / n_years — trend slope.
      - Bear/Bull       → ±40% of base growth, capped to sensible bounds.
    """
    rev      = r.get("revenue", [])
    ebitda_m = r.get("operating_margin", [])
    capex    = r.get("capex", [])
    cfo      = r.get("cfo", [])
    debt     = r.get("total_debt", [])
    equity   = r.get("equity", [])

    n = r.get("n_years", 5)

    def valid_floats(lst):
        out = []
        for v in lst:
            try:
                f = float(v)
                if not np.isnan(f):
                    out.append(f)
            except:
                pass
        return out

    # ── Revenue growth ──
    cagr3 = r.get("revenue_cagr_3y", np.nan)
    cagr5 = r.get("revenue_cagr_5y", np.nan)
    c3 = float(cagr3) if (cagr3 is not None and not np.isnan(float(cagr3 if cagr3 is not None else np.nan))) else None
    c5 = float(cagr5) if (cagr5 is not None and not np.isnan(float(cagr5 if cagr5 is not None else np.nan))) else None

    if c3 is not None and c5 is not None:
        raw_growth = (0.60 * c3 + 0.40 * c5) / 100   # weighted blend, convert % → decimal
    elif c3 is not None:
        raw_growth = c3 / 100
    elif c5 is not None:
        raw_growth = c5 / 100
    else:
        raw_growth = 0.10  # fallback: 10%

    # Mean-revert: blend 70% historical, 30% long-run 10% anchor
    base_growth = 0.70 * raw_growth + 0.30 * 0.10
    base_growth = float(np.clip(base_growth, 0.03, 0.35))

    # ── EBITDA / operating margin (use median for robustness) ──
    valid_margins = valid_floats(ebitda_m)
    if valid_margins:
        base_ebitda_margin_pct = float(np.median(valid_margins))
    else:
        base_ebitda_margin_pct = 15.0
    base_ebitda_margin_pct = float(np.clip(base_ebitda_margin_pct, 3.0, 60.0))

    # Margin trend slope (pp per year)
    if len(valid_margins) >= 3:
        margin_delta = (valid_margins[-1] - valid_margins[0]) / (len(valid_margins) - 1) / 100
        margin_delta = float(np.clip(margin_delta, -0.02, 0.02))
    else:
        margin_delta = 0.002

    # ── Capex % revenue ──
    rev_vals    = valid_floats(rev)
    capex_vals  = valid_floats(capex)
    capex_ratios = []
    for c_v, r_v in zip(capex_vals[-n:], rev_vals[-n:]):
        if r_v > 0:
            capex_ratios.append(abs(c_v) / r_v)
    if capex_ratios:
        capex_pct = float(np.clip(np.median(capex_ratios), 0.02, 0.25))
    else:
        capex_pct = 0.07

    # ── Working capital % revenue (from CFO vs PAT difference as proxy) ──
    pat_vals = valid_floats(r.get("pat", []))
    cfo_vals = valid_floats(cfo)
    wc_changes = []
    for cf_v, pt_v, rv in zip(cfo_vals[-n:], pat_vals[-n:], rev_vals[-n:]):
        if rv > 0 and pt_v != 0:
            # WC change ≈ PAT - CFO (simplified: non-cash adjustments excluded)
            wc_change = abs(pt_v - cf_v)
            wc_changes.append(wc_change / rv)
    wc_pct = float(np.clip(np.median(wc_changes), 0.005, 0.05)) if wc_changes else 0.015

    # ── WACC via Hamada equation ──
    debt_vals   = valid_floats(debt)
    equity_vals = valid_floats(equity)
    latest_debt   = debt_vals[-1]   if debt_vals   else 0.0
    latest_equity = equity_vals[-1] if equity_vals else 1.0
    de_ratio = latest_debt / latest_equity if latest_equity > 0 else 0.5

    # Hamada: levered_beta = unlevered_beta * (1 + (1 - tax_rate) * D/E)
    unlevered_beta = 0.80   # reasonable Indian mid-cap prior
    tax_rate_est   = 0.25
    levered_beta   = unlevered_beta * (1 + (1 - tax_rate_est) * de_ratio)
    levered_beta   = float(np.clip(levered_beta, 0.5, 2.5))

    rfr = 0.07           # 10Y Gsec
    erp = 0.055          # India ERP
    cost_of_equity = rfr + levered_beta * erp

    # Cost of debt: estimate from interest / total debt
    interest_vals = valid_floats(r.get("interest", []))
    if interest_vals and debt_vals and debt_vals[-1] > 0:
        raw_cod = interest_vals[-1] / debt_vals[-1]
        cost_of_debt = float(np.clip(raw_cod, 0.06, 0.16))
    else:
        cost_of_debt = 0.09

    market_cap = float(data.get("market_cap") or 0)
    total_cap  = market_cap + latest_debt
    w_eq = market_cap / total_cap if total_cap > 0 else 0.8
    w_dt = 1 - w_eq
    base_wacc = w_eq * cost_of_equity + w_dt * cost_of_debt * (1 - tax_rate_est)
    base_wacc = float(np.clip(base_wacc, 0.07, 0.18))

    # ── Terminal growth ──
    # Conservative: half of base growth, bounded between 3% and 6%
    base_tgr = float(np.clip(base_growth / 2, 0.03, 0.06))

    # ── Bull / Bear scenarios ──
    bear_growth = float(np.clip(base_growth * 0.50, 0.02, 0.15))
    bull_growth = float(np.clip(base_growth * 1.60, base_growth + 0.03, 0.40))

    return {
        # Core growth
        "base_growth":        round(base_growth, 4),
        "bear_growth":        round(bear_growth, 4),
        "bull_growth":        round(bull_growth, 4),
        # Margins
        "base_ebitda_margin": round(base_ebitda_margin_pct, 2),   # in %
        "base_margin_delta":  round(margin_delta, 4),
        "bear_margin_delta":  round(margin_delta - 0.005, 4),
        "bull_margin_delta":  round(margin_delta + 0.005, 4),
        # Capital structure / WACC
        "base_wacc":          round(base_wacc, 4),
        "beta":               round(levered_beta, 2),
        "risk_free_rate":     rfr,
        "erp":                erp,
        "cost_of_debt":       round(cost_of_debt, 4),
        "tax_rate":           tax_rate_est,
        # Capex / WC
        "capex_pct":          round(capex_pct, 4),
        "wc_pct":             round(wc_pct, 4),
        # Terminal growth
        "base_tgr":           round(base_tgr, 4),
        "bear_tgr":           round(max(base_tgr - 0.01, 0.02), 4),
        "bull_tgr":           round(min(base_tgr + 0.01, 0.06), 4),
        # For display
        "de_ratio":           round(de_ratio, 2),
        "raw_growth_pct":     round(raw_growth * 100, 2),
    }


# ─────────────────────────────────────────────
# WACC CALCULATOR
# ─────────────────────────────────────────────

def calculate_wacc(
    market_cap: float,
    total_debt: float,
    beta: float = 1.0,
    risk_free_rate: float = 0.07,
    equity_risk_premium: float = 0.055,
    cost_of_debt: float = 0.09,
    tax_rate: float = 0.25,
) -> dict:
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    total_capital  = market_cap + total_debt
    if total_capital == 0:
        return {"wacc": 0.11, "cost_of_equity": cost_of_equity,
                "cost_of_debt_after_tax": cost_of_debt * (1 - tax_rate),
                "equity_weight": 1.0, "debt_weight": 0.0}

    w_equity = market_cap / total_capital
    w_debt   = total_debt / total_capital
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc = (w_equity * cost_of_equity) + (w_debt * after_tax_cost_of_debt)

    return {
        "wacc":                  round(wacc, 4),
        "cost_of_equity":        round(cost_of_equity, 4),
        "cost_of_debt_after_tax":round(after_tax_cost_of_debt, 4),
        "equity_weight":         round(w_equity, 4),
        "debt_weight":           round(w_debt, 4),
        "beta":                  beta,
        "risk_free_rate":        risk_free_rate,
        "erp":                   equity_risk_premium,
    }


# ─────────────────────────────────────────────
# FCF PROJECTION
# ─────────────────────────────────────────────

def project_fcff(
    base_revenue: float,
    base_ebitda_margin: float,      # decimal (e.g. 0.18)
    revenue_growth: float,
    margin_improvement: float,
    capex_pct_revenue: float,
    wc_change_pct_revenue: float,
    tax_rate: float,
    n_years: int = 5,
) -> pd.DataFrame:
    rows = []
    revenue       = base_revenue
    ebitda_margin = base_ebitda_margin

    for yr in range(1, n_years + 1):
        revenue       = revenue * (1 + revenue_growth)
        ebitda_margin = min(ebitda_margin + margin_improvement, 0.60)
        ebitda  = revenue * ebitda_margin
        nopat   = ebitda * (1 - tax_rate)
        capex   = revenue * capex_pct_revenue
        delta_wc= revenue * wc_change_pct_revenue
        fcff    = nopat - capex - delta_wc

        rows.append({
            "Year":             f"FY+{yr}",
            "Revenue":          round(revenue, 1),
            "EBITDA Margin %":  round(ebitda_margin * 100, 1),
            "EBITDA":           round(ebitda, 1),
            "NOPAT":            round(nopat, 1),
            "Capex":            round(capex, 1),
            "ΔWorking Capital": round(delta_wc, 1),
            "FCFF":             round(fcff, 1),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# DCF VALUATION (SINGLE SCENARIO)
# ─────────────────────────────────────────────

def dcf_valuation(
    fcff_df: pd.DataFrame,
    wacc: float,
    terminal_growth: float,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
) -> dict:
    fcffs = fcff_df["FCFF"].tolist()
    n     = len(fcffs)

    pv_fcffs    = [fcff / ((1 + wacc) ** (i + 1)) for i, fcff in enumerate(fcffs)]
    sum_pv_fcff = sum(pv_fcffs)

    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.01
    terminal_value    = terminal_fcff / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)

    enterprise_value    = sum_pv_fcff + pv_terminal_value
    equity_value        = max(enterprise_value - total_debt + cash, 0)
    intrinsic_per_share = (equity_value / shares_outstanding) if shares_outstanding > 0 else np.nan

    return {
        "pv_fcffs":           [round(v, 1) for v in pv_fcffs],
        "sum_pv_fcff":        round(sum_pv_fcff, 1),
        "terminal_value":     round(terminal_value, 1),
        "pv_terminal_value":  round(pv_terminal_value, 1),
        "enterprise_value":   round(enterprise_value, 1),
        "equity_value":       round(equity_value, 1),
        "intrinsic_per_share":round(intrinsic_per_share, 2) if not np.isnan(intrinsic_per_share) else np.nan,
        "tv_pct_of_ev":       round(pv_terminal_value / enterprise_value * 100, 1) if enterprise_value else np.nan,
    }


# ─────────────────────────────────────────────
# COMPOSITE VALUATION SIGNAL
# ─────────────────────────────────────────────

def compute_composite_signal(
    current_price: float,
    base_iv: float,            # DCF intrinsic value
    r: dict,                   # ratios dict
    data: dict,                # raw scraped data
) -> dict:
    """
    Three-factor composite signal, each scored -1 (bearish) to +1 (bullish):

      Factor 1 — DCF Upside (weight 50%)
        score = clamp((IV/CMP - 1) / 0.30, -1, +1)
        i.e. ±30% upside/downside maps to full score

      Factor 2 — P/E Mean Reversion (weight 25%)
        Compare current P/E to 5-year median implied P/E (PAT CAGR adjusted).
        Uses: median historical EPS growth to project "fair P/E" via PEG=1 anchor.
        score = clamp((fair_pe - current_pe) / fair_pe, -1, +1)

      Factor 3 — EV/EBITDA vs Historical (weight 25%)
        Compare current EV/EBITDA to median of last 5Y (approximated via
        EBITDA trend). Lower current ratio = undervalued.
        score = clamp((hist_ev_ebitda - current_ev_ebitda) / hist_ev_ebitda, -1, +1)

    Final composite score → BUY (>0.20), SELL (<-0.20), HOLD otherwise.
    """
    signals = {}

    # ── Factor 1: DCF upside ──
    dcf_score = 0.0
    dcf_upside_pct = np.nan
    if (base_iv and not np.isnan(base_iv) and
            current_price and not np.isnan(current_price) and current_price > 0):
        dcf_upside_pct = (base_iv - current_price) / current_price * 100
        dcf_score = float(np.clip((base_iv / current_price - 1) / 0.30, -1.0, 1.0))
    signals["dcf"] = {"score": round(dcf_score, 3), "upside_pct": round(dcf_upside_pct, 1)
                       if not np.isnan(dcf_upside_pct) else np.nan}

    # ── Factor 2: P/E mean reversion ──
    pe_score = 0.0
    current_pe = data.get("pe_ratio")
    eps_cagr3  = r.get("eps_cagr_3y")

    if (current_pe and eps_cagr3 is not None and
            not np.isnan(float(current_pe if current_pe else np.nan)) and
            not np.isnan(float(eps_cagr3))):
        current_pe   = float(current_pe)
        eps_growth   = float(eps_cagr3)
        # PEG = 1 anchor: fair P/E = max(10, min(40, EPS growth rate))
        # (EPS growth in %, so 15% growth → fair P/E ≈ 15-20)
        # We use 1.2× the EPS CAGR as a reasonable Indian market premium
        fair_pe = float(np.clip(eps_growth * 1.2, 8.0, 45.0))
        if fair_pe > 0 and current_pe > 0:
            pe_score = float(np.clip((fair_pe - current_pe) / fair_pe, -1.0, 1.0))
        signals["pe"] = {"score": round(pe_score, 3),
                         "current_pe": round(current_pe, 1),
                         "fair_pe": round(fair_pe, 1)}
    else:
        signals["pe"] = {"score": 0.0, "note": "insufficient data"}

    # ── Factor 3: EV/EBITDA vs historical trend ──
    ev_score = 0.0
    current_ev_ebitda = r.get("ev_ebitda")
    ebitda_vals = [v for v in r.get("ebitda", [])
                   if v is not None and not np.isnan(float(v))]

    market_cap  = float(data.get("market_cap") or 0)
    latest_debt = float(last_valid(r.get("total_debt", [])) or 0)
    latest_cash = float(last_valid(r.get("cash", [])) or 0)
    ev_now      = market_cap + latest_debt - latest_cash

    if len(ebitda_vals) >= 3 and ev_now > 0:
        # Approximate historical EV by assuming market cap scaled with EBITDA growth
        # (simplification: use median EBITDA to estimate a "normalised" EV/EBITDA)
        median_ebitda = float(np.median(ebitda_vals))
        latest_ebitda = float(ebitda_vals[-1])
        # Implied historical EV/EBITDA using current EV but median EBITDA
        hist_ev_ebitda_implied = ev_now / median_ebitda if median_ebitda > 0 else np.nan
        curr_ev_ebitda_calc    = ev_now / latest_ebitda  if latest_ebitda > 0 else np.nan

        if (not np.isnan(hist_ev_ebitda_implied) and
                not np.isnan(curr_ev_ebitda_calc) and
                hist_ev_ebitda_implied > 0):
            ev_score = float(np.clip(
                (hist_ev_ebitda_implied - curr_ev_ebitda_calc) / hist_ev_ebitda_implied,
                -1.0, 1.0,
            ))
        signals["ev_ebitda"] = {
            "score":               round(ev_score, 3),
            "current_ev_ebitda":   round(curr_ev_ebitda_calc, 1) if not np.isnan(curr_ev_ebitda_calc) else np.nan,
            "hist_ev_ebitda_norm": round(hist_ev_ebitda_implied, 1) if not np.isnan(hist_ev_ebitda_implied) else np.nan,
        }
    else:
        signals["ev_ebitda"] = {"score": 0.0, "note": "insufficient data"}

    # ── Composite score (weighted) ──
    composite = (0.50 * dcf_score +
                 0.25 * pe_score  +
                 0.25 * ev_score)

    if composite > 0.20:
        signal       = "BUY"
        signal_color = "green"
    elif composite < -0.20:
        signal       = "SELL"
        signal_color = "red"
    else:
        signal       = "HOLD"
        signal_color = "orange"

    # Conviction level
    abs_c = abs(composite)
    if abs_c > 0.55:
        conviction = "Strong"
    elif abs_c > 0.30:
        conviction = "Moderate"
    else:
        conviction = "Weak"

    return {
        "signal":         signal,
        "signal_color":   signal_color,
        "composite_score":round(composite, 3),
        "conviction":     conviction,
        "upside_pct":     round(dcf_upside_pct, 1) if not np.isnan(dcf_upside_pct) else np.nan,
        "factor_scores":  signals,
    }


# ─────────────────────────────────────────────
# THREE SCENARIOS (BULL / BASE / BEAR)
# ─────────────────────────────────────────────

def run_three_scenarios(data: dict, r: dict, assumptions: dict) -> dict:
    """
    Run DCF for Bull, Base, Bear.  If assumptions contain 'auto_derived'=True
    they came from derive_assumptions_from_history(); otherwise user overrides.
    """
    base_revenue       = last_valid(r.get("revenue", []))
    # Use the history-derived base EBITDA margin (in %) from assumptions
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None  or np.isnan(float(total_debt  if total_debt  is not None else np.nan))) else float(total_debt)
    cash_val           = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None  or np.isnan(float(cash_val    if cash_val    is not None else np.nan))) else float(cash_val)
    market_cap         = data.get("market_cap") or np.nan
    current_price      = data.get("current_price") or np.nan

    shares = (market_cap / current_price) if (market_cap and current_price) else 100.0

    wacc_result = calculate_wacc(
        market_cap=market_cap or 0,
        total_debt=total_debt or 0,
        beta=assumptions.get("beta", 1.0),
        risk_free_rate=assumptions.get("risk_free_rate", 0.07),
        equity_risk_premium=assumptions.get("erp", 0.055),
        cost_of_debt=assumptions.get("cost_of_debt", 0.09),
        tax_rate=assumptions.get("tax_rate", 0.25),
    )
    wacc = wacc_result["wacc"]

    scenarios_params = {
        "Bear": {
            "revenue_growth":        assumptions.get("bear_growth", 0.05),
            "margin_improvement":    assumptions.get("bear_margin_delta", -0.005),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.08),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.02),
            "terminal_growth":       assumptions.get("bear_tgr", 0.03),
        },
        "Base": {
            "revenue_growth":        assumptions.get("base_growth", 0.12),
            "margin_improvement":    assumptions.get("base_margin_delta", 0.002),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.07),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.015),
            "terminal_growth":       assumptions.get("base_tgr", 0.04),
        },
        "Bull": {
            "revenue_growth":        assumptions.get("bull_growth", 0.20),
            "margin_improvement":    assumptions.get("bull_margin_delta", 0.008),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.06),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.01),
            "terminal_growth":       assumptions.get("bull_tgr", 0.05),
        },
    }

    results = {}
    for name, s in scenarios_params.items():
        fcff_df = project_fcff(
            base_revenue=base_revenue or 10000,
            base_ebitda_margin=base_ebitda_margin,
            revenue_growth=s["revenue_growth"],
            margin_improvement=s["margin_improvement"],
            capex_pct_revenue=s["capex_pct_revenue"],
            wc_change_pct_revenue=s["wc_change_pct_revenue"],
            tax_rate=assumptions.get("tax_rate", 0.25),
        )
        val = dcf_valuation(
            fcff_df=fcff_df,
            wacc=wacc,
            terminal_growth=s["terminal_growth"],
            total_debt=total_debt or 0,
            cash=cash_val or 0,
            shares_outstanding=shares,
        )
        results[name] = {**val, "fcff_df": fcff_df, "assumptions": s}

    base_iv = results["Base"]["intrinsic_per_share"]

    # ── Composite signal (history-driven, replaces simple upside check) ──
    composite = compute_composite_signal(
        current_price=current_price,
        base_iv=base_iv,
        r=r,
        data=data,
    )

    return {
        "scenarios":       results,
        "wacc_result":     wacc_result,
        "current_price":   current_price,
        "base_iv":         base_iv,
        "upside_pct":      composite["upside_pct"],
        "margin_of_safety":composite["upside_pct"],
        "signal":          composite["signal"],
        "signal_color":    composite["signal_color"],
        "composite_score": composite["composite_score"],
        "conviction":      composite["conviction"],
        "factor_scores":   composite["factor_scores"],
        "shares":          shares,
        "assumptions_used":assumptions,
    }


# ─────────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def run_monte_carlo(data: dict, r: dict, assumptions: dict, n_simulations: int = 10000) -> dict:
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None or np.isnan(float(total_debt  if total_debt  is not None else np.nan))) else float(total_debt) or 0
    cash_val           = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None or np.isnan(float(cash_val    if cash_val    is not None else np.nan))) else float(cash_val) or 0
    market_cap         = data.get("market_cap") or 100000
    current_price      = data.get("current_price") or 1000
    shares             = (market_cap / current_price) if current_price else 100

    base_growth = assumptions.get("base_growth", 0.12)
    base_tgr    = assumptions.get("base_tgr", 0.04)
    base_wacc   = assumptions.get("base_wacc",  assumptions.get("wacc", 0.11))
    capex_pct   = assumptions.get("capex_pct", 0.07)
    tax_rate    = assumptions.get("tax_rate", 0.25)

    # Std devs based on historical volatility where possible
    rev_vals = [v for v in r.get("revenue", [])
                if v is not None and not np.isnan(float(v))]
    if len(rev_vals) >= 3:
        yoy_growths = [(rev_vals[i+1] - rev_vals[i]) / abs(rev_vals[i])
                       for i in range(len(rev_vals)-1) if rev_vals[i] != 0]
        growth_std = float(np.clip(np.std(yoy_growths), 0.03, 0.12)) if yoy_growths else 0.06
    else:
        growth_std = 0.06

    np.random.seed(42)
    intrinsic_values = []

    for _ in range(n_simulations):
        rev_growth   = float(np.clip(np.random.normal(base_growth, growth_std), -0.10, 0.40))
        tgr          = float(np.clip(np.random.normal(base_tgr, 0.01), 0.01, 0.06))
        wacc         = float(np.clip(np.random.normal(base_wacc, 0.015), 0.06, 0.20))
        margin_delta = float(np.random.normal(assumptions.get("base_margin_delta", 0.002), 0.003))
        cap_pct      = float(np.clip(np.random.normal(capex_pct, 0.01), 0.02, 0.20))

        try:
            fcff_df = project_fcff(
                base_revenue=base_revenue,
                base_ebitda_margin=base_ebitda_margin,
                revenue_growth=rev_growth,
                margin_improvement=margin_delta,
                capex_pct_revenue=cap_pct,
                wc_change_pct_revenue=0.015,
                tax_rate=tax_rate,
            )
            val = dcf_valuation(fcff_df, wacc, tgr, total_debt, cash_val, shares)
            iv  = val["intrinsic_per_share"]
            if iv and not np.isnan(iv) and iv > 0:
                intrinsic_values.append(iv)
        except:
            pass

    if not intrinsic_values:
        return {"error": "Monte Carlo failed — insufficient data"}

    arr = np.array(intrinsic_values)
    return {
        "values":           arr.tolist(),
        "mean":             round(float(arr.mean()), 2),
        "median":           round(float(np.median(arr)), 2),
        "p10":              round(float(np.percentile(arr, 10)), 2),
        "p25":              round(float(np.percentile(arr, 25)), 2),
        "p75":              round(float(np.percentile(arr, 75)), 2),
        "p90":              round(float(np.percentile(arr, 90)), 2),
        "std":              round(float(arr.std()), 2),
        "n_simulations":    len(arr),
        "current_price":    current_price,
        "prob_undervalued": round(float((arr > current_price).mean() * 100), 1),
    }


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS (TORNADO CHART DATA)
# ─────────────────────────────────────────────

def run_sensitivity(data: dict, r: dict, assumptions: dict) -> pd.DataFrame:
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None or np.isnan(float(total_debt  if total_debt  is not None else np.nan))) else float(total_debt) or 0
    cash_val           = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None or np.isnan(float(cash_val    if cash_val    is not None else np.nan))) else float(cash_val) or 0
    market_cap         = data.get("market_cap") or 100000
    current_price      = data.get("current_price") or 1000
    shares             = (market_cap / current_price) if current_price else 100

    base_wacc   = assumptions.get("base_wacc", assumptions.get("wacc", 0.11))
    base_growth = assumptions.get("base_growth", 0.12)
    base_tgr    = assumptions.get("base_tgr", 0.04)
    capex_pct   = assumptions.get("capex_pct", 0.07)
    tax_rate    = assumptions.get("tax_rate", 0.25)

    def compute_iv(rev_growth, wacc, tgr, cap_pct, margin_delta=0.002):
        try:
            fcff_df = project_fcff(
                base_revenue=base_revenue,
                base_ebitda_margin=base_ebitda_margin,
                revenue_growth=rev_growth,
                margin_improvement=margin_delta,
                capex_pct_revenue=cap_pct,
                wc_change_pct_revenue=0.015,
                tax_rate=tax_rate,
            )
            val = dcf_valuation(fcff_df, wacc, tgr, total_debt, cash_val, shares)
            return val["intrinsic_per_share"] or np.nan
        except:
            return np.nan

    base_iv = compute_iv(base_growth, base_wacc, base_tgr, capex_pct)

    variables = [
        ("Revenue Growth",   base_growth, 0.05, lambda v: compute_iv(v, base_wacc, base_tgr, capex_pct)),
        ("WACC",             base_wacc,   0.02, lambda v: compute_iv(base_growth, v, base_tgr, capex_pct)),
        ("Terminal Growth",  base_tgr,    0.01, lambda v: compute_iv(base_growth, base_wacc, v, capex_pct)),
        ("Capex % Revenue",  capex_pct,   0.02, lambda v: compute_iv(base_growth, base_wacc, base_tgr, v)),
        ("Tax Rate",         tax_rate,    0.05, lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct)),
        ("EBITDA Margin Δ",  0.002,       0.005,lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct, v)),
    ]

    rows = []
    for name, base_val, delta, fn in variables:
        low_iv  = fn(base_val - delta)
        high_iv = fn(base_val + delta)
        impact  = abs((high_iv or base_iv) - (low_iv or base_iv))
        rows.append({
            "Variable":   name,
            "Base IV":    round(base_iv, 2) if not np.isnan(base_iv)  else np.nan,
            "Low IV":     round(low_iv,  2) if not np.isnan(low_iv)   else np.nan,
            "High IV":    round(high_iv, 2) if not np.isnan(high_iv)  else np.nan,
            "Impact":     round(impact,  2),
            "Low Label":  f"{name} - {delta*100:.0f}%",
            "High Label": f"{name} + {delta*100:.0f}%",
        })

    return pd.DataFrame(rows).sort_values("Impact", ascending=True)
