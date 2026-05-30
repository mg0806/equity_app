"""
validation_engine.py

Historical validation engine for EquityLens signal credibility.

The engine supports two operating modes:
1. Single-stock validation from the current report's available price history.
2. Batch validation by concatenating per-stock validation outputs.

True institutional point-in-time validation requires archived historical
fundamental snapshots. This engine is built so those snapshots can be plugged
in later through a signal_model callback, while still producing useful audited
price-return evidence from today's available data.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}


def _safe_float(value, default=np.nan):
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _clean_history(history):
    if history is None or not hasattr(history, "empty") or history.empty:
        return pd.DataFrame()
    df = history.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    if "Close" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.NaT
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


def _series_return(close, idx, days):
    if idx + days >= len(close) or close.iloc[idx] == 0:
        return np.nan
    return (close.iloc[idx + days] / close.iloc[idx] - 1) * 100


def _forward_max_drawdown(close, idx, days=252):
    if idx + 1 >= len(close):
        return np.nan
    end = min(idx + days, len(close) - 1)
    window = close.iloc[idx:end + 1]
    if window.empty or window.iloc[0] == 0:
        return np.nan
    running_peak = window.cummax()
    drawdown = (window / running_peak - 1) * 100
    return float(drawdown.min())


def _market_cap_bucket(market_cap):
    mcap = _safe_float(market_cap)
    if np.isnan(mcap):
        return "Unknown"
    if mcap >= 50000:
        return "Large Cap"
    if mcap >= 10000:
        return "Mid Cap"
    return "Small Cap"


def _market_regime(bench_close, idx):
    if bench_close is None or len(bench_close) <= idx or idx < 252:
        return "Unknown"
    current = bench_close.iloc[idx]
    ma200 = bench_close.iloc[:idx + 1].rolling(200).mean().iloc[-1]
    ret12 = bench_close.iloc[idx] / bench_close.iloc[idx - 252] - 1 if bench_close.iloc[idx - 252] else np.nan
    if not np.isnan(ret12) and current > ma200 and ret12 > 0.08:
        return "Bull"
    if not np.isnan(ret12) and current < ma200 and ret12 < -0.08:
        return "Bear"
    return "Sideways"


def default_historical_signal(stock_close, idx, benchmark_close=None):
    """
    Point-in-time market signal using only information available up to idx.

    This mirrors the app's momentum, valuation-band and drawdown overlay. When
    archived fundamentals become available, pass a signal_model callback to
    run_historical_validation() to replace this policy with full DCF/fundamental
    replay.
    """
    if idx < 252 or idx >= len(stock_close):
        return "HOLD", 0.0, "Insufficient lookback"

    trailing = stock_close.iloc[:idx + 1]
    price = trailing.iloc[-1]
    dma50 = trailing.rolling(50).mean().iloc[-1]
    dma200 = trailing.rolling(200).mean().iloc[-1]
    ret6 = price / stock_close.iloc[idx - 126] - 1 if stock_close.iloc[idx - 126] else np.nan
    ret12 = price / stock_close.iloc[idx - 252] - 1 if stock_close.iloc[idx - 252] else np.nan
    high252 = trailing.tail(252).max()
    drawdown = price / high252 - 1 if high252 else np.nan
    percentile = trailing.tail(min(len(trailing), 1260)).rank(pct=True).iloc[-1]

    rel_strength = np.nan
    if benchmark_close is not None and len(benchmark_close) > idx and idx >= 252 and benchmark_close.iloc[idx - 252]:
        bench_ret12 = benchmark_close.iloc[idx] / benchmark_close.iloc[idx - 252] - 1
        rel_strength = ret12 - bench_ret12 if not np.isnan(ret12) else np.nan

    score = 0.0
    score += 1.2 if price > dma50 else -1.0
    score += 1.5 if price > dma200 else -1.5
    score += 1.0 if not np.isnan(ret6) and ret6 > 0.08 else -0.8 if not np.isnan(ret6) and ret6 < -0.08 else 0
    score += 0.8 if not np.isnan(rel_strength) and rel_strength > 0 else -0.8 if not np.isnan(rel_strength) and rel_strength < -0.08 else 0
    score += 0.7 if not np.isnan(drawdown) and drawdown > -0.18 else -0.9 if not np.isnan(drawdown) and drawdown < -0.30 else 0
    score += 0.5 if percentile < 0.75 else -0.6 if percentile > 0.90 else 0

    if score >= 2.2:
        signal = "BUY"
    elif score <= -2.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    rationale = (
        f"score={score:.2f}; price {'>' if price > dma50 else '<='} 50DMA; "
        f"price {'>' if price > dma200 else '<='} 200DMA; "
        f"6M={ret6 * 100:.1f}%"
    )
    return signal, float(score), rationale


def run_historical_validation(data, r=None, signal_model=None, step_days=63, min_years=5):
    """
    Replay historical signal states and measure forward returns.

    Parameters
    ----------
    data : dict
        Canonical company object with market_data.price_history.
    r : dict
        Ratio dictionary. Used for sector/metadata only in this engine.
    signal_model : callable, optional
        Callback signature: signal_model(stock_close, idx, benchmark_close, data, r)
        returning (signal, score, rationale). This is where archived
        point-in-time fundamentals can be injected.
    step_days : int
        Test cadence. 63 trading days approximates quarterly signal review.
    min_years : int
        Preferred historical depth. The engine still runs with less history but
        marks evidence quality lower.
    """
    market = data.get("market_data") or {}
    hist = _clean_history(market.get("price_history"))
    bench = _clean_history(market.get("benchmark_history"))
    if hist.empty or len(hist) < 320:
        return {
            "status": "Unavailable",
            "evidence_quality": "Low",
            "headline": "Historical validation unavailable: insufficient price history.",
            "summary": pd.DataFrame(),
            "by_regime": pd.DataFrame(),
            "by_sector": pd.DataFrame(),
            "by_market_cap": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "note": "Need at least about 320 trading days for 12M forward-return validation.",
        }

    close = hist["Close"].reset_index(drop=True)
    bench_close = bench["Close"].reset_index(drop=True) if not bench.empty and "Close" in bench else None
    years = len(close) / 252
    max_horizon = TRADING_DAYS["12M"]
    model = signal_model or default_historical_signal

    sector = data.get("sector") or data.get("industry") or "Unknown"
    mcap_bucket = _market_cap_bucket(data.get("market_cap"))
    ticker = data.get("ticker", "")
    company = data.get("company_name", ticker)

    rows = []
    start = 252
    stop = len(close) - max_horizon
    for idx in range(start, max(stop, start), step_days):
        try:
            result = model(close, idx, bench_close, data, r)
        except TypeError:
            result = model(close, idx, bench_close)
        signal, score, rationale = result
        row = {
            "Ticker": ticker,
            "Company": company,
            "Date": hist["Date"].iloc[idx] if "Date" in hist else idx,
            "Signal": signal,
            "Signal Score": score,
            "Price": close.iloc[idx],
            "Sector": sector,
            "Market Cap Bucket": mcap_bucket,
            "Market Regime": _market_regime(bench_close, idx),
            "Rationale": rationale,
            "Forward Max Drawdown %": _forward_max_drawdown(close, idx, max_horizon),
        }
        for label, days in TRADING_DAYS.items():
            row[f"{label} Return %"] = _series_return(close, idx, days)
            if bench_close is not None and len(bench_close) > idx + days:
                row[f"{label} Excess Return %"] = _series_return(close, idx, days) - _series_return(bench_close, idx, days)
            else:
                row[f"{label} Excess Return %"] = np.nan
        rows.append(row)

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {
            "status": "Unavailable",
            "evidence_quality": "Low",
            "headline": "Historical validation unavailable: no test windows produced.",
            "summary": pd.DataFrame(),
            "by_regime": pd.DataFrame(),
            "by_sector": pd.DataFrame(),
            "by_market_cap": pd.DataFrame(),
            "trades": trades,
            "note": "No rolling windows were available after applying lookback and forward-return requirements.",
        }

    def summarize(group_cols):
        grouped = trades.groupby(group_cols, dropna=False)
        rows_out = []
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(group_cols, keys)}
            row["Observations"] = len(g)
            for label in ["1M", "3M", "6M", "12M"]:
                col = f"{label} Return %"
                ex_col = f"{label} Excess Return %"
                row[f"Median {label} Return %"] = g[col].median()
                row[f"{label} Hit Rate %"] = (g[col] > 0).mean() * 100
                row[f"Median {label} Excess %"] = g[ex_col].median()
            row["Median Forward Max Drawdown %"] = g["Forward Max Drawdown %"].median()
            rows_out.append(row)
        return pd.DataFrame(rows_out).round(1)

    summary = summarize(["Signal"]).sort_values("Signal")
    by_regime = summarize(["Market Regime", "Signal"]).sort_values(["Market Regime", "Signal"])
    by_sector = summarize(["Sector", "Signal"]).sort_values(["Sector", "Signal"])
    by_market_cap = summarize(["Market Cap Bucket", "Signal"]).sort_values(["Market Cap Bucket", "Signal"])

    buy = summary[summary["Signal"] == "BUY"]
    if not buy.empty:
        buy_median = _safe_float(buy["Median 12M Return %"].iloc[0])
        buy_hit = _safe_float(buy["12M Hit Rate %"].iloc[0])
        buy_n = int(buy["Observations"].iloc[0])
        headline = f"Over {years:.1f} years, BUY signals delivered {buy_median:.1f}% median 12M return with {buy_hit:.1f}% hit rate (n={buy_n})."
    else:
        headline = f"Over {years:.1f} years, no BUY validation windows were generated; review signal thresholds."

    evidence_quality = "High" if years >= min_years and len(trades) >= 12 else "Medium" if len(trades) >= 6 else "Low"
    return {
        "status": "Computed",
        "evidence_quality": evidence_quality,
        "headline": headline,
        "years": round(float(years), 1),
        "observations": int(len(trades)),
        "summary": summary,
        "by_regime": by_regime,
        "by_sector": by_sector,
        "by_market_cap": by_market_cap,
        "trades": trades.round(2),
        "note": "Validation uses rolling historical signal states and forward returns. Archived fundamentals are required for fully point-in-time DCF replay.",
    }


def combine_validation_results(results):
    """Aggregate validation outputs from many stocks into portfolio evidence."""
    trades = []
    for result in results or []:
        df = result.get("trades") if isinstance(result, dict) else None
        if df is not None and hasattr(df, "empty") and not df.empty:
            trades.append(df)
    if not trades:
        return {
            "status": "Unavailable",
            "headline": "No validation trades available.",
            "summary": pd.DataFrame(),
            "by_regime": pd.DataFrame(),
            "by_sector": pd.DataFrame(),
            "by_market_cap": pd.DataFrame(),
            "trades": pd.DataFrame(),
        }
    data = {"market_data": {"price_history": pd.DataFrame({"Close": [1] * 600})}}
    combined = pd.concat(trades, ignore_index=True)

    def summarize(group_cols):
        rows = []
        for keys, g in combined.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(group_cols, keys)}
            row["Observations"] = len(g)
            for label in ["1M", "3M", "6M", "12M"]:
                col = f"{label} Return %"
                ex_col = f"{label} Excess Return %"
                row[f"Median {label} Return %"] = g[col].median()
                row[f"{label} Hit Rate %"] = (g[col] > 0).mean() * 100
                row[f"Median {label} Excess %"] = g[ex_col].median()
            row["Median Forward Max Drawdown %"] = g["Forward Max Drawdown %"].median()
            rows.append(row)
        return pd.DataFrame(rows).round(1)

    summary = summarize(["Signal"]).sort_values("Signal")
    buy = summary[summary["Signal"] == "BUY"]
    if not buy.empty:
        headline = (
            f"Across {combined['Ticker'].nunique()} stocks, BUY signals delivered "
            f"{_safe_float(buy['Median 12M Return %'].iloc[0]):.1f}% median 12M return "
            f"with {_safe_float(buy['12M Hit Rate %'].iloc[0]):.1f}% hit rate "
            f"(n={int(buy['Observations'].iloc[0])})."
        )
    else:
        headline = "No BUY validation windows were generated in the combined sample."
    return {
        "status": "Computed",
        "headline": headline,
        "summary": summary,
        "by_regime": summarize(["Market Regime", "Signal"]),
        "by_sector": summarize(["Sector", "Signal"]),
        "by_market_cap": summarize(["Market Cap Bucket", "Signal"]),
        "trades": combined.round(2),
        "note": "Combined historical validation across available tickers.",
    }
