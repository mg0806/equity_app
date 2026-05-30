"""Yahoo Finance adapter using yfinance when available."""

import numpy as np
import pandas as pd


def _yf_symbol(ticker: str) -> str:
    ticker = str(ticker or "").upper().strip()
    if ticker.startswith("^") or ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    return f"{ticker}.NS"


def _clean_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reset_index().copy()
    out.columns = [str(c).title().replace(" ", "_") for c in out.columns]
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in out.columns]
    return out[keep]


def _return_pct(series: pd.Series, periods: int) -> float:
    vals = series.dropna()
    if len(vals) <= periods or vals.iloc[-periods] == 0:
        return np.nan
    return (vals.iloc[-1] / vals.iloc[-periods] - 1) * 100


def fetch_market_data(ticker: str, period: str = "5y") -> dict:
    """
    Fetch price history and market snapshot.

    Returns an error key instead of raising so Screener can remain the primary
    source when Yahoo is unavailable.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        return {"source": "Yahoo Finance", "error": f"yfinance unavailable: {exc}"}

    symbol = _yf_symbol(ticker)
    try:
        yf_ticker = yf.Ticker(symbol)
        hist = yf_ticker.history(period=period, auto_adjust=False)
        info = {}
        try:
            info = yf_ticker.fast_info or {}
        except Exception:
            info = {}

        benchmark = yf.Ticker("^NSEI").history(period=period, auto_adjust=False)
        hist_clean = _clean_history(hist)
        bench_clean = _clean_history(benchmark)

        close = hist["Close"] if hist is not None and not hist.empty and "Close" in hist else pd.Series(dtype=float)
        daily_returns = close.pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100) if not daily_returns.empty else np.nan

        beta = np.nan
        if benchmark is not None and not benchmark.empty and "Close" in benchmark and not daily_returns.empty:
            stock_ret = hist["Close"].pct_change()
            bench_ret = benchmark["Close"].pct_change()
            aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
            if len(aligned) > 30 and aligned.iloc[:, 1].var() != 0:
                beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())

        last_price = np.nan
        for key in ("last_price", "lastPrice"):
            try:
                last_price = float(info.get(key))
                break
            except Exception:
                pass
        if np.isnan(last_price) and not close.empty:
            last_price = float(close.iloc[-1])

        return {
            "source": "Yahoo Finance",
            "symbol": symbol,
            "current_price": last_price,
            "price_history": hist_clean,
            "benchmark_history": bench_clean,
            "return_1m_pct": _return_pct(close, 21),
            "return_3m_pct": _return_pct(close, 63),
            "return_1y_pct": _return_pct(close, 252),
            "volatility_1y_pct": volatility,
            "beta_vs_nifty": beta,
        }
    except Exception as exc:
        return {"source": "Yahoo Finance", "symbol": symbol, "error": str(exc)}
