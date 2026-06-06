"""Yahoo Finance adapter using yfinance when available."""

import numpy as np
import pandas as pd


def _yf_symbol(ticker: str) -> str:
    ticker = str(ticker or "").upper().strip()
    if ticker.startswith("^") or ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    return f"{ticker}.NS"


def _safe_float(value, default=np.nan):
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _info_float(info: dict, *keys, scale: float = 1.0):
    for key in keys:
        value = _safe_float((info or {}).get(key))
        if not np.isnan(value):
            return value / scale
    return np.nan


def _statement_value(df: pd.DataFrame, aliases, col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    for alias in aliases:
        if alias in df.index:
            return _safe_float(df.loc[alias, col])
    lowered = {str(idx).lower(): idx for idx in df.index}
    for alias in aliases:
        idx = lowered.get(str(alias).lower())
        if idx is not None:
            return _safe_float(df.loc[idx, col])
    return np.nan


def _statement_table(
    df: pd.DataFrame,
    row_map: list,
    scale_rows=None,
    max_years: int = 5,
    currency_multiplier: float = 1.0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    scale_rows = set(scale_rows or [])
    cols = sorted(list(df.columns))[-max_years:]
    labels = [pd.to_datetime(c).strftime("%b %Y") for c in cols]
    rows = []
    for label, aliases in row_map:
        vals = []
        for col in cols:
            value = _statement_value(df, aliases, col)
            if not np.isnan(value):
                value = value * currency_multiplier
            if label in scale_rows and not np.isnan(value):
                value = value / 1e7
            vals.append(value)
        rows.append([label] + vals)
    return pd.DataFrame(rows, columns=["Metric"] + labels)


def _currency_to_inr_multiplier(currency: str) -> float:
    currency = str(currency or "INR").upper()
    if currency in ("", "INR"):
        return 1.0
    try:
        import yfinance as yf
        pair = "INR=X" if currency == "USD" else f"{currency}INR=X"
        hist = yf.Ticker(pair).history(period="5d", auto_adjust=False)
        if hist is not None and not hist.empty and "Close" in hist:
            value = _safe_float(hist["Close"].dropna().iloc[-1])
            if not np.isnan(value) and value > 0:
                return value
    except Exception:
        pass
    if currency == "USD":
        return 83.0
    return 1.0


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


def fetch_fundamentals_from_yahoo(ticker: str) -> dict:
    """
    Build the app's canonical fundamentals object from Yahoo Finance.

    This is a fallback for hosts where Screener.in refuses outbound connections.
    Yahoo statement values are reported in INR, so financial statement and market
    cap values are converted to crore to match the rest of the app.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError(f"yfinance unavailable: {exc}") from exc

    symbol = _yf_symbol(ticker)
    yf_ticker = yf.Ticker(symbol)

    try:
        financials = yf_ticker.financials
    except Exception:
        financials = pd.DataFrame()
    try:
        balance_sheet = yf_ticker.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()
    try:
        cashflow = yf_ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    if financials.empty and balance_sheet.empty and cashflow.empty:
        raise RuntimeError(f"Yahoo Finance fundamentals unavailable for {symbol}")

    info = {}
    try:
        info = yf_ticker.info or {}
    except Exception:
        info = {}
    financial_currency = info.get("financialCurrency") or info.get("currency") or "INR"
    currency_multiplier = _currency_to_inr_multiplier(financial_currency)

    pl_rows = [
        ("Sales", ["Total Revenue", "Operating Revenue"]),
        ("Operating Profit", ["EBITDA", "Normalized EBITDA", "EBIT"]),
        ("Net Profit", ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"]),
        ("Interest", ["Interest Expense"]),
        ("Depreciation", ["Reconciled Depreciation", "Depreciation And Amortization"]),
        ("Profit before tax", ["Pretax Income"]),
        ("Tax", ["Tax Provision"]),
        ("EPS", ["Diluted EPS", "Basic EPS"]),
    ]
    bs_rows = [
        ("Total Assets", ["Total Assets"]),
        ("Total Debt", ["Total Debt", "Net Debt"]),
        ("Total Equity", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
        ("Current Assets", ["Current Assets", "Working Capital"]),
        ("Current Liabilities", ["Current Liabilities"]),
        ("Inventory", ["Inventory"]),
        ("Receivables", ["Accounts Receivable", "Receivables"]),
        ("Payables", ["Accounts Payable", "Payables"]),
        ("Cash", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        ("Fixed Assets", ["Net PPE", "Properties"]),
    ]
    cf_rows = [
        ("Cash from Operating Activity", ["Operating Cash Flow"]),
        ("Cash from Investing Activity", ["Investing Cash Flow"]),
        ("Cash from Financing Activity", ["Financing Cash Flow"]),
        ("Capital Expenditure", ["Capital Expenditure"]),
        ("Free Cash Flow", ["Free Cash Flow"]),
    ]

    crore_rows = {label for label, _ in pl_rows + bs_rows + cf_rows if label != "EPS"}
    market_cap = _info_float(info, "marketCap", scale=1e7)
    current_price = _info_float(info, "currentPrice", "regularMarketPrice", "previousClose")
    book_value = _info_float(info, "bookValue")
    price_to_book = _info_float(info, "priceToBook")
    if np.isnan(book_value) and not np.isnan(current_price) and price_to_book:
        book_value = current_price / price_to_book
    cf_table = _statement_table(cashflow, cf_rows, crore_rows, currency_multiplier=currency_multiplier)
    if not cf_table.empty:
        capex_mask = cf_table.iloc[:, 0].astype(str).str.fullmatch("Capital Expenditure", case=False, na=False)
        if capex_mask.any():
            cf_table.loc[capex_mask, cf_table.columns[1:]] = (
                cf_table.loc[capex_mask, cf_table.columns[1:]].abs()
            )

    data = {
        "ticker": str(ticker or "").upper().strip(),
        "url": f"https://finance.yahoo.com/quote/{symbol}",
        "company_name": info.get("longName") or info.get("shortName") or str(ticker or "").upper().strip(),
        "current_price": current_price,
        "market_cap": market_cap,
        "pe_ratio": _info_float(info, "trailingPE", "forwardPE"),
        "book_value": book_value,
        "dividend_yield": _info_float(info, "dividendYield"),
        "roce": np.nan,
        "roe": _info_float(info, "returnOnEquity", scale=0.01),
        "face_value": np.nan,
        "high_52w": _info_float(info, "fiftyTwoWeekHigh"),
        "low_52w": _info_float(info, "fiftyTwoWeekLow"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "about": info.get("longBusinessSummary", "")[:500],
        "pl": _statement_table(financials, pl_rows, crore_rows, currency_multiplier=currency_multiplier),
        "bs": _statement_table(balance_sheet, bs_rows, crore_rows, currency_multiplier=currency_multiplier),
        "cf": cf_table,
        "quarters": pd.DataFrame(),
        "peers": [],
        "all_ratios": {},
        "growth_table": {},
        "growth_estimate": {},
        "sources": {
            "financials": "Yahoo Finance fallback",
            "fundamentals": "Yahoo Finance fallback",
            "top_ratios": "Yahoo Finance fallback",
            "peers": "fallback peer map",
        },
        "source_status": {"Screener.in": "unavailable; Yahoo Finance fallback used"},
        "data_quality_notes": [
            "Screener.in was unavailable from this app host; fundamentals are from Yahoo Finance.",
            f"Yahoo statement currency {financial_currency} was converted to INR crore.",
        ],
    }
    return data
