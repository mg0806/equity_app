"""
scraper.py — Fetches 5-year financial data from Screener.in for any NSE ticker.
Parses P&L, Balance Sheet, Cash Flow, top ratios, and peer tickers.
"""

import requests
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SCREENER_CACHE_DIR = Path(".cache") / "equity_app" / "screener"
LEGACY_TICKER_ALIASES = {
    # HDFC Ltd merged into HDFC Bank; keep the app on the active NSE/Screener symbol.
    "HDFC": "HDFCBANK",
}


def normalize_ticker(ticker: str) -> str:
    """Return the canonical active ticker used by the data adapters."""
    cleaned = str(ticker or "").upper().strip()
    return LEGACY_TICKER_ALIASES.get(cleaned, cleaned)


def _screener_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _cache_path(ticker: str, suffix: str) -> Path:
    page = "consolidated" if suffix == "/consolidated/" else "standalone"
    return SCREENER_CACHE_DIR / f"{ticker}_{page}.html"


def _read_cached_html(ticker: str):
    for suffix in ("/consolidated/", "/"):
        path = _cache_path(ticker, suffix)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8"), f"https://www.screener.in/company/{ticker}{suffix}"
            except OSError:
                continue
    return None, None


def _write_cached_html(ticker: str, suffix: str, html: str) -> None:
    try:
        SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(ticker, suffix).write_text(html, encoding="utf-8")
    except OSError:
        pass


def _dedupe_columns(cols: list, fallback_prefix: str = "Col") -> list:
    """Return non-empty, unique column labels for Streamlit-safe DataFrames."""
    out = []
    seen = {}
    for idx, col in enumerate(cols):
        label = str(col).strip()
        if not label:
            label = f"{fallback_prefix} {idx + 1}"
        if label in seen:
            seen[label] += 1
            label = f"{label} {seen[label]}"
        else:
            seen[label] = 1
        out.append(label)
    return out


INDEX_TICKERS = {
    "1001", "1002", "1003", "1004", "1005", "1006",
    "NIFTY", "CNX500", "CNXIT", "BANKNIFTY", "NIFTYBANK",
    "SENSEX", "BSESENSEX", "BSE500", "BSE100", "BSE200",
}


FALLBACK_PEERS = {
    "RELIANCE": ["IOC", "BPCL", "HINDPETRO", "ONGC", "OIL"],
    "BAJAJHFL": ["LICHSGFIN", "PNBHOUSING", "CANFINHOME", "AAVAS", "HOMEFIRST"],
    "BSE": ["MCX", "CDSL", "CAMS", "KFINTECH"],
    "HDFCBANK": ["ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK"],
    "ICICIBANK": ["HDFCBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK"],
    "INFY": ["TCS", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "TCS": ["INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
}


def fallback_peers_for(data: dict) -> list:
    """Return a conservative fallback peer set when Screener has only indices or no peers."""
    ticker = str(data.get("ticker", "") or "").upper().strip()
    if ticker in FALLBACK_PEERS:
        return [p for p in FALLBACK_PEERS[ticker] if p != ticker]

    text = " ".join(str(data.get(k, "") or "") for k in ("company_name", "sector", "industry", "about")).lower()
    if "housing finance" in text or "home loan" in text:
        peers = FALLBACK_PEERS["BAJAJHFL"]
    elif "bank" in text:
        peers = FALLBACK_PEERS["HDFCBANK"]
    elif "information technology" in text or "software" in text or "it services" in text:
        peers = FALLBACK_PEERS["INFY"]
    elif "oil" in text or "refinery" in text or "petrochemical" in text:
        peers = FALLBACK_PEERS["RELIANCE"]
    elif "exchange" in text or "capital market" in text or "depository" in text:
        peers = FALLBACK_PEERS["BSE"]
    else:
        peers = []
    return [p for p in peers if p != ticker and not is_index_or_benchmark(p)]


def is_index_or_benchmark(ticker: str, company_name: str = "") -> bool:
    """Return True for Screener benchmark/index rows, not operating peers."""
    t = str(ticker or "").upper().strip()
    name = str(company_name or "").lower()
    if not t:
        return True
    if t in INDEX_TICKERS or t.isdigit():
        return True
    index_words = ("sensex", "nifty", "bse 500", "bse 100", "bse 200", "index")
    return any(word in name for word in index_words)


def to_num(val):
    """Safely convert string to float."""
    try:
        cleaned = str(val).replace(",", "").replace("₹", "").replace("â‚¹", "")
        cleaned = cleaned.replace("%", "").replace("Cr", "").strip()
        cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
        return float(cleaned)
    except:
        return np.nan


def parse_screener_table(soup, section_id: str) -> pd.DataFrame:
    """Parse a financial table from a Screener.in section."""
    section = soup.find("section", id=section_id)
    if not section:
        return pd.DataFrame()
    table = section.find("table")
    if not table:
        return pd.DataFrame()

    thead = table.find("thead")
    tbody = table.find("tbody")
    if not thead or not tbody:
        return pd.DataFrame()

    cols = [th.get_text(strip=True) for th in thead.find_all("th")]
    rows = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True).replace(",", "") for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not cols or not rows:
        return pd.DataFrame()

    try:
        width = len(rows[0])
        if len(cols) < width:
            cols = cols + [f"Col {i + 1}" for i in range(len(cols), width)]
        cols = _dedupe_columns(cols[:width])
        df = pd.DataFrame(rows, columns=cols)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_screener_data(ticker: str) -> dict:
    """
    Main function: scrape all financial data for an NSE ticker from Screener.in.
    Returns a dict with company info, financial tables, and peer list.
    """
    requested_ticker = str(ticker or "").upper().strip()
    ticker = normalize_ticker(requested_ticker)
    session = _screener_session()

    # Try consolidated first, then standalone
    last_error = None
    last_status = None
    resp = None
    url = None
    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{ticker}{suffix}"
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            last_status = resp.status_code
            if resp.status_code == 200:
                _write_cached_html(ticker, suffix, resp.text)
                break
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = str(e)
    else:
        cached_html, cached_url = _read_cached_html(ticker)
        if cached_html:
            resp = None
            url = cached_url
        elif last_status == 404:
            raise ValueError(f"Ticker '{ticker}' not found on Screener.in. Check the symbol.")
        elif last_error:
            raise ConnectionError(
                "Screener.in is temporarily unreachable from this app host. "
                "Please try again in a few minutes or use a saved snapshot. "
                f"Last error: {last_error}"
            )
        else:
            raise ValueError(f"Ticker '{ticker}' not found on Screener.in. Check the symbol.")

    html = cached_html if "cached_html" in locals() and cached_html else resp.text
    soup = BeautifulSoup(html, "lxml")

    data = {
        "ticker": ticker,
        "requested_ticker": requested_ticker,
        "url": url,
        "company_name": ticker,
        "current_price": None,
        "market_cap": None,
        "pe_ratio": None,
        "book_value": None,
        "dividend_yield": None,
        "roce": None,
        "roe": None,
        "face_value": None,
        "high_52w": None,
        "low_52w": None,
        "sector": None,
        "industry": None,
    }

    # ── Company name ──
    name_tag = soup.find("h1", class_="margin-0") or soup.find("h1")
    if name_tag:
        data["company_name"] = name_tag.get_text(strip=True)

    # ── Top ratios (price, market cap, PE, etc.) ──
    top_ratios = soup.find("ul", id="top-ratios")
    if top_ratios:
        for li in top_ratios.find_all("li"):
            name_span = li.find("span", class_="name")
            val_span  = li.find("span", class_="nowrap") or li.find("span", class_="number")
            if not name_span or not val_span:
                continue
            name_txt = name_span.get_text(strip=True).lower()
            val_txt  = val_span.get_text(strip=True)
            val      = to_num(val_txt)

            if "market cap" in name_txt:
                data["market_cap"] = val
            elif "current price" in name_txt or "price" in name_txt:
                data["current_price"] = val
            elif "stock p/e" in name_txt or "p/e" in name_txt:
                data["pe_ratio"] = val
            elif "book value" in name_txt:
                data["book_value"] = val
            elif "dividend yield" in name_txt:
                data["dividend_yield"] = val
            elif "roce" in name_txt:
                data["roce"] = val
            elif "roe" in name_txt:
                data["roe"] = val
            elif "face value" in name_txt:
                data["face_value"] = val
            elif "52 week high" in name_txt:
                data["high_52w"] = val
            elif "52 week low" in name_txt:
                data["low_52w"] = val

    # ── Financial tables ──
    data["pl"] = parse_screener_table(soup, "profit-loss")
    data["bs"] = parse_screener_table(soup, "balance-sheet")
    data["cf"] = parse_screener_table(soup, "cash-flow")
    data["quarters"] = parse_screener_table(soup, "quarters")

    # ── Shareholding / about ──
    about = soup.find("div", class_="company-profile") or soup.find("section", id="about")
    if about:
        data["about"] = about.get_text(strip=True)[:500]
    else:
        data["about"] = ""

    # ── Additional Screener metrics & ratios ──
    data["all_ratios"] = _extract_all_ratios(soup)
    data["growth_table"] = _extract_growth_table(soup)
    
    # ── Growth estimates from analyst data (if available) ──
    data["growth_estimate"] = _extract_growth_estimates(soup)

    # ── Peers ──
    data["peers"] = []
    peers_section = soup.find("section", id="peers")
    if peers_section:
        for a in peers_section.find_all("a", href=True):
            m = re.search(r"/company/([A-Z0-9\-&]+)/", a["href"], re.IGNORECASE)
            if m:
                peer = m.group(1).upper()
                peer_name = a.get_text(" ", strip=True)
                if (peer != ticker and peer not in data["peers"]
                        and not is_index_or_benchmark(peer, peer_name)):
                    data["peers"].append(peer)
        data["peers"] = data["peers"][:5]

    if not data["peers"]:
        data["peers"] = fallback_peers_for(data)[:5]

    # ── Sector from breadcrumb ──
    breadcrumb = soup.find("div", class_="breadcrumb") or soup.find("nav", class_="breadcrumb")
    if breadcrumb:
        links = breadcrumb.find_all("a")
        if len(links) >= 2:
            data["sector"] = links[-1].get_text(strip=True)

    if requested_ticker and requested_ticker != ticker:
        data.setdefault("data_quality_notes", []).append(
            f"Input ticker {requested_ticker} was mapped to active symbol {ticker}."
        )
    if "cached_html" in locals() and cached_html:
        data.setdefault("source_status", {})["Screener.in"] = "cached"
        data.setdefault("data_quality_notes", []).append(
            "Live Screener.in was unavailable; used the latest cached Screener page."
        )

    return data


def _extract_all_ratios(soup) -> dict:
    """
    Extract all available ratios from Screener.in page.
    Looks for multiple ratio sections and compiles them.
    """
    ratios = {}
    
    # Try to find ratio tables/sections
    ratio_sections = soup.find_all("section")
    for section in ratio_sections:
        section_id = section.get("id", "")
        if "ratio" in section_id.lower() or "metrics" in section_id.lower():
            table = section.find("table")
            if table:
                thead = table.find("thead")
                tbody = table.find("tbody")
                if thead and tbody:
                    for tr in tbody.find_all("tr"):
                        cells = tr.find_all(["td", "th"])
                        if len(cells) >= 2:
                            name = cells[0].get_text(strip=True)
                            value = to_num(cells[1].get_text(strip=True))
                            if name and not np.isnan(value):
                                ratios[name] = value
    
    # Also check for metrics in list format (common in Screener.in)
    metric_lists = soup.find_all("ul", class_=re.compile(r"(metric|ratio|list)", re.I))
    for ul in metric_lists:
        for li in ul.find_all("li"):
            text = li.get_text(strip=True)
            # Look for patterns like "Metric: Value"
            if ":" in text:
                parts = text.split(":", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    value = to_num(parts[1])
                    if name and not np.isnan(value):
                        ratios[name] = value
    
    return ratios


def _extract_growth_table(soup) -> dict:
    """
    Extract Screener's CAGR summary table.

    Typical rows include Compounded Sales Growth, Compounded Profit Growth,
    Stock Price CAGR, and Return on Equity.
    """
    out = {}
    labels = (
        "Compounded Sales Growth",
        "Compounded Profit Growth",
        "Stock Price CAGR",
        "Return on Equity",
    )

    text_node = soup.find(string=re.compile("Compounded Sales Growth", re.I))
    table = text_node.find_parent("table") if text_node else None
    if not table:
        return out

    current_label = None
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue

        matched_label = next((label for label in labels if label.lower() in cells[0].lower()), None)
        if matched_label:
            current_label = matched_label
            out.setdefault(current_label, {})
            continue

        if current_label and len(cells) >= 2:
            period = cells[0].replace(":", "").strip()
            value = to_num(cells[1])
            if period and not np.isnan(value):
                out[current_label][period] = value

    out = {k: v for k, v in out.items() if v}

    return out


def _extract_growth_estimates(soup) -> dict:
    """
    Extract analyst growth estimates or projections from Screener.in.
    Looks for EPS growth, revenue growth, and other forward estimates.
    """
    estimates = {}
    
    # Look for analyst estimates section
    est_sections = soup.find_all(["section", "div"], attrs={"class": re.compile(r"(estimate|forecast|projection)", re.I)})
    
    for section in est_sections:
        divs = section.find_all("div")
        for div in divs:
            text = div.get_text(strip=True)
            # Look for growth rate mentions
            if "growth" in text.lower() or "eps" in text.lower():
                # Try to extract percentage values
                percentages = re.findall(r'([\d.]+)\s*%', text)
                if percentages:
                    estimates["found_growth"] = float(percentages[0]) / 100
    
    # If no dedicated section, try to extract from page text
    if not estimates:
        # Look in main content for forward guidance or analyst estimates
        page_text = soup.get_text()
        if "expected growth" in page_text.lower() or "cagr" in page_text.lower():
            growth_values = re.findall(r'(?:expected growth|CAGR|projected growth)[\s:]*(\d+(?:\.\d+)?)\s*%', page_text, re.I)
            if growth_values:
                estimates["cagr_estimate"] = float(growth_values[0]) / 100
    
    return estimates


def fetch_peer_data(peers: list, delay: float = 1.5) -> list:
    """
    Fetch key financial data for a list of peer tickers.
    Returns list of dicts with company name, ticker, and key ratios.
    """
    results = []
    for peer in peers:
        if is_index_or_benchmark(peer):
            continue
        try:
            time.sleep(delay)
            d = fetch_screener_data(peer)
            if is_index_or_benchmark(d.get("ticker"), d.get("company_name")):
                continue
            results.append(d)
        except Exception as e:
            print(f"Could not fetch peer {peer}: {e}")
    return results


def extract_series(df: pd.DataFrame, patterns: list, n_years: int = 5) -> list:
    """
    Extract a numeric time series from a financial table by matching row name patterns.
    Returns a list of n_years floats (NaN where data missing).
    """
    if df.empty:
        return [np.nan] * n_years

    for pattern in patterns:
        matches = df[df.iloc[:, 0].str.contains(pattern, case=False, na=False, regex=True)]
        if not matches.empty:
            row = matches.iloc[0]
            vals = [to_num(v) for v in row.iloc[1:]]
            # Return last n_years values
            if len(vals) >= n_years:
                return vals[-n_years:]
            return vals + [np.nan] * (n_years - len(vals))

    return [np.nan] * n_years


def get_year_labels(df: pd.DataFrame, n_years: int = 5) -> list:
    """Extract year column headers from a financial table."""
    if df.empty:
        return [f"FY-{i}" for i in range(n_years - 1, -1, -1)]

    cols = [str(c).strip() for c in list(df.columns[1:]) if str(c).strip()]
    year_like = [c for c in cols if re.search(r"(?:19|20)\d{2}|FY|Mar|Dec|Sep|Jun|TTM", c, re.I)]
    labels = year_like[-n_years:] if len(year_like) >= n_years else cols[-n_years:]

    if len(labels) < n_years:
        missing = n_years - len(labels)
        labels = [f"FY-{i}" for i in range(missing, 0, -1)] + labels

    return _dedupe_columns(labels[-n_years:], fallback_prefix="FY")
