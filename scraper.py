"""
scraper.py — Fetches 5-year financial data from Screener.in for any NSE ticker.
Parses P&L, Balance Sheet, Cash Flow, top ratios, and peer tickers.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def to_num(val):
    """Safely convert string to float."""
    try:
        cleaned = str(val).replace(",", "").replace("₹", "").replace("%", "").replace("Cr", "").strip()
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
        df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_screener_data(ticker: str) -> dict:
    """
    Main function: scrape all financial data for an NSE ticker from Screener.in.
    Returns a dict with company info, financial tables, and peer list.
    """
    ticker = ticker.upper().strip()
    session = requests.Session()

    # Try consolidated first, then standalone
    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{ticker}{suffix}"
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                break
        except requests.RequestException as e:
            raise ConnectionError(f"Could not connect to Screener.in: {e}")
    else:
        raise ValueError(f"Ticker '{ticker}' not found on Screener.in. Check the symbol.")

    soup = BeautifulSoup(resp.text, "lxml")

    data = {
        "ticker": ticker,
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

    # ── Shareholding / about ──
    about = soup.find("div", class_="company-profile") or soup.find("section", id="about")
    if about:
        data["about"] = about.get_text(strip=True)[:500]
    else:
        data["about"] = ""

    # ── Peers ──
    data["peers"] = []
    peers_section = soup.find("section", id="peers")
    if peers_section:
        for a in peers_section.find_all("a", href=True):
            m = re.search(r"/company/([A-Z0-9\-&]+)/", a["href"], re.IGNORECASE)
            if m:
                peer = m.group(1).upper()
                if peer != ticker and peer not in data["peers"]:
                    data["peers"].append(peer)
        data["peers"] = data["peers"][:5]

    # ── Sector from breadcrumb ──
    breadcrumb = soup.find("div", class_="breadcrumb") or soup.find("nav", class_="breadcrumb")
    if breadcrumb:
        links = breadcrumb.find_all("a")
        if len(links) >= 2:
            data["sector"] = links[-1].get_text(strip=True)

    return data


def fetch_peer_data(peers: list, delay: float = 1.5) -> list:
    """
    Fetch key financial data for a list of peer tickers.
    Returns list of dicts with company name, ticker, and key ratios.
    """
    results = []
    for peer in peers:
        try:
            time.sleep(delay)
            d = fetch_screener_data(peer)
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
        return [f"FY{y}" for y in range(20, 25)]
    cols = list(df.columns[1:])
    if len(cols) >= n_years:
        return [str(c) for c in cols[-n_years:]]
    result = [str(c) for c in cols]
    return result + [""] * (n_years - len(result))
