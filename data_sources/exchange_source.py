"""No-key NSE exchange event adapter.

This module fetches public corporate actions and announcements from NSE's
website JSON endpoints. It is deliberately best-effort: exchange websites can
change headers, cookies or response shapes, so failures return source status
metadata instead of breaking valuation reports.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


NSE_BASE_URL = "https://www.nseindia.com"
BSE_BASE_URL = "https://api.bseindia.com/BseIndiaAPI/api"
CACHE_DIR = Path(os.getenv("EQUITY_APP_CACHE_DIR", ".cache/equity_app"))

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
}
BSE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
    "User-Agent": HEADERS["User-Agent"],
}
CACHE_TTL_SECONDS = 6 * 60 * 60
BSE_MASTER_TTL_SECONDS = 7 * 24 * 60 * 60


def _clean_symbol(ticker: str) -> str:
    symbol = str(ticker or "").upper().strip()
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str, ttl_seconds: int) -> Any | None:
    path = _cache_path(key)
    try:
        if not path.exists() or time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(key: str, payload: Any) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(NSE_BASE_URL, timeout=12)
    return session


def _request_json(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{NSE_BASE_URL}{path}"
    session = _session()
    resp = session.get(url, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _request_json_cached(path: str, params: dict[str, Any] | None, cache_key: str, ttl_seconds: int) -> Any:
    cached = _load_cache(cache_key, ttl_seconds)
    if cached is not None:
        return cached
    payload = _request_json(path, params)
    _save_cache(cache_key, payload)
    return payload


def _request_bse_json(path: str, params: dict[str, Any] | None, cache_key: str, ttl_seconds: int) -> Any:
    cached = _load_cache(cache_key, ttl_seconds)
    if cached is not None:
        return cached
    url = f"{BSE_BASE_URL}{path}"
    resp = requests.get(url, headers=BSE_HEADERS, params=params or {}, timeout=25)
    resp.raise_for_status()
    payload = resp.json()
    _save_cache(cache_key, payload)
    return payload


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "rows", "result", "items", "Table"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return [payload] if payload else []


def _normalize(records: list[dict[str, Any]], preferred: list[str], limit: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return df

    cols = []
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        col = lower_map.get(name.lower())
        if col and col not in cols:
            cols.append(col)
    cols.extend([c for c in df.columns if c not in cols])
    return df[cols].head(limit).copy()


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "Source", source)
    return out


def fetch_nse_corporate_actions(ticker: str, limit: int = 25) -> dict:
    """Fetch recent/upcoming NSE corporate actions for one equity symbol."""
    symbol = _clean_symbol(ticker)
    try:
        payload = _request_json_cached(
            "/api/corporates-corporateActions",
            params={"index": "equities", "symbol": symbol},
            cache_key=f"nse_actions_{symbol}",
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        records = _as_records(payload)
        df = _normalize(
            records,
            [
                "symbol",
                "series",
                "subject",
                "faceVal",
                "exDate",
                "recDate",
                "bcStartDate",
                "bcEndDate",
                "ndStartDate",
                "ndEndDate",
            ],
            limit,
        )
        return {"source": "NSE India", "symbol": symbol, "data": df}
    except Exception as exc:
        return {"source": "NSE India", "symbol": symbol, "error": str(exc), "data": pd.DataFrame()}


def fetch_nse_announcements(ticker: str, lookback_days: int = 90, limit: int = 25) -> dict:
    """Fetch recent NSE corporate announcements for one equity symbol."""
    symbol = _clean_symbol(ticker)
    today = date.today()
    start = today - timedelta(days=lookback_days)
    params = {
        "index": "equities",
        "symbol": symbol,
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": today.strftime("%d-%m-%Y"),
    }
    try:
        payload = _request_json_cached(
            "/api/corporate-announcements",
            params=params,
            cache_key=f"nse_announcements_{symbol}_{start:%Y%m%d}_{today:%Y%m%d}",
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        records = _as_records(payload)
        df = _normalize(
            records,
            [
                "symbol",
                "companyName",
                "desc",
                "attchmntText",
                "an_dt",
                "sort_date",
                "attchmntFile",
            ],
            limit,
        )
        return {"source": "NSE India", "symbol": symbol, "data": df}
    except Exception as exc:
        return {"source": "NSE India", "symbol": symbol, "error": str(exc), "data": pd.DataFrame()}


def fetch_bse_scrip_master() -> pd.DataFrame:
    """Fetch active BSE equity scrip master list and cache it locally."""
    payload = _request_bse_json(
        "/ListofScripData/w",
        params={"Group": "", "Scripcode": "", "industry": "", "segment": "Equity", "status": "Active"},
        cache_key="bse_active_equity_master",
        ttl_seconds=BSE_MASTER_TTL_SECONDS,
    )
    return pd.DataFrame(payload if isinstance(payload, list) else [])


def resolve_bse_scrip(ticker: str, company_name: str | None = None) -> dict:
    """Resolve NSE ticker/company text to a BSE scrip code using the public BSE master."""
    symbol = _clean_symbol(ticker)
    try:
        master = fetch_bse_scrip_master()
        if master.empty:
            return {"source": "BSE India", "symbol": symbol, "error": "BSE scrip master unavailable"}

        symbol_norm = _norm_text(symbol)
        name_norm = _norm_text(company_name or "")
        candidates = master.copy()
        candidates["_scrip_id_norm"] = candidates.get("scrip_id", "").astype(str).map(_norm_text)
        candidates["_name_norm"] = candidates.get("Scrip_Name", "").astype(str).map(_norm_text)
        candidates["_issuer_norm"] = candidates.get("Issuer_Name", "").astype(str).map(_norm_text)

        exact = candidates[candidates["_scrip_id_norm"] == symbol_norm]
        if exact.empty and name_norm:
            exact = candidates[
                candidates["_name_norm"].str.contains(name_norm, na=False)
                | candidates["_issuer_norm"].str.contains(name_norm, na=False)
                | candidates.apply(lambda row: row["_name_norm"] in name_norm or row["_issuer_norm"] in name_norm, axis=1)
            ]
        if exact.empty:
            exact = candidates[candidates["_name_norm"].str.contains(symbol_norm, na=False)]
        if exact.empty:
            return {"source": "BSE India", "symbol": symbol, "error": "Could not resolve BSE scrip code"}

        row = exact.iloc[0].to_dict()
        return {
            "source": "BSE India",
            "symbol": symbol,
            "scrip_code": str(row.get("SCRIP_CD") or ""),
            "scrip_id": row.get("scrip_id"),
            "scrip_name": row.get("Scrip_Name"),
            "isin": row.get("ISIN_NUMBER"),
            "match": {k: row.get(k) for k in ("SCRIP_CD", "Scrip_Name", "scrip_id", "ISIN_NUMBER", "NSURL")},
        }
    except Exception as exc:
        return {"source": "BSE India", "symbol": symbol, "error": str(exc)}


def fetch_bse_corporate_actions(ticker: str, company_name: str | None = None, limit: int = 25) -> dict:
    """Fetch BSE corporate actions for one resolved scrip code."""
    resolved = resolve_bse_scrip(ticker, company_name)
    if resolved.get("error"):
        return {**resolved, "data": pd.DataFrame()}
    code = resolved["scrip_code"]
    try:
        payload = _request_bse_json(
            "/CorporateAction/w",
            params={"scripcode": code},
            cache_key=f"bse_actions_{code}",
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        records = _as_records(payload)
        df = _normalize(records, ["purpose_name", "BCRD_from", "Amount"], limit)
        if not df.empty:
            df.insert(0, "SCRIP_CD", code)
            df.insert(1, "Scrip_Name", resolved.get("scrip_name"))
        return {**resolved, "data": df}
    except Exception as exc:
        return {**resolved, "error": str(exc), "data": pd.DataFrame()}


def fetch_bse_announcements(ticker: str, company_name: str | None = None, lookback_days: int = 365, limit: int = 25) -> dict:
    """Fetch BSE corporate announcements for one resolved scrip code."""
    resolved = resolve_bse_scrip(ticker, company_name)
    if resolved.get("error"):
        return {**resolved, "data": pd.DataFrame()}
    code = resolved["scrip_code"]
    today = date.today()
    start = today - timedelta(days=lookback_days)
    try:
        payload = _request_bse_json(
            "/AnnGetData/w",
            params={
                "pageno": 1,
                "strCat": "-1",
                "strPrevDate": start.strftime("%Y%m%d"),
                "strScrip": code,
                "strSearch": "P",
                "strToDate": today.strftime("%Y%m%d"),
                "strType": "C",
            },
            cache_key=f"bse_announcements_{code}_{start:%Y%m%d}_{today:%Y%m%d}",
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        records = _as_records(payload)
        df = _normalize(
            records,
            ["SCRIP_CD", "NEWSSUB", "DT_TM", "NEWS_DT", "CATEGORYNAME", "ATTACHMENTNAME", "MORE"],
            limit,
        )
        return {**resolved, "data": df}
    except Exception as exc:
        return {**resolved, "error": str(exc), "data": pd.DataFrame()}


def fetch_exchange_events(ticker: str, company_name: str | None = None) -> dict:
    """Fetch no-key exchange events used for source audit and report context."""
    nse_actions = fetch_nse_corporate_actions(ticker)
    nse_announcements = fetch_nse_announcements(ticker)
    bse_actions = fetch_bse_corporate_actions(ticker, company_name)
    bse_announcements = fetch_bse_announcements(ticker, company_name)

    errors = {}
    source_errors = {}
    for source, action_payload, announcement_payload in (
        ("NSE India", nse_actions, nse_announcements),
        ("BSE India", bse_actions, bse_announcements),
    ):
        source_errors[source] = {}
        if action_payload.get("error"):
            source_errors[source]["corporate_actions"] = action_payload["error"]
            errors[f"{source} corporate_actions"] = action_payload["error"]
        if announcement_payload.get("error"):
            source_errors[source]["announcements"] = announcement_payload["error"]
            errors[f"{source} announcements"] = announcement_payload["error"]

    action_frames = [
        _with_source(nse_actions.get("data", pd.DataFrame()), "NSE India"),
        _with_source(bse_actions.get("data", pd.DataFrame()), "BSE India"),
    ]
    announcement_frames = [
        _with_source(nse_announcements.get("data", pd.DataFrame()), "NSE India"),
        _with_source(bse_announcements.get("data", pd.DataFrame()), "BSE India"),
    ]
    corporate_actions = pd.concat([df for df in action_frames if not df.empty], ignore_index=True, sort=False) if any(not df.empty for df in action_frames) else pd.DataFrame()
    announcements = pd.concat([df for df in announcement_frames if not df.empty], ignore_index=True, sort=False) if any(not df.empty for df in announcement_frames) else pd.DataFrame()

    return {
        "source": "NSE India + BSE India",
        "symbol": _clean_symbol(ticker),
        "bse_scrip": {k: bse_actions.get(k) for k in ("scrip_code", "scrip_id", "scrip_name", "isin", "match")},
        "corporate_actions": corporate_actions,
        "announcements": announcements,
        "source_errors": source_errors,
        "errors": errors,
    }
