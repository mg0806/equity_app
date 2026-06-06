"""Aggregate multiple data sources into the app's canonical company object."""

import numpy as np
import pandas as pd

from .exchange_source import fetch_exchange_events
from .screener_source import fetch_fundamentals, fetch_peers
from .yahoo_source import fetch_market_data
from scraper import normalize_ticker
from snapshot_store import SnapshotStore


def _safe_float(value, default=np.nan):
    try:
        f = float(value)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _merge_market_data(data: dict, market: dict) -> dict:
    data = dict(data)
    data.setdefault("sources", {})
    data.setdefault("source_status", {})

    if not market or market.get("error"):
        data["source_status"]["Yahoo Finance"] = market.get("error", "unavailable") if market else "unavailable"
        data["market_data"] = market or {}
        return data

    yahoo_price = _safe_float(market.get("current_price"), np.nan)
    screener_price = _safe_float(data.get("current_price"), np.nan)
    if not np.isnan(yahoo_price):
        data["yahoo_current_price"] = yahoo_price
        if np.isnan(screener_price):
            data["current_price"] = yahoo_price
            data["sources"]["current_price"] = "Yahoo Finance"
        else:
            data["sources"]["current_price"] = "Screener.in"
            diff_pct = abs(yahoo_price - screener_price) / screener_price * 100 if screener_price else np.nan
            if not np.isnan(diff_pct) and diff_pct > 3:
                data.setdefault("data_quality_notes", []).append(
                    f"Yahoo/Screener price differs by {diff_pct:.1f}%"
                )

    data["market_data"] = market
    data["sources"]["price_history"] = "Yahoo Finance"
    data["source_status"]["Yahoo Finance"] = "ok"
    return data


def _merge_exchange_events(data: dict, events: dict) -> dict:
    data = dict(data)
    data.setdefault("sources", {})
    data.setdefault("source_status", {})
    data.setdefault("data_quality_notes", [])

    if not events:
        data["source_status"]["NSE India"] = "unavailable"
        data["exchange_events"] = {}
        return data

    data["exchange_events"] = events
    errors = events.get("errors") or {}
    source_errors = events.get("source_errors") or {}
    actions = events.get("corporate_actions")
    announcements = events.get("announcements")
    has_actions = actions is not None and hasattr(actions, "empty") and not actions.empty
    has_announcements = announcements is not None and hasattr(announcements, "empty") and not announcements.empty

    if has_actions:
        data["sources"]["corporate_actions"] = events.get("source", "NSE/BSE public endpoints")
    if has_announcements:
        data["sources"]["corporate_announcements"] = events.get("source", "NSE/BSE public endpoints")

    for source in ("NSE India", "BSE India"):
        source_err = source_errors.get(source) or {}
        if not source_err:
            data["source_status"][source] = "ok"
        elif len(source_err) < 2:
            data["source_status"][source] = "partial"
        else:
            data["source_status"][source] = "unavailable: " + "; ".join(
                f"{k}={v}" for k, v in source_err.items()
            )

    if errors and not (has_actions or has_announcements):
        data["source_status"]["Exchange events"] = "unavailable: " + "; ".join(
            f"{k}={v}" for k, v in errors.items()
        )
    elif errors:
        data["source_status"]["Exchange events"] = "partial"
        data["data_quality_notes"].append(
            "Exchange events partially unavailable: "
            + "; ".join(f"{k}={v}" for k, v in errors.items())
        )
    else:
        data["source_status"]["Exchange events"] = "ok"

    return data


def _snapshot_fallback(ticker: str, error: Exception) -> dict:
    store = None
    canonical = normalize_ticker(ticker)
    try:
        store = SnapshotStore()
        snapshot = store.latest_snapshot(canonical)
    except Exception:
        snapshot = None
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    if not snapshot:
        raise error

    data = dict(snapshot.get("data") or {})
    data.setdefault("ticker", canonical)
    data.setdefault("company_name", snapshot.get("company_name") or canonical)
    data.setdefault("sources", {})
    data.setdefault("source_status", {})
    data.setdefault("data_quality_notes", [])
    data["sources"]["financials"] = f"Saved snapshot ({snapshot.get('snapshot_date')})"
    data["sources"]["fundamentals"] = f"Saved snapshot ({snapshot.get('snapshot_date')})"
    data["source_status"]["Screener.in"] = f"unavailable, using saved snapshot: {error}"
    data["data_quality_notes"].append(
        "Live Screener.in was unavailable; report uses the latest saved fundamentals snapshot."
    )
    return data


def fetch_company_data(ticker: str, include_market: bool = True, include_exchange_events: bool = True) -> dict:
    """
    Canonical data fetch used by the app.

    Screener is the primary source for financial statements and ratios. Yahoo
    enriches market history, beta, volatility, and fallback current price. NSE
    public NSE/BSE endpoints add corporate actions and announcements when
    available.
    """
    requested_ticker = str(ticker or "").upper().strip()
    ticker = normalize_ticker(requested_ticker)
    try:
        data = fetch_fundamentals(ticker)
    except Exception as exc:
        data = _snapshot_fallback(ticker, exc)
    if requested_ticker and requested_ticker != ticker:
        data["requested_ticker"] = requested_ticker
    data.setdefault("sources", {})
    data.setdefault("data_quality_notes", [])
    if requested_ticker and requested_ticker != ticker:
        note = f"Input ticker {requested_ticker} was mapped to active symbol {ticker}."
        if note not in data["data_quality_notes"]:
            data["data_quality_notes"].append(note)
    data["sources"].setdefault("financials", "Screener.in")
    data["sources"].setdefault("fundamentals", "Screener.in")
    data["sources"].setdefault("top_ratios", "Screener.in")
    data["sources"].setdefault("peers", "Screener.in / fallback peer map")

    if include_market:
        market = fetch_market_data(ticker)
        data = _merge_market_data(data, market)
    else:
        data["market_data"] = {}

    if include_exchange_events:
        events = fetch_exchange_events(ticker, company_name=data.get("company_name"))
        data = _merge_exchange_events(data, events)
    else:
        data["exchange_events"] = {}
        data["source_status"]["Exchange events"] = "disabled"

    return data


def fetch_peer_data_multi_source(peers: list, delay: float = 1.0, include_market: bool = False) -> list:
    """
    Fetch peer data with the same canonical shape.

    Defaults to Screener-only for speed; market enrichment can be enabled later.
    """
    if not include_market:
        return fetch_peers(peers, delay=delay)

    out = []
    for peer in peers:
        try:
            out.append(fetch_company_data(peer, include_market=True))
        except Exception:
            continue
    return out


def source_summary(data: dict) -> pd.DataFrame:
    rows = []
    for key, source in (data.get("sources") or {}).items():
        rows.append({"Field": key, "Source": source})
    for source, status in (data.get("source_status") or {}).items():
        rows.append({"Field": f"{source} status", "Source": status})
    return pd.DataFrame(rows)
