"""Screener.in adapter."""

from scraper import fetch_screener_data, fetch_peer_data


def fetch_fundamentals(ticker: str) -> dict:
    data = fetch_screener_data(ticker)
    data.setdefault("sources", {})
    data["sources"]["fundamentals"] = "Screener.in"
    data["sources"]["peers"] = "Screener.in / fallback peer map"
    return data


def fetch_peers(peers: list, delay: float = 1.0) -> list:
    return fetch_peer_data(peers, delay=delay)
