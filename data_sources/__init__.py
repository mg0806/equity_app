"""Data-source adapters and aggregation for EquityLens."""

from .aggregator import fetch_company_data, fetch_peer_data_multi_source
from .exchange_source import fetch_exchange_events

__all__ = [
    "fetch_company_data",
    "fetch_peer_data_multi_source",
    "fetch_exchange_events",
]
