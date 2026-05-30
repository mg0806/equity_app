"""Capture point-in-time EquityLens snapshots into SQLite.

Examples:
    python scripts/capture_snapshots.py --tickers BSE RELIANCE INFY
    python scripts/capture_snapshots.py --tickers-file tickers.txt --snapshot-date 2026-03-31
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_sources import fetch_company_data, fetch_peer_data_multi_source
from dcf import derive_assumptions_from_screener, run_three_scenarios
from ratios import build_peer_comparison, calculate_ratios
from red_flags import detect_red_flags
from market_ready import build_market_ready_report
from snapshot_store import DEFAULT_DB_PATH, DEFAULT_MODEL_VERSION, DEFAULT_SOURCE_VERSION, SnapshotStore


def _read_tickers(args):
    tickers = list(args.tickers or [])
    if args.tickers_file:
        tickers.extend(
            line.strip().upper()
            for line in Path(args.tickers_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return sorted(set(t.upper().strip() for t in tickers if t.strip()))


def capture_one(ticker, peer_limit=3, include_peers=True):
    data = fetch_company_data(ticker, include_market=True)
    ratios = calculate_ratios(data)
    peer_df = pd.DataFrame()
    if include_peers and data.get("peers"):
        peer_data = fetch_peer_data_multi_source(data["peers"][:peer_limit], include_market=False)
        if peer_data:
            peer_df = build_peer_comparison(ticker, data, ratios, peer_data)
    assumptions = derive_assumptions_from_screener(data, ratios, peer_df)
    try:
        dcf_result = run_three_scenarios(data, ratios, assumptions)
    except Exception as exc:
        dcf_result = {"signal": "N/A", "error": str(exc)}
    flags = detect_red_flags(ratios)
    market_ready = build_market_ready_report(data, ratios, flags, peer_df, dcf_result, None, assumptions)
    return data, ratios, peer_df, assumptions, dcf_result, market_ready


def main():
    parser = argparse.ArgumentParser(description="Capture point-in-time EquityLens snapshots.")
    parser.add_argument("--tickers", nargs="*", default=[], help="Tickers to capture")
    parser.add_argument("--tickers-file", help="File with one ticker per line")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--snapshot-date", help="Override snapshot date, e.g. quarter end 2026-03-31")
    parser.add_argument("--fiscal-period", help="Fiscal label, e.g. FY26Q4")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--peer-limit", type=int, default=3)
    parser.add_argument("--no-peers", action="store_true")
    args = parser.parse_args()

    tickers = _read_tickers(args)
    if not tickers:
        raise SystemExit("Provide --tickers or --tickers-file")

    store = SnapshotStore(args.db)
    try:
        errors = []
        for ticker in tickers:
            try:
                print(f"Capturing {ticker}...")
                data, ratios, peer_df, assumptions, dcf_result, market_ready = capture_one(
                    ticker,
                    peer_limit=args.peer_limit,
                    include_peers=not args.no_peers,
                )
                store.save_snapshot(
                    ticker,
                    data,
                    ratios=ratios,
                    peer_df=peer_df,
                    assumptions=assumptions,
                    dcf_result=dcf_result,
                    market_ready=market_ready,
                    snapshot_date=args.snapshot_date,
                    fiscal_period=args.fiscal_period,
                    source_version=args.source_version,
                    model_version=args.model_version,
                )
                print(f"Saved {ticker} snapshot")
            except Exception as exc:
                errors.append({"Ticker": ticker, "Error": str(exc)})
                print(f"ERROR {ticker}: {exc}")
        inv = store.inventory()
        print(inv.to_string(index=False) if not inv.empty else "No snapshots saved")
        if errors:
            print("Errors:")
            print(pd.DataFrame(errors).to_string(index=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
