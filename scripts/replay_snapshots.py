"""Replay saved point-in-time snapshots and export validation evidence.

Examples:
    python scripts/replay_snapshots.py
    python scripts/replay_snapshots.py --tickers BSE RELIANCE --start 2024-01-01 --end 2026-03-31
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from snapshot_store import DEFAULT_DB_PATH, SnapshotStore, replay_snapshot_backtest


def main():
    parser = argparse.ArgumentParser(description="Replay EquityLens point-in-time snapshots.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--tickers", nargs="*", help="Optional tickers")
    parser.add_argument("--start", help="Start snapshot date")
    parser.add_argument("--end", help="End snapshot date")
    parser.add_argument("--run-name", help="Replay run name")
    parser.add_argument("--output", default="snapshot_replay_results.xlsx")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    store = SnapshotStore(args.db)
    try:
        replay = replay_snapshot_backtest(
            store,
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            run_name=args.run_name,
            persist=not args.no_persist,
        )
        table = replay["table"]
        summary = replay["summary"]
        out = Path(args.output)
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            if summary is not None and not summary.empty:
                summary.to_excel(writer, sheet_name="summary", index=False)
            if table is not None and not table.empty:
                table.to_excel(writer, sheet_name="replay_results", index=False)
            inv = store.inventory()
            if not inv.empty:
                inv.to_excel(writer, sheet_name="snapshot_inventory", index=False)
        print(f"Run: {replay['run_name']}")
        print(summary.to_string(index=False) if summary is not None and not summary.empty else "No replay rows")
        print(f"Wrote {out.resolve()}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
