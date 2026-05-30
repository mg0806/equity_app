"""Run historical validation for a list of NSE tickers.

Examples:
    python scripts/run_validation.py --tickers BSE RELIANCE INFY HDFCBANK
    python scripts/run_validation.py --tickers-file tickers.txt --output validation_results.xlsx

The script fetches current canonical data, runs EquityLens historical signal
validation, and writes combined summary/trade sheets to Excel.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_sources import fetch_company_data
from ratios import calculate_ratios
from validation_engine import combine_validation_results, run_historical_validation


def _read_tickers(args):
    tickers = list(args.tickers or [])
    if args.tickers_file:
        path = Path(args.tickers_file)
        tickers.extend(
            line.strip().upper()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return sorted(set(t.upper().strip() for t in tickers if t.strip()))


def main():
    parser = argparse.ArgumentParser(description="Run EquityLens historical validation.")
    parser.add_argument("--tickers", nargs="*", default=[], help="Ticker symbols, e.g. BSE RELIANCE INFY")
    parser.add_argument("--tickers-file", help="Text file with one ticker per line")
    parser.add_argument("--output", default="validation_results.xlsx", help="Excel output path")
    args = parser.parse_args()

    tickers = _read_tickers(args)
    if not tickers:
        raise SystemExit("Provide --tickers or --tickers-file")

    results = []
    errors = []
    for ticker in tickers:
        try:
            print(f"Validating {ticker}...")
            data = fetch_company_data(ticker, include_market=True)
            r = calculate_ratios(data)
            results.append(run_historical_validation(data, r))
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)})

    combined = combine_validation_results(results)
    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name in ["summary", "by_regime", "by_sector", "by_market_cap", "trades"]:
            df = combined.get(name)
            if df is not None and hasattr(df, "empty") and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="errors", index=False)

    print(combined.get("headline", "Validation complete."))
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
