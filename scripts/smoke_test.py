"""Offline smoke tests for core EquityLens modules.

Run:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from data_sources.aggregator import _merge_exchange_events, _merge_market_data
from ratios import calculate_ratios
from market_ready import build_market_ready_report
from report_pdf import generate_pdf
from excel_export import export_excel
from snapshot_store import SnapshotStore, replay_snapshot_backtest


def main():
    r_data = {
        "ticker": "TEST",
        "company_name": "Test Ltd",
        "current_price": 100,
        "market_cap": 1000,
        "pe_ratio": 20,
        "book_value": 50,
        "dividend_yield": 1,
        "high_52w": 130,
        "low_52w": 70,
        "sector": "Test",
        "pl": pd.DataFrame(),
        "bs": pd.DataFrame(),
        "cf": pd.DataFrame(),
        "sources": {"financials": "Screener.in"},
    }
    dates = pd.date_range("2020-01-01", periods=1400, freq="B")
    trend = np.linspace(100, 220, len(dates))
    cycle = np.sin(np.linspace(0, 28, len(dates))) * 12
    close = trend + cycle
    benchmark = np.linspace(100, 180, len(dates)) + np.sin(np.linspace(0, 18, len(dates))) * 6
    market = {
        "source": "Yahoo Finance",
        "current_price": 101,
        "return_1m_pct": 2.5,
        "return_3m_pct": 8,
        "return_1y_pct": 30,
        "volatility_1y_pct": 22,
        "beta_vs_nifty": 1.1,
        "price_history": pd.DataFrame({"Date": dates, "Close": close, "Volume": 100000}),
        "benchmark_history": pd.DataFrame({"Date": dates, "Close": benchmark}),
    }
    data = _merge_market_data(r_data, market)
    data = _merge_exchange_events(
        data,
        {
            "source": "NSE India",
            "symbol": "TEST",
            "corporate_actions": pd.DataFrame([{"symbol": "TEST", "subject": "Dividend"}]),
            "announcements": pd.DataFrame([{"symbol": "TEST", "desc": "Board Meeting"}]),
            "errors": {},
        },
    )
    r = calculate_ratios(data)
    r.update({
        "years": ["FY1", "FY2", "FY3", "FY4", "FY5"],
        "revenue": [100, 120, 150, 180, 230],
        "ebitda": [20, 25, 35, 45, 60],
        "pat": [10, 12, 18, 25, 33],
        "eps": [5, 6, 9, 12, 16],
        "cfo": [12, 15, 20, 27, 36],
        "fcf": [7, 9, 12, 19, 26],
        "operating_margin": [20, 21, 23, 25, 26],
        "roe": [12, 13, 17, 20, 22],
        "roce": [15, 16, 20, 24, 28],
        "debt_equity": [0.25, 0.2, 0.14, 0.1, 0.07],
        "revenue_cagr_3y": 32,
        "eps_cagr_3y": 38,
    })
    dcf = {
        "base_iv": 125,
        "current_price": 100,
        "upside_pct": 25,
        "signal": "BUY",
        "composite_score": 0.5,
        "wacc_result": {"wacc": 0.11},
        "scenarios": {
            "Bear": {"intrinsic_per_share": 80, "enterprise_value": 800, "pv_terminal_value": 500, "tv_pct_of_ev": 60, "fcff_df": pd.DataFrame()},
            "Base": {"intrinsic_per_share": 125, "enterprise_value": 1200, "pv_terminal_value": 700, "tv_pct_of_ev": 58, "fcff_df": pd.DataFrame({"Year": ["FY+1"], "Revenue": [250], "EBITDA": [70], "FCFF": [40]})},
            "Bull": {"intrinsic_per_share": 170, "enterprise_value": 1600, "pv_terminal_value": 900, "tv_pct_of_ev": 56, "fcff_df": pd.DataFrame()},
        },
    }
    assumptions = {
        "base_growth": 0.15,
        "base_ebitda_margin": 25,
        "base_margin_delta": 0.01,
        "tax_rate": 0.25,
        "capex_pct": 0.05,
        "wc_pct": 0.01,
        "quality_score": 75,
        "business_type": "stable",
    }
    flags = [{"triggered": False, "severity": "low", "check": "Debt check", "detail": "OK"}]
    mr = build_market_ready_report(data, r, flags, None, dcf, {"prob_undervalued": 65}, assumptions)
    assert mr["backtest"]["historical_validation"]["status"] == "Computed"
    pdf = generate_pdf("TEST", data, r, flags, None, dcf, mr)
    xlsx = export_excel("TEST", data, r, flags, None, dcf, mr)
    assert len(pdf) > 10_000
    assert len(xlsx) > 5_000
    store = SnapshotStore(":memory:")
    store.save_snapshot("TEST", data, ratios=r, assumptions=assumptions, dcf_result=dcf, market_ready=mr, snapshot_date="2022-01-03")
    store.save_snapshot("TEST", data, ratios=r, assumptions=assumptions, dcf_result=dcf, market_ready=mr, snapshot_date="2023-01-03")
    replay = replay_snapshot_backtest(store, tickers=["TEST"], persist=False)
    assert replay["summary"] is not None and not replay["summary"].empty
    store.close()
    print("Smoke test passed")


if __name__ == "__main__":
    main()
