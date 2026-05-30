# EquityLens

EquityLens is an automated Indian equity research and valuation workstation. It turns a stock ticker into an analyst-style research pack containing fundamentals, ratios, red flags, peer comparison, DCF valuation, Monte Carlo simulation, target price bridge, risk scoring, signal confidence, forecasts, PDF report and Excel workbook.

## What It Does

- Fetches public company data from Screener.in.
- Enriches market history through Yahoo Finance.
- Adds NSE/BSE corporate actions and announcements when public endpoints are available.
- Recalculates financial ratios internally.
- Detects accounting and balance-sheet red flags.
- Classifies business type and scores business quality.
- Derives DCF assumptions from historical data and source CAGR tables.
- Runs bull/base/bear DCF and Monte Carlo simulations.
- Adds sector-aware valuation, including exchange/platform valuation for BSE-like businesses.
- Builds a market-ready Decision tab with target price, risk, confidence, data quality and backtest readiness.
- Saves model-versioned report snapshots to local SQLite history.
- Exports detailed PDF and Excel research reports.

## Current Data Sources

| Source | Usage |
|---|---|
| Screener.in | Financial statements, ratios, CAGR tables, peer links |
| Yahoo Finance | Price history, returns, volatility, beta vs Nifty |
| NSE/BSE public pages | Corporate actions and corporate announcements |
| Internal model | Ratios, DCF, Monte Carlo, target bridge, risk, confidence |
| Fallback peer map | Backup peer universe when public peer data is sparse |

## Project Structure

```text
equity_app/
  app.py                         Streamlit application
  scraper.py                     Screener parser and peer fallback logic
  ratios.py                      Financial ratio calculations
  dcf.py                         DCF, Monte Carlo, business classification, signal logic
  market_ready.py                Decision layer, target bridge, risk, confidence, forecasts
  report_pdf.py                  Detailed PDF export
  excel_export.py                Detailed Excel export
  charts.py                      Plotly charts
  red_flags.py                   Red flag checks
  snapshot_store.py              SQLite report history and point-in-time snapshots
  data_sources/
    aggregator.py                Multi-source canonical company object
    exchange_source.py           NSE/BSE corporate actions and announcements adapter
    screener_source.py           Screener adapter
    yahoo_source.py              Yahoo Finance adapter
  docs/
    PRODUCT_BRIEF.md
    MODEL_METHODOLOGY.md
    DATA_SOURCES_AND_COMPLIANCE.md
    DEPLOYMENT_RUNBOOK.md
    VALIDATION_BACKTEST_PLAN.md
    COMMERCIAL_READINESS_CHECKLIST.md
    SECURITY_AND_PRIVACY.md
  scripts/
    smoke_test.py
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

No API keys are required for the default setup. The app uses Screener.in for fundamentals, Yahoo Finance for market history, and public NSE/BSE pages for corporate actions/announcements. Exchange responses are cached locally under `.cache/` to reduce repeated website calls. MSN can be added later as a fallback only if Yahoo coverage becomes insufficient.

Open:

```text
http://localhost:8501
```

## Demo Tickers

- `RELIANCE`: diversified large-cap.
- `BAJAJHFL`: housing finance/NBFC.
- `BSE`: exchange-platform valuation.
- `INFY`: IT services.
- `TCS`: high-margin stable business.

## Report Output

### Report History

Every generated report is saved as a model-versioned SQLite snapshot in:

```text
data/equitylens_snapshots.sqlite
```

The sidebar Report History panel can reload saved reports and regenerate PDF/Excel outputs from stored snapshot data. Saved snapshots include the model version, source version, company data, ratios, peer table, assumptions, DCF result and market-ready decision layer.

### PDF

The PDF is a thesis-style research report covering:

- Company overview.
- Financial summary.
- Ratio snapshot.
- Red flags.
- Trend charts.
- DuPont decomposition.
- DCF valuation.
- Peer comparison.
- Investment decision dashboard.
- Target price bridge.
- Forecast financial statements.
- Risk and confidence thesis.
- Peer ranking.
- Methodology and source audit.

### Excel

The Excel workbook includes:

- Summary.
- Raw P&L.
- Raw balance sheet.
- Raw cash flow.
- All ratios.
- Red flags.
- Peer comparison.
- DCF valuation.
- Decision thesis.
- Target bridge.
- Forecasts.
- Risk and confidence.
- Peer ranking.
- Data sources.
- Price history.
- Methodology.

## Smoke Test

Run the offline smoke test:

```powershell
python scripts/smoke_test.py
```

This verifies:

- Multi-source merge logic.
- Ratio calculation.
- Market-ready report builder.
- PDF generation.
- Excel generation.

## Commercial Readiness

EquityLens is suitable as a strong analyst-workflow pilot or demo product. Before selling as an institutional production platform, complete:

- Licensed data-source review.
- True point-in-time signal backtesting.
- User authentication.
- Assumption override audit log.
- Deployment hardening.

See:

- [Product Brief](docs/PRODUCT_BRIEF.md)
- [Model Methodology](docs/MODEL_METHODOLOGY.md)
- [Workflow And Signal Generation](docs/WORKFLOW_AND_SIGNAL_GENERATION.md)
- [Data Sources And Compliance](docs/DATA_SOURCES_AND_COMPLIANCE.md)
- [Validation And Backtesting Plan](docs/VALIDATION_BACKTEST_PLAN.md)
- [Deployment Runbook](docs/DEPLOYMENT_RUNBOOK.md)
- [Commercial Readiness Checklist](docs/COMMERCIAL_READINESS_CHECKLIST.md)
- [Security And Privacy](docs/SECURITY_AND_PRIVACY.md)

## Disclaimer

This project is an automated research aid and is not investment advice. Consult a SEBI-registered investment advisor or qualified financial professional before making investment decisions. All outputs must be reviewed by a qualified analyst before use in investment decisions or client-facing research.
