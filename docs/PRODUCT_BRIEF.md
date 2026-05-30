# EquityLens Product Brief

## Positioning

EquityLens is an automated Indian equity research workstation for analysts, portfolio teams, and independent research desks. It converts public financial and market data into a structured research report with fundamentals, peer comparison, DCF, Monte Carlo, risk scoring, forecasts, and a buy/hold/sell signal.

## Core Value Proposition

- Reduces first-pass stock research time from hours to minutes.
- Produces analyst-style PDF and Excel reports from one ticker.
- Combines historical fundamentals with market-history overlays.
- Makes assumptions explicit through a source audit, target bridge, and model methodology.
- Supports sector-aware valuation, including exchange/platform businesses like BSE.
- Preserves generated reports as model-versioned SQLite snapshots for audit and reload.

## Current Data Sources

- Screener.in: financial statements, top ratios, peer links, CAGR tables.
- Yahoo Finance: price history, beta, volatility, market-return metrics.
- NSE/BSE public pages: corporate actions and corporate announcements when public endpoints are available.
- Internal calculations: ratios, DCF, Monte Carlo, red flags, quality score, risk score, forecasts, target price bridge.
- Fallback peer map: used only when public peer data is sparse or polluted by benchmark indices.

## Current Architecture

```text
Streamlit UI
  -> data_sources/aggregator.py
     -> Screener.in fundamentals
     -> Yahoo Finance market history
     -> NSE/BSE public event context
  -> ratios/red_flags/dcf/market_ready
  -> PDF and Excel exporters
  -> snapshot_store.py SQLite report history
```

## Key Screens

- Decision: final target, signal confidence, risk score, data quality, source audit.
- Financials: historical P&L, balance sheet, cash flow and growth.
- Ratios: profitability, solvency, liquidity, efficiency and valuation ratios.
- Red Flags: automated financial risk checks.
- Business Quality: quality score and business classification.
- Peers: peer comparison and peer quality ranking.
- DCF Valuation: bull/base/bear intrinsic value, WACC, sensitivity.
- Monte Carlo: valuation probability distribution.
- Report History: reload saved model-versioned snapshots and regenerate PDF/Excel.

## Buyer Personas

- Sell-side research analysts who need fast first-draft coverage.
- Buy-side analysts screening large watchlists.
- Wealth research teams preparing client notes.
- Independent advisors producing standardized equity memos.
- Finance students and research interns learning model construction.

## Commercial Readiness Status

The app is strong as a research automation prototype and demo product. Before institutional deployment, complete:

- Historical signal backtesting across many tickers and dates.
- Data provider licensing review.
- User authentication and role-based access.
- Assumption override audit log.
- In-app validation dashboard for snapshot replay results.
- Production deployment hardening.
