# Model Methodology

## Research Pipeline

1. Fetch public financial statements and ratio tables.
2. Enrich the company record with market price history.
3. Add exchange-event context from NSE/BSE corporate actions and announcements when available.
4. Recalculate ratios internally from raw statements.
5. Classify the business type.
6. Score business quality.
7. Derive valuation assumptions from history, Screener CAGR tables, peer medians and business type.
8. Run DCF bull/base/bear scenarios.
9. Run Monte Carlo simulation.
10. Build target price bridge and decision dashboard.
11. Export PDF and Excel research packs.

## Ratio Calculations

The app recalculates ratios instead of blindly trusting scraped display values.

- Profitability: operating margin, net margin, ROE, ROA, ROCE.
- Liquidity: current ratio, quick ratio, cash ratio.
- Solvency: debt/equity, debt/EBITDA, interest coverage.
- Efficiency: asset turnover, receivable days, payable days, inventory days, cash conversion cycle.
- Valuation: P/E, P/B, EV/EBITDA.
- Market history: 1M, 3M and 1Y returns, 1Y volatility, beta vs Nifty.

## Business Classification

The model classifies companies into:

- `exchange-platform`
- `high-margin-stable`
- `stable`
- `cyclical`
- `commodity`

Classification affects growth ceilings, margin assumptions, WACC priors, terminal growth, margin-of-safety requirements and signal logic.

## Quality Score

The quality score is out of 100.

- Financial health: debt/equity, interest coverage, current ratio.
- Profitability: ROCE, margin stability, margin trend.
- Cash-flow quality: CFO/PAT, FCF conversion, positive FCF years.
- Growth quality: revenue CAGR, EPS vs revenue growth, PAT consistency.

## DCF

DCF uses FCFF projections:

```text
Revenue -> EBITDA -> NOPAT -> FCFF
FCFF = NOPAT - Capex - Working Capital Investment
```

Enterprise value is:

```text
PV explicit FCFF + PV terminal value
```

Equity value is:

```text
Enterprise value - debt + cash
```

Intrinsic value per share is:

```text
Equity value / shares outstanding
```

## WACC

WACC uses market-value capital weights where available:

```text
Cost of equity = risk-free rate + beta * equity risk premium
After-tax cost of debt = cost of debt * (1 - tax rate)
WACC = equity weight * cost of equity + debt weight * after-tax cost of debt
```

## Monte Carlo

Monte Carlo randomizes:

- Revenue growth.
- WACC.
- Terminal growth.
- Margin improvement.
- Capex intensity.
- Working-capital intensity.

Outputs:

- Mean, median, P10, P25, P75, P90 valuation.
- Probability of undervaluation.
- Monte Carlo signal.

## Sector-Specific Valuation

For exchange/platform companies such as BSE, pure FCFF DCF may understate re-rating potential. The app adds a broker-style FY+2 EPS multiple model:

```text
Target price = FY+2 EPS * target P/E multiple
```

The final signal blends platform valuation, DCF, P/E mean reversion and EV/EBITDA.

The market-ready layer also selects sector-specific models:

| Category | Examples | Primary model |
|---|---|---|
| Banks/NBFCs | HDFCBANK, ICICIBANK, BAJAJHFL | P/B model driven by ROE, book value and credit-growth proxy |
| Exchanges | BSE, MCX-like businesses | Forward EPS/platform multiple plus DCF |
| IT Services | INFY, TCS, HCLTECH | Forward P/E adjusted for growth and quality |
| FMCG/Consumer | Consumer compounders | Quality-adjusted forward P/E |
| Metals/Commodity | Steel, metals, oil & gas cyclicals | Cycle-adjusted EV/EBITDA |
| Real Estate | Developers/asset-heavy real estate | NAV/P/B proxy |
| Insurance/AMC | Insurers, AMCs | Forward earnings proxy until AUM/VNB data exists |
| General Corporate | Other operating companies | Growth-adjusted fair P/E plus DCF |

## Decision Layer

The Decision tab combines:

- Blended target price.
- Signal confidence.
- Risk score.
- Data quality score.
- Exchange event context.
- Model/source version stored in report history.
- Technical 52-week overlay.
- Forecast financials.
- Peer ranking.
- Backtest readiness.

## Important Limitations

- The output is not investment advice; users should consult a SEBI-registered investment advisor or qualified financial professional before making investment decisions.
- Public data can be stale or incomplete.
- Scraped pages can change structure.
- Full institutional-grade credibility requires historical signal backtesting.
- Analyst judgement is still required for management quality, governance, regulatory risk, and industry structure.
