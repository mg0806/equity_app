# Workflow And Signal Generation

This document explains how EquityLens moves from a stock ticker to a research report, BUY/HOLD/SELL signal, validation evidence and point-in-time replay.

## Current End-To-End Flow

```text
User enters ticker
  -> app fetches Screener + Yahoo + NSE/BSE event data through the data-source aggregator
  -> Screener tables, quarterly results and peers are parsed
  -> ratios are recalculated internally
  -> peers are cleaned, fetched and ranked
  -> red flags are checked
  -> DCF assumptions are derived from Screener, history and peers
  -> DCF scenarios and optional Monte Carlo simulation are run
  -> market-ready decision layer is built
  -> report inputs and model/source versions are saved to SQLite
  -> historical signal validation is run from price history
  -> PDF and Excel reports are generated
```

## Point-In-Time Research Flow

For institutional backtesting, the project now supports a separate snapshot workflow:

```text
Capture date / quarter end
  -> fetch company, quarterly, peer and price data
  -> calculate ratios, assumptions, DCF and decision layer
  -> save immutable snapshot to SQLite
  -> later replay only snapshots available at that historical date
  -> measure forward 1M / 3M / 6M / 12M return and drawdown
  -> summarize BUY / HOLD / SELL hit rate and median returns
```

This is handled by:

- `snapshot_store.py`
- `scripts/capture_snapshots.py`
- `scripts/replay_snapshots.py`
- `docs/POINT_IN_TIME_SNAPSHOT_DATABASE.md`

The Streamlit app also exposes saved reports in the sidebar Report History panel. Loading a saved report reconstructs PDF/Excel outputs from the stored snapshot instead of fetching live data.

Saved reports use:

```text
model_version = equitylens_v1.1_sector_nse_bse_events
source_version = screener_yahoo_nse_bse_v1
```

## Main Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, orchestration, report cache, downloads |
| `data_sources/aggregator.py` | Creates one canonical company object from multiple sources |
| `data_sources/exchange_source.py` | NSE/BSE corporate actions and announcements adapter |
| `data_sources/screener_source.py` | Screener fundamentals adapter |
| `data_sources/yahoo_source.py` | Yahoo price history, volatility and beta adapter |
| `scraper.py` | Screener parsing, quarterly table parsing, peers, fallback peer logic |
| `ratios.py` | Recalculates financial ratios from raw statements |
| `red_flags.py` | Accounting and balance-sheet warning checks |
| `dcf.py` | Business classification, assumptions, DCF, Monte Carlo and core signal |
| `market_ready.py` | Quarterly checks, sector valuation, target bridge, confidence, explainability |
| `validation_engine.py` | Rolling historical signal validation from market history |
| `snapshot_store.py` | SQLite point-in-time snapshot capture and replay |
| `report_pdf.py` | Thesis-style PDF export |
| `excel_export.py` | Detailed Excel workbook export |

## Data Sources

The app calls:

```python
fetch_company_data(ticker, include_market=True)
```

The canonical `data` dictionary combines:

- Screener financial statements.
- Screener quarterly results.
- Screener top ratios.
- Screener peers and fallback peers.
- Yahoo price history.
- Yahoo benchmark history.
- Yahoo volatility and beta.
- NSE/BSE corporate actions and announcements when available.
- Source status and data-quality notes.

If Yahoo or NSE/BSE event endpoints fail, the app continues with Screener fundamentals. If Screener peers are indices such as Nifty, Sensex or BSE 500, they are filtered out.

## Ratio Calculation

The app calls:

```python
calculate_ratios(data)
```

Ratio groups include:

- Profitability: operating margin, net margin, ROE, ROA, ROCE.
- Liquidity: current ratio, quick ratio, cash ratio.
- Solvency: debt/equity, debt/EBITDA, interest coverage.
- Efficiency: asset turnover, receivable days, payable days, inventory days.
- Valuation: P/B, EV/EBITDA.
- Growth: revenue CAGR, PAT CAGR, EPS CAGR.
- Market history: 1M/3M/1Y return, volatility and beta.

## Peer Workflow

Peers are built through:

```python
fetch_peer_data_multi_source(...)
build_peer_comparison(...)
```

The peer layer:

- Removes index/benchmark rows.
- Uses fallback peers for common sectors when Screener peers are weak.
- Calculates peer comparison ratios.
- Ranks peers using quality, growth, valuation, leverage and market-cap similarity.

## Business Classification

`dcf.py` classifies the company as:

- `exchange-platform`
- `high-margin-stable`
- `stable`
- `cyclical`
- `commodity`

The market-ready layer also maps the company to valuation sectors:

- Banks/NBFCs.
- Exchanges.
- IT services.
- FMCG/Consumer.
- Metals/Commodity.
- Real Estate.
- Insurance/AMC.
- General Corporate.

## Assumption Derivation

The app calls:

```python
derive_assumptions_from_screener(data, r, peer_df)
```

Assumptions include:

- Base/bear/bull revenue growth.
- EBITDA margin and margin change.
- WACC, beta and cost of debt.
- Tax rate.
- Capex and working-capital intensity.
- Terminal growth.
- Business quality score.
- Required margin of safety.

Inputs include Screener CAGR tables, historical growth, peer margins, debt/equity, tax/PBT, interest/debt and business-type rules.

## DCF And Monte Carlo

The DCF engine runs:

```python
run_three_scenarios(data, r, assumptions)
```

Each scenario projects:

```text
Revenue -> EBITDA -> NOPAT -> FCFF -> Enterprise Value -> Equity Value -> Intrinsic Value / Share
```

Monte Carlo optionally runs:

```python
run_monte_carlo(data, r, assumptions)
```

It randomizes growth, WACC, terminal growth, margins, capex and working capital, then estimates undervaluation probability.

## Sector-Specific Valuation

`market_ready.py` now selects sector valuation models:

| Sector | Main Logic |
|---|---|
| Banks/NBFCs | P/B based on ROE, book value and credit-growth proxy |
| Exchanges | Forward EPS multiple / platform valuation |
| IT Services | Forward P/E adjusted for growth and quality |
| FMCG/Consumer | Forward P/E adjusted for ROCE and growth |
| Metals/Commodity | Cycle-adjusted EV/EBITDA proxy |
| Real Estate | NAV/P/B proxy |
| Insurance/AMC | Forward earnings proxy until AUM/VNB data exists |
| General Corporate | Growth-adjusted fair P/E |

This prevents DCF from being the only valuation model for every business type.

## Quarterly And TTM Signal Layer

The Screener quarterly table is parsed and converted into:

- Latest quarterly revenue.
- Latest quarterly PAT.
- Latest quarterly EPS.
- Latest operating margin.
- YoY revenue growth.
- QoQ revenue growth.
- YoY PAT growth.
- QoQ PAT growth.
- TTM revenue, PAT and EPS.
- TTM margin expansion/contraction.

This improves signal timing versus annual-only financials.

## Valuation Bands

The market-ready layer calculates historical valuation context using price history:

- 5Y price percentile.
- Drawdown from historical high.
- Approximate P/E band.
- Approximate P/B band.
- Current valuation status: discounted, near band or expensive.

## Earnings Revision Proxy

Because broker targets move when earnings expectations move, the app now estimates upgrade/downgrade bias from:

- Revenue growth acceleration.
- PAT growth acceleration.
- EPS growth acceleration.
- Latest quarterly revenue YoY.
- Latest quarterly PAT YoY.
- TTM margin change.

Output:

- `Upgrade Bias`
- `Neutral`
- `Downgrade Risk`

## Momentum Overlay

Market behavior is used as a secondary timing check:

- 50 DMA.
- 200 DMA.
- 6M return.
- Relative strength versus Nifty.
- Drawdown from 52-week high.
- Volume trend.

Output:

- `Bullish`
- `Neutral`
- `Bearish`

## Composite Signal Logic

The core DCF signal still considers:

- DCF upside versus current price.
- Required margin of safety.
- Business quality score.
- P/E mean reversion.
- EV/EBITDA normalization.
- Exchange/platform forward EPS model when relevant.
- Monte Carlo validation.

The market-ready layer then adds:

- Sector-specific target.
- Quarterly trend.
- Valuation band status.
- Earnings revision proxy.
- Momentum overlay.
- Risk score.
- Data/source reliability.
- Historical validation evidence.

## BUY / HOLD / SELL Interpretation

### BUY

Generated when the valuation upside is attractive, quality is acceptable, risk is controlled and validation/Monte Carlo evidence is supportive.

### HOLD

Generated when upside exists but is not strong enough, model evidence is mixed, data confidence is medium/low, or risk offsets valuation upside.

### SELL

Generated when valuation is stretched, downside risk is high, fundamentals are weak, momentum/revision evidence is negative, or the target price is below current price.

### BUY Block

If business quality is too weak, the model blocks BUY even when apparent upside exists.

## Market-Ready Decision Layer

The app calls:

```python
build_market_ready_report(...)
```

It produces:

- Data quality score.
- Quarterly/TTM snapshot.
- Forecast financial statements.
- Sector-specific valuation.
- Blended target price.
- Valuation bands.
- Earnings revision signal.
- Momentum overlay.
- Peer ranking.
- Source reliability score.
- Risk score.
- Signal confidence score.
- Historical validation output.
- Top reasons and top risks.

## Blended Target Price

The target bridge may include:

- Sector-specific model.
- DCF base case.
- Platform EPS model.
- Fair P/E.
- Fair P/B.
- Peer median P/E.

Each component has a weight and rationale.

## Signal Confidence

Confidence considers:

- Data quality.
- Source reliability.
- Composite signal strength.
- Monte Carlo agreement.
- Momentum overlay.
- Earnings revision signal.
- Historical validation hit rate.
- Risk score.

Ratings:

- High.
- Medium.
- Low.

## Explainability

Every signal includes:

- Top 5 positive reasons.
- Top 5 risks.

Examples:

- ROCE is strong.
- 3Y EPS CAGR is high.
- Debt/equity is low.
- Monte Carlo undervaluation probability is supportive.
- Valuation is high versus historical band.
- Momentum overlay is bearish.
- Source reliability is low.

## Historical Validation Engine

The app runs:

```python
run_historical_validation(data, r)
```

It replays rolling signal states from price history and calculates:

- 1M forward return.
- 3M forward return.
- 6M forward return.
- 12M forward return.
- Hit rate.
- Median return.
- Excess return versus benchmark.
- Forward max drawdown.
- Performance by market regime.
- Performance by sector.
- Performance by market-cap bucket.

The output creates buyer-facing evidence such as:

```text
Over 5 years, BUY signals delivered 18% median 12M return with 62% hit rate.
```

## Point-In-Time Snapshot Database

The project now supports SQLite-backed point-in-time storage.

Capture snapshots:

```bash
python scripts/capture_snapshots.py --tickers BSE RELIANCE INFY --snapshot-date 2026-03-31 --fiscal-period FY26Q4
```

Replay snapshots:

```bash
python scripts/replay_snapshots.py --output snapshot_replay_results.xlsx
```

Snapshot replay:

- Uses only stored data from that snapshot.
- Re-runs DCF and market-ready signal generation.
- Measures later forward returns from available price history.
- Stores replay results in SQLite.
- Exports summary to Excel.

This is the foundation for institutional-grade validation without look-ahead bias.

## Report Generation

The app generates:

```python
generate_pdf(...)
export_excel(...)
```

PDF includes:

- Investment thesis dashboard.
- Financial summary.
- Ratio snapshot.
- Red flags.
- DCF valuation.
- Peer comparison.
- Forecasts.
- Target bridge.
- Quarterly/TTM overlay.
- Valuation band and momentum overlay.
- Signal reasons and risks.
- Historical validation summary.
- Methodology and source audit.

Excel includes:

- Raw financial statements.
- All ratios.
- DCF projections.
- Decision thesis.
- Target bridge.
- Forecasts.
- Risk and confidence notes.
- Explainability.
- Quarterly/TTM data.
- Valuation bands.
- Momentum and revision sheets.
- Sector model.
- Historical validation trades and summaries.
- Peer ranking.
- Data sources.
- Price history.
- Methodology.

## Example Signal Flow

```text
Quality score = 76
Business type = exchange-platform
Quarterly PAT YoY = strong
Revision proxy = Upgrade Bias
Momentum = Bullish
Sector/platform target upside = attractive
Monte Carlo support = acceptable
Historical BUY hit rate = supportive
Risk score = Medium

Result = BUY
Reason = Upside exceeds required margin of safety, quality is acceptable, quarterly trend is improving, and validation evidence supports the signal.
```

## Important Limitations

- This system is an automated research aid, not investment advice; users should consult a SEBI-registered investment advisor or qualified financial professional before making investment decisions.
- It does not guarantee returns.
- Screener and Yahoo source structures can change.
- Web-scraped data may be incomplete.
- Snapshot replay becomes stronger only as the snapshot database grows over real quarters.
- Full 9.5/10 institutional validation still requires a broad survivorship-bias-controlled universe and licensed historical consensus estimates.
