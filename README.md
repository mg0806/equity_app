# Equity Research & Valuation Web App

## Abstract

Equity Research & Valuation Web App is an automated financial analysis platform that performs comprehensive fundamental analysis on Indian NSE-listed companies. Built with Streamlit, the application integrates live data scraping from Screener.in, calculates 20+ financial ratios, identifies 10 risk indicators, and employs DCF valuation with Monte Carlo simulations (10,000 iterations) to determine intrinsic value. Users can generate professional PDF reports and Excel exports with a single click, enabling data-driven investment decisions. The tool democratizes institutional-grade equity research for individual investors.

## Purpose

This is a comprehensive equity research and valuation tool designed for fundamental analysis of Indian NSE-listed companies. The application automates the process of:

- **Data Collection**: Scrapes live financial data (balance sheet, income statement, cash flow) from Screener.in
- **Financial Analysis**: Calculates 20+ financial ratios covering profitability, liquidity, solvency, and efficiency
- **Risk Assessment**: Identifies red flags through 10 automated checks (debt levels, profitability trends, liquidity issues, etc.)
- **Valuation**: Performs DCF (Discounted Cash Flow) valuation with Monte Carlo simulations (10,000 iterations) and sensitivity analysis
- **Reporting**: Generates professional PDF reports and Excel exports with all analysis in a single click

The tool provides institutional-grade financial analysis accessible to individual investors.

## Folder Structure
```
equity_app/
├── app.py                  ← Streamlit entry point (run this)
├── scraper.py              ← Screener.in data scraping
├── ratios.py               ← 20+ financial ratio calculations
├── red_flags.py            ← Red flag detection (10 checks)
├── dcf.py                  ← DCF valuation + Monte Carlo + Sensitivity
├── report_pdf.py           ← 2-page PDF report generator
├── excel_export.py         ← Excel export with all data
├── charts.py               ← All Plotly charts for the UI
├── requirements.txt
└── README.md
```

## Installation & Setup

```bash
# 1. Create a folder and put all files in it
mkdir equity_app && cd equity_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at: http://localhost:8501

## Steps to Use

### Step 1: Enter Stock Ticker
- Open the web app in your browser (http://localhost:8501)
- Enter any NSE ticker in the input field (examples: RELIANCE, HDFCBANK, INFY, TCS, WIPRO)

### Step 2: Generate Report
- Click the "Generate Report" button
- The app will:
  - Fetch live financial data from Screener.in
  - Calculate financial ratios and metrics
  - Run DCF valuation model with 10,000 Monte Carlo simulations
  - Identify potential red flags
  - Generate interactive charts

### Step 3: View Analysis
- **Financials Tab**: View 10+ years of historical balance sheet, income statement, and cash flow data
- **Ratios Tab**: Analyze 20+ financial ratios (P/E, ROE, Debt-to-Equity, Current Ratio, etc.)
- **Red Flags Tab**: Check for 10 identified risk indicators in the company's financials
- **DCF Valuation Tab**: Review:
  - Intrinsic value calculation based on free cash flow projections
  - Valuation range (conservative to optimistic scenarios)
  - Fair value per share

### Step 4: Monte Carlo & Sensitivity
- View 10,000 simulation results showing probability distribution of valuations
- Analyze sensitivity of valuation to discount rate and growth assumptions

### Step 5: Download Reports
- **PDF Report**: Download a professional 2-page summary with key metrics and valuation
- **Excel Export**: Export all data (financials, ratios, projections) for further analysis

## Output

The application generates the following outputs:

### 1. **Interactive Web Interface**
   - Real-time data visualization with Plotly charts
   - Responsive dashboard accessible from any browser
   - Multiple tabs for different analysis views

### 2. **Financial Metrics**
   - 10+ years of historical financials (Balance Sheet, Income Statement, Cash Flow)
   - 20+ calculated financial ratios
   - Key metrics: Revenue, Profit, EPS, Book Value, etc.

### 3. **Valuation Results**
   - DCF intrinsic value per share
   - Valuation range (Min-Max)
   - Current price vs fair value comparison
   - Monte Carlo probability distribution of valuations

### 4. **Risk Assessment**
   - 10 red flags checked against company's financials
   - Debt sustainability analysis
   - Profitability trends
   - Liquidity position assessment

### 5. **PDF Report**
   - Professional 2-page report containing:
     - Key financial metrics
     - Valuation summary
     - Red flags summary
     - Price recommendation based on DCF analysis
   - Saved as: `{TICKER}_Report.pdf`

### 6. **Excel Export**
   - Complete data dump including:
     - Historical financials
     - Calculated ratios
     - DCF projections
     - Sensitivity analysis tables
   - Saved as: `{TICKER}_Data.xlsx`

## Notes
- Data is scraped live from Screener.in (consolidated financials)
- DCF model uses real scraped data as base for projections
- Monte Carlo runs 10,000 simulations for robust valuation ranges
- PDF and Excel are generated fresh for every ticker
- All calculations use standard financial formulas and methodologies
