# Equity Research & Valuation Web App

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

## Setup & Run

```bash
# 1. Create a folder and put all files in it
mkdir equity_app && cd equity_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at: http://localhost:8501

## Usage
1. Enter any NSE ticker (e.g. RELIANCE, HDFCBANK, INFY, TCS)
2. Click "Generate Report"
3. View financials, ratios, charts, DCF valuation, Monte Carlo
4. Download PDF report and Excel file

## Notes
- Data is scraped live from Screener.in (consolidated financials)
- DCF model uses real scraped data as base for projections
- Monte Carlo runs 10,000 simulations
- PDF and Excel are generated fresh for every ticker
