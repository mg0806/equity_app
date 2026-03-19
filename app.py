"""
app.py — Equity Research & Valuation Web UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="EquityLens — NSE Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0A0F1E; }
section[data-testid="stSidebar"] { background-color: #0D1424; border-right: 1px solid #1E2A45; }
[data-testid="metric-container"] { background-color: #111827; border: 1px solid #1E2A45; border-radius: 10px; padding: 16px; }
.stTabs [data-baseweb="tab-list"] { background-color: #111827; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #8B9AB5; border-radius: 6px; font-weight: 500; }
.stTabs [aria-selected="true"] { background-color: #1B4F8A !important; color: white !important; }
.stDownloadButton > button { background-color: #1B4F8A; color: white; border: none; border-radius: 8px; font-weight: 600; padding: 10px 20px; width: 100%; }
.stDownloadButton > button:hover { background-color: #2563EB; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #1B4F8A, #2563EB); color: white; border: none; border-radius: 10px; font-weight: 700; font-size: 16px; padding: 14px 28px; width: 100%; letter-spacing: 0.5px; box-shadow: 0 4px 20px rgba(27,79,138,0.4); }
.flag-high { background: rgba(231,76,60,0.12); border-left: 4px solid #E74C3C; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.flag-medium { background: rgba(243,156,18,0.12); border-left: 4px solid #F39C12; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.flag-ok { background: rgba(39,174,96,0.12); border-left: 4px solid #27AE60; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.signal-buy  { background:#27AE60; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-sell { background:#E74C3C; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-hold { background:#F39C12; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-na   { background:#95A5A6; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.sec-title { font-size:18px; font-weight:700; color:#E2E8F0; margin-bottom:12px; padding-bottom:6px; border-bottom:2px solid #1B4F8A; }
.card { background:#111827; border:1px solid #1E2A45; border-radius:10px; padding:20px; margin-bottom:16px; }
.chart-tip { text-align:right; font-size:11px; color:#8B9AB5; margin-top:-6px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

from scraper import fetch_screener_data, fetch_peer_data
from ratios import calculate_ratios, build_peer_comparison, last_valid
from red_flags import detect_red_flags, flags_summary
from dcf import run_three_scenarios, run_monte_carlo, run_sensitivity
from report_pdf import generate_pdf
from excel_export import export_excel
from charts import (chart_financial_trend, chart_cashflow_trend, chart_margins,
                    chart_dupont, chart_balance_sheet, chart_dcf_waterfall,
                    chart_scenario_comparison, chart_monte_carlo, chart_tornado,
                    chart_peer_radar)


# ── Helpers ──
def fmt(val, suffix="", prefix="", dec=1, na="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    try:
        return f"{prefix}{float(val):,.{dec}f}{suffix}"
    except:
        return na

def H(tag, content, cls="", style="", **attrs):
    """Safe HTML helper - uses double quotes internally, no escaping needed."""
    attr_str = ""
    if cls:
        attr_str += f' class="{cls}"'
    if style:
        attr_str += f' style="{style}"'
    for k, v in attrs.items():
        attr_str += f' {k.replace("_","-")}="{v}"'
    return f"<{tag}{attr_str}>{content}</{tag}>"

def chart_hint():
    st.markdown(
        '<div class="chart-tip">💡 <em>Click ⛶ (top-right of chart) for fullscreen &nbsp;|&nbsp; '
        'Use Pan &amp; Zoom on the toolbar to explore</em></div>',
        unsafe_allow_html=True,
    )

def plotly(fig):
    st.plotly_chart(fig, use_container_width=True)
    chart_hint()

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


# ── SIDEBAR ──
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:20px 0 10px;">'
        '<div style="font-size:36px;">📊</div>'
        '<div style="font-size:20px; font-weight:700; color:#E2E8F0;">EquityLens</div>'
        '<div style="font-size:12px; color:#8B9AB5; margin-top:4px;">NSE Research Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    ticker_input = st.text_input(
        "🔍 NSE Ticker Symbol",
        value="",
        placeholder="e.g. RELIANCE, HDFCBANK, INFY",
    ).upper().strip()

    st.markdown("**⚙️ Options**")
    fetch_peers   = st.checkbox("Fetch peer comparison", value=True)
    run_dcf_opt   = st.checkbox("Run DCF valuation",     value=True)
    run_mc_opt    = st.checkbox("Run Monte Carlo (10K)", value=True)
    n_peer_limit  = st.slider("Max peers", 1, 5, 3)

    generate_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)

    st.divider()
    with st.expander("🔧 DCF Assumptions"):
        base_growth = st.slider("Base Revenue Growth %", 3, 35, 12) / 100
        bear_growth = st.slider("Bear Revenue Growth %", 0, 15, 5)  / 100
        bull_growth = st.slider("Bull Revenue Growth %", 10, 40, 20)/ 100
        base_wacc   = st.slider("WACC %", 6, 18, 11)     / 100
        base_tgr    = st.slider("Terminal Growth %", 1, 6, 4) / 100
        beta_input  = st.slider("Beta", 0.3, 2.5, 1.0, step=0.1)
        tax_rate    = st.slider("Tax Rate %", 15, 35, 25) / 100

    dcf_assumptions = {
        "base_growth": base_growth, "bear_growth": bear_growth, "bull_growth": bull_growth,
        "base_wacc": base_wacc, "base_tgr": base_tgr,
        "bear_tgr": max(base_tgr - 0.01, 0.02), "bull_tgr": min(base_tgr + 0.01, 0.06),
        "beta": beta_input, "tax_rate": tax_rate,
        "capex_pct": 0.07, "wc_pct": 0.015, "risk_free_rate": 0.07,
        "erp": 0.055, "cost_of_debt": 0.09,
    }

    st.divider()
    st.markdown('<div style="font-size:11px; color:#8B9AB5; text-align:center;">Data: Screener.in<br>Not investment advice</div>', unsafe_allow_html=True)


# ── LANDING PAGE ──
if not generate_btn or not ticker_input:
    st.markdown(
        '<div style="text-align:center; padding:60px 0 40px;">'
        '<div style="font-size:54px; margin-bottom:12px;">📊</div>'
        '<div style="font-size:38px; font-weight:800; color:#E2E8F0; letter-spacing:-1px;">EquityLens</div>'
        '<div style="font-size:18px; color:#8B9AB5; margin-top:8px;">Professional equity research, DCF valuation &amp; '
        'Monte Carlo analysis for any NSE-listed company.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    features = [
        (c1, "📈", "5-Year Financials", "Revenue, PAT, EBITDA, EPS, CFO & FCF"),
        (c2, "🧮", "20+ Ratios",        "Profitability, liquidity, solvency & valuation"),
        (c3, "⚠️", "Red Flags",         "10 automated checks with explanations"),
        (c4, "🎯", "DCF Valuation",     "Bull/Base/Bear + Monte Carlo simulation"),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:28px; margin-bottom:8px;">{icon}</div>'
                f'<div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:12px; color:#8B9AB5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#8B9AB5; font-size:14px;">← Enter a ticker in the sidebar and click <strong style="color:#4A90D9;">Generate Report</strong></div>', unsafe_allow_html=True)
    st.markdown("**Popular tickers:**")
    cols = st.columns(8)
    for col, ex in zip(cols, ["RELIANCE","HDFCBANK","INFY","TCS","WIPRO","ICICIBANK","KOTAKBANK","AXISBANK"]):
        with col:
            st.code(ex, language=None)
    st.stop()


# ── REPORT GENERATION ──
ticker = ticker_input
prog = st.progress(0, text="Starting...")
stat = st.empty()

def upd(pct, msg):
    prog.progress(pct, text=msg)
    stat.markdown(f'<div style="color:#8B9AB5; font-size:13px;">⟳ {msg}</div>', unsafe_allow_html=True)

try:
    upd(10, f"Fetching {ticker} from Screener.in...")
    data = fetch_screener_data(ticker)

    upd(25, "Calculating ratios...")
    r = calculate_ratios(data)

    upd(35, "Red flag analysis...")
    flags     = detect_red_flags(r)
    flag_summ = flags_summary(flags)

    peer_data_list, peer_df = [], None
    if fetch_peers and data.get("peers"):
        upd(45, f"Fetching peers: {', '.join(data['peers'][:n_peer_limit])}...")
        peer_data_list = fetch_peer_data(data["peers"][:n_peer_limit], delay=1.0)
        peer_df = build_peer_comparison(ticker, data, r, peer_data_list)

    dcf_result = None
    if run_dcf_opt:
        upd(60, "Running DCF valuation...")
        dcf_result = run_three_scenarios(data, r, dcf_assumptions)

    mc_result = None
    if run_mc_opt and run_dcf_opt:
        upd(72, "Monte Carlo simulation (10,000 runs)...")
        mc_result = run_monte_carlo(data, r, dcf_assumptions)

    sens_df = None
    if run_dcf_opt:
        upd(82, "Sensitivity analysis...")
        sens_df = run_sensitivity(data, r, dcf_assumptions)

    upd(90, "Generating PDF & Excel...")
    pdf_bytes   = generate_pdf(ticker, data, r, flags, peer_df, dcf_result)
    excel_bytes = export_excel(ticker, data, r, flags, peer_df, dcf_result)

    upd(100, "Done!")
    time.sleep(0.3)
    prog.empty(); stat.empty()

except Exception as e:
    prog.empty(); stat.empty()
    st.error(f"❌ Error for **{ticker}**: {e}")
    st.info("Check the ticker is valid on Screener.in (e.g. RELIANCE, HDFCBANK, INFY)")
    st.stop()


# ── COMPANY HEADER ──
company = data.get("company_name", ticker)
price   = data.get("current_price")
mktcap  = data.get("market_cap")
high52  = data.get("high_52w")
low52   = data.get("low_52w")

price_str = f"₹{price:,.2f}" if price else "—"
w52_str   = f"52W: {fmt(low52, prefix='₹', dec=0)} – {fmt(high52, prefix='₹', dec=0)}"
sector_str = f" &nbsp;•&nbsp; {data.get('sector','')}" if data.get('sector') else ""

st.markdown(
    f'<div class="card" style="display:flex; justify-content:space-between; align-items:center; padding:24px 28px;">'
    f'<div>'
    f'<div style="font-size:26px; font-weight:800; color:#E2E8F0;">{company}</div>'
    f'<div style="font-size:14px; color:#8B9AB5; margin-top:4px;">NSE: {ticker}{sector_str}</div>'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div style="font-size:32px; font-weight:800; color:#4A90D9;">{price_str}</div>'
    f'<div style="font-size:12px; color:#8B9AB5; margin-top:2px;">{w52_str}</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── METRIC STRIP ──
m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
for col, label, val in [
    (m1, "Mkt Cap",    fmt(mktcap, suffix=" Cr", prefix="₹", dec=0)),
    (m2, "P/E",        fmt(data.get("pe_ratio"), "x")),
    (m3, "P/B",        fmt(r.get("pb_ratio"), "x", dec=2)),
    (m4, "EV/EBITDA",  fmt(r.get("ev_ebitda"), "x")),
    (m5, "Div Yield",  fmt(data.get("dividend_yield"), "%")),
    (m6, "ROCE",       fmt(last_valid(r.get("roce",[])), "%")),
    (m7, "ROE",        fmt(last_valid(r.get("roe",[])), "%")),
]:
    with col:
        st.metric(label, val)

# ── DOWNLOAD BUTTONS ──
st.markdown("<br>", unsafe_allow_html=True)
dl1, dl2, dl3 = st.columns([2, 2, 3])
with dl1:
    st.download_button("📄 Download PDF Report", pdf_bytes,
                       f"{ticker}_EquityResearch_Report.pdf", "application/pdf",
                       use_container_width=True)
with dl2:
    st.download_button("📊 Download Excel Data", excel_bytes,
                       f"{ticker}_EquityResearch_Data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
with dl3:
    if dcf_result:
        signal   = dcf_result.get("signal", "N/A")
        base_iv  = dcf_result.get("base_iv")
        upside   = dcf_result.get("upside_pct")
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:16px; padding:8px 0;">'
            f'<span class="signal-{signal.lower()}">{signal}</span>'
            f'<div>'
            f'<div style="color:#E2E8F0; font-size:15px; font-weight:600;">Base IV: {fmt(base_iv, prefix="₹", dec=0)}</div>'
            f'<div style="color:#8B9AB5; font-size:12px;">Upside: {fmt(upside, "%")} vs CMP {fmt(price, prefix="₹", dec=0)}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── TABS ──
years = r.get("years", [])

tab_fin, tab_rat, tab_cf, tab_flags, tab_dupont, tab_peers, tab_dcf, tab_mc = st.tabs([
    "📈 Financials", "🧮 Ratios", "📊 Cash Flow", "⚠️ Red Flags",
    "🔄 DuPont", "🏢 Peers", "🎯 DCF Valuation", "🎲 Monte Carlo",
])


# ══════════════════════════════════════════
# TAB 1 — FINANCIALS
# ══════════════════════════════════════════
with tab_fin:
    sec("5-Year Financial Summary (₹ Cr)")

    fin_rows = {
        "Revenue":  r.get("revenue", []),
        "EBITDA":   r.get("ebitda",  []),
        "PAT":      r.get("pat",     []),
        "EPS (₹)":  r.get("eps",     []),
    }
    try:
        fin_df = pd.DataFrame(fin_rows, index=years).T
        fin_df = fin_df.applymap(lambda v: f"{float(v):,.1f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(fin_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build financial table: {e}")

    plotly(chart_financial_trend(r, ticker))

    sec("Growth Rates (CAGR)")
    gc = st.columns(6)
    for col, label, val in [
        (gc[0], "Revenue 3Y CAGR", r.get("revenue_cagr_3y")),
        (gc[1], "Revenue 5Y CAGR", r.get("revenue_cagr_5y")),
        (gc[2], "PAT 3Y CAGR",     r.get("pat_cagr_3y")),
        (gc[3], "PAT 5Y CAGR",     r.get("pat_cagr_5y")),
        (gc[4], "EPS 3Y CAGR",     r.get("eps_cagr_3y")),
        (gc[5], "EPS 5Y CAGR",     r.get("eps_cagr_5y")),
    ]:
        with col:
            st.metric(label, fmt(val, "%"))

    plotly(chart_balance_sheet(r, ticker))

    with st.expander("📋 Raw P&L Statement"):
        pl_df = data.get("pl", pd.DataFrame())

        if not pl_df.empty:
            st.dataframe(pl_df, use_container_width=True)
        else:
            st.info("Not available.")
    with st.expander("📋 Raw Balance Sheet"):
        bs_df = data.get("bs", pd.DataFrame())

        if not bs_df.empty:
            st.dataframe(bs_df, use_container_width=True)
        else:
            st.info("Not available.")


# ══════════════════════════════════════════
# TAB 2 — RATIOS
# ══════════════════════════════════════════
with tab_rat:
    sec("Key Ratio Snapshot (Latest Year)")

    rc = st.columns(6)
    for col, label, key in [
        (rc[0], "Operating Margin", "operating_margin"),
        (rc[1], "Net Margin",       "net_margin"),
        (rc[2], "ROE",              "roe"),
        (rc[3], "ROCE",             "roce"),
        (rc[4], "Current Ratio",    "current_ratio"),
        (rc[5], "Debt/Equity",      "debt_equity"),
    ]:
        with col:
            suffix = "%" if "margin" in key.lower() or key in ("roe","roce") else "x"
            st.metric(label, fmt(last_valid(r.get(key,[])), suffix))

    st.markdown("<br>", unsafe_allow_html=True)
    sec("All Ratios — 5-Year Trend")

    ratio_series = {
        "Operating Margin %":  r.get("operating_margin", []),
        "Net Margin %":        r.get("net_margin", []),
        "ROE %":               r.get("roe", []),
        "ROA %":               r.get("roa", []),
        "ROCE %":              r.get("roce", []),
        "Current Ratio":       r.get("current_ratio", []),
        "Quick Ratio":         r.get("quick_ratio", []),
        "Debt/Equity":         r.get("debt_equity", []),
        "Interest Coverage":   r.get("interest_cover", []),
        "Asset Turnover":      r.get("asset_turnover", []),
        "Receivable Days":     r.get("receivable_days", []),
        "Inventory Days":      r.get("inventory_days", []),
        "Payable Days":        r.get("payable_days", []),
        "Cash Conversion":     r.get("cash_conversion", []),
    }
    try:
        ratio_df = pd.DataFrame(ratio_series, index=years).T
        ratio_df = ratio_df.applymap(lambda v: f"{float(v):,.2f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(ratio_df, use_container_width=True, height=420)
    except Exception as e:
        st.warning(f"Could not build ratio table: {e}")

    plotly(chart_margins(r, ticker))


# ══════════════════════════════════════════
# TAB 3 — CASH FLOW
# ══════════════════════════════════════════
with tab_cf:
    sec("Cash Flow Analysis")

    cfc = st.columns(4)
    for col, label, key in [
        (cfc[0], "Latest CFO", "cfo"),
        (cfc[1], "Latest CFI", "cfi"),
        (cfc[2], "Latest CFF", "cff"),
        (cfc[3], "Latest FCF", "fcf"),
    ]:
        with col:
            st.metric(label, fmt(last_valid(r.get(key,[])), suffix=" Cr", prefix="₹", dec=0))

    plotly(chart_cashflow_trend(r, ticker))

    sec("Cash Flow Summary")
    try:
        cf_rows = {
            "CFO (₹ Cr)": r.get("cfo", []),
            "CFI (₹ Cr)": r.get("cfi", []),
            "CFF (₹ Cr)": r.get("cff", []),
            "FCF (₹ Cr)": r.get("fcf", []),
            "PAT (₹ Cr)": r.get("pat", []),
        }
        cfo_list = r.get("cfo", [])
        pat_list = r.get("pat", [])
        cfo_pat = []
        for c, p in zip(cfo_list, pat_list):
            try:
                fc, fp = float(c), float(p)
                cfo_pat.append(f"{fc/fp:.2f}x" if fp != 0 and not np.isnan(fc) and not np.isnan(fp) else "—")
            except:
                cfo_pat.append("—")
        cf_rows["CFO / PAT"] = cfo_pat
        cf_df = pd.DataFrame(cf_rows, index=years).T
        cf_df = cf_df.applymap(lambda v: v if isinstance(v, str) else (f"{float(v):,.0f}" if v is not None and not np.isnan(float(v)) else "—"))
        st.dataframe(cf_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build CF table: {e}")

    with st.expander("📋 Raw Cash Flow Statement"):
        raw_cf = data.get("cf", pd.DataFrame())

        # FIX
        if not raw_cf.empty:
            st.dataframe(raw_cf, use_container_width=True)
        else:
            st.info("Not available.")


# ══════════════════════════════════════════
# TAB 4 — RED FLAGS
# ══════════════════════════════════════════
with tab_flags:
    sec("Red Flag Analysis")

    rfc = st.columns(3)
    for col, label, val, color in [
        (rfc[0], "High Severity",   flag_summ["high"],   "#E74C3C"),
        (rfc[1], "Medium Severity", flag_summ["medium"], "#F39C12"),
        (rfc[2], "Checks Passed",   flag_summ["ok"],     "#27AE60"),
    ]:
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center; border-color:{color};">'
                f'<div style="font-size:36px; font-weight:800; color:{color};">{val}</div>'
                f'<div style="color:#8B9AB5; font-size:13px;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    for flag in flags:
        if flag["triggered"]:
            cls  = "flag-high" if flag["severity"] == "high" else "flag-medium"
            icon = "🔴" if flag["severity"] == "high" else "🟡"
        else:
            cls, icon = "flag-ok", "✅"

        st.markdown(
            f'<div class="{cls}">'
            f'<div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:4px;">'
            f'{icon} {flag["name"]} — {flag["check"]}</div>'
            f'<div style="font-size:12px; color:#B0BCC8; margin-bottom:4px;">{flag["detail"]}</div>'
            f'<div style="font-size:11px; color:#8B9AB5;">{flag["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════
# TAB 5 — DUPONT
# ══════════════════════════════════════════
with tab_dupont:
    sec("DuPont ROE Decomposition (3-Step)")

    st.markdown(
        '<div class="card" style="font-size:13px; color:#B0BCC8;">'
        '<strong style="color:#E2E8F0;">ROE = Net Profit Margin × Asset Turnover × Equity Multiplier</strong><br>'
        'This breaks down <em>why</em> ROE changed — was it profitability, efficiency, or leverage?'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        du_table = {
            "Net Margin %":          r.get("dupont_net_margin", []),
            "Asset Turnover (×)":    r.get("dupont_asset_turnover", []),
            "Equity Multiplier (×)": r.get("dupont_equity_mult", []),
            "DuPont ROE %":          r.get("dupont_roe", []),
        }
        du_df = pd.DataFrame(du_table, index=years).T
        du_df = du_df.applymap(lambda v: f"{float(v):,.2f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(du_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build DuPont table: {e}")

    plotly(chart_dupont(r, ticker))

    nm = last_valid(r.get("dupont_net_margin", []))
    at = last_valid(r.get("dupont_asset_turnover", []))
    em = last_valid(r.get("dupont_equity_mult", []))
    if not any(np.isnan(v) for v in [nm, at, em]):
        st.markdown(
            '<div class="card">'
            '<div style="font-size:14px; font-weight:700; color:#E2E8F0; margin-bottom:8px;">Latest Year Interpretation</div>'
            f'<div style="font-size:13px; color:#B0BCC8; line-height:1.9;">'
            f'• <strong style="color:#E2E8F0;">Net Margin {nm:.1f}%</strong> — keeps ₹{nm:.1f} of every ₹100 revenue.<br>'
            f'• <strong style="color:#E2E8F0;">Asset Turnover {at:.2f}×</strong> — generates ₹{at:.2f} revenue per ₹1 of assets.<br>'
            f'• <strong style="color:#E2E8F0;">Equity Multiplier {em:.2f}×</strong> — assets are {em:.2f}× larger than equity (leverage).'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════
# TAB 6 — PEERS
# ══════════════════════════════════════════
with tab_peers:
    sec("Peer Comparison")

    if peer_df is not None and len(peer_df) > 1:
        display_cols = [c for c in ["Company","Ticker","Mkt Cap (Cr)","CMP (₹)","P/E","P/B",
                                     "Net Margin %","ROE %","ROCE %","D/E","Current Ratio","Revenue CAGR 3Y %"]
                        if c in peer_df.columns]

        def highlight_rows(row):
            if row.get("_is_target") is True:
                return ["background-color: #1B3A5C; font-weight:bold"] * len(row)
            if row.get("_is_target") is None:
                return ["background-color: #1A3326; font-style:italic"] * len(row)
            return [""] * len(row)

        disp_df = peer_df[display_cols + ["_is_target"]].copy()
        st.dataframe(disp_df.style.apply(highlight_rows, axis=1), use_container_width=True, height=280)
        st.markdown('<small style="color:#8B9AB5;">★ Blue row = target company &nbsp;|&nbsp; Green row = Industry Median</small>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        plotly(chart_peer_radar(peer_df, ticker))
    elif not fetch_peers:
        st.info("Enable 'Fetch peer comparison' in the sidebar.")
    else:
        st.warning(f"No peers found for {ticker}, or peer data could not be fetched.")


# ══════════════════════════════════════════
# TAB 7 — DCF VALUATION
# ══════════════════════════════════════════
with tab_dcf:
    if not run_dcf_opt or dcf_result is None:
        st.info("Enable 'Run DCF valuation' in the sidebar.")
    else:
        sec("DCF Valuation — Bull / Base / Bear Scenarios")

        signal    = dcf_result.get("signal", "N/A")
        upside    = dcf_result.get("upside_pct")
        base_iv   = dcf_result.get("base_iv")
        bear_iv   = dcf_result["scenarios"]["Bear"]["intrinsic_per_share"]
        bull_iv   = dcf_result["scenarios"]["Bull"]["intrinsic_per_share"]
        wacc_used = dcf_result["wacc_result"]["wacc"] * 100

        dh1, dh2, dh3, dh4 = st.columns(4)
        with dh1:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:13px; color:#8B9AB5; margin-bottom:8px;">Signal</div>'
                f'<span class="signal-{signal.lower()}">{signal}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with dh2:
            st.metric("Base Case IV", fmt(base_iv, prefix="₹", dec=0))
        with dh3:
            st.metric("Upside / (Downside)", fmt(upside, "%"))
        with dh4:
            st.metric("WACC Used", fmt(wacc_used, "%"))

        wr = dcf_result["wacc_result"]
        with st.expander("🔍 WACC Breakdown"):
            wc = st.columns(5)
            for col, label, val in [
                (wc[0], "WACC",           wr["wacc"]*100),
                (wc[1], "Cost of Equity", wr["cost_of_equity"]*100),
                (wc[2], "After-tax CoD",  wr["cost_of_debt_after_tax"]*100),
                (wc[3], "Equity Weight",  wr["equity_weight"]*100),
                (wc[4], "Debt Weight",    wr["debt_weight"]*100),
            ]:
                col.metric(label, fmt(val, "%"))

        plotly(chart_scenario_comparison(dcf_result, data.get("current_price", 0)))

        sec("Projected Cash Flows (₹ Cr)")
        sc1, sc2, sc3 = st.columns(3)
        for col, scenario, icon in [(sc1,"Bear","🔴"), (sc2,"Base","🟡"), (sc3,"Bull","🟢")]:
            with col:
                s = dcf_result["scenarios"][scenario]
                st.markdown(f"**{icon} {scenario} Case**")
                fcff_df = s.get("fcff_df")
                if fcff_df is not None:
                    st.dataframe(fcff_df[["Year","Revenue","EBITDA","FCFF"]].set_index("Year"),
                                 use_container_width=True)
                st.metric("Intrinsic Value / Share", fmt(s["intrinsic_per_share"], prefix="₹", dec=0))
                st.metric("Enterprise Value", fmt(s["enterprise_value"], suffix=" Cr", prefix="₹", dec=0))

        plotly(chart_dcf_waterfall(dcf_result, "Base"))

        if sens_df is not None and not sens_df.empty:
            sec("Sensitivity Analysis — Tornado Chart")
            plotly(chart_tornado(sens_df, base_iv or 0))
            st.dataframe(sens_df.set_index("Variable")[["Low IV","Base IV","High IV","Impact"]],
                         use_container_width=True)


# ══════════════════════════════════════════
# TAB 8 — MONTE CARLO
# ══════════════════════════════════════════
with tab_mc:
    if not run_mc_opt or mc_result is None:
        st.info("Enable 'Run Monte Carlo' in the sidebar.")
    elif "error" in mc_result:
        st.error(mc_result["error"])
    else:
        sec("Monte Carlo Simulation — 10,000 Iterations")

        st.markdown(
            '<div class="card" style="font-size:13px; color:#B0BCC8;">'
            'Each run randomizes revenue growth, WACC, terminal growth rate, and capex % '
            'using a normal distribution around your base assumptions.'
            '</div>',
            unsafe_allow_html=True,
        )

        mcc = st.columns(6)
        for col, label, val in [
            (mcc[0], "Mean IV",    mc_result["mean"]),
            (mcc[1], "Median IV",  mc_result["median"]),
            (mcc[2], "P10 (Bear)", mc_result["p10"]),
            (mcc[3], "P25",        mc_result["p25"]),
            (mcc[4], "P75",        mc_result["p90"]),
            (mcc[5], "P90 (Bull)", mc_result["p90"]),
        ]:
            with col:
                st.metric(label, fmt(val, prefix="₹", dec=0))

        prob = mc_result["prob_undervalued"]
        cmp  = mc_result["current_price"]
        st.markdown(
            f'<div class="card">'
            f'<span style="color:#4A90D9; font-size:20px; font-weight:700;">{prob}%</span>'
            f'<span style="color:#B0BCC8; font-size:14px;"> of simulations suggest the stock is '
            f'undervalued vs CMP {fmt(cmp, prefix="₹", dec=0)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        plotly(chart_monte_carlo(mc_result, ticker))

        dist_df = pd.DataFrame({
            "Percentile": ["P10","P25","Median","Mean","P75","P90"],
            "Intrinsic Value (₹)": [mc_result["p10"],mc_result["p25"],mc_result["median"],
                                    mc_result["mean"],mc_result["p75"],mc_result["p90"]],
            "vs CMP": [f"{(v-cmp)/cmp*100:.1f}%" if cmp else "—"
                       for v in [mc_result["p10"],mc_result["p25"],mc_result["median"],
                                 mc_result["mean"],mc_result["p75"],mc_result["p90"]]],
        })
        st.dataframe(dist_df.set_index("Percentile"), use_container_width=True)


# ── FOOTER ──
st.divider()
st.markdown(
    '<div style="text-align:center; color:#8B9AB5; font-size:12px; padding:10px 0;">'
    'EquityLens — Automated equity research for NSE-listed companies.<br>'
    'Data: <a href="https://www.screener.in" target="_blank" style="color:#4A90D9;">Screener.in</a> &nbsp;|&nbsp; '
    'Not investment advice. All figures ₹ Crore unless stated.'
    '</div>',
    unsafe_allow_html=True,
)
