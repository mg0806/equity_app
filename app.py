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
.flag-high   { background: rgba(231,76,60,0.12);  border-left: 4px solid #E74C3C; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.flag-medium { background: rgba(243,156,18,0.12); border-left: 4px solid #F39C12; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.flag-ok     { background: rgba(39,174,96,0.12);  border-left: 4px solid #27AE60; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
.signal-buy       { background:#27AE60; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-strong\ buy { background:#1a7a3c; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-sell  { background:#E74C3C; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-hold  { background:#F39C12; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.signal-na    { background:#95A5A6; color:white; padding:6px 18px; border-radius:20px; font-weight:700; font-size:18px; display:inline-block; }
.sec-title    { font-size:18px; font-weight:700; color:#E2E8F0; margin-bottom:12px; padding-bottom:6px; border-bottom:2px solid #1B4F8A; }
.card         { background:#111827; border:1px solid #1E2A45; border-radius:10px; padding:20px; margin-bottom:16px; }
.chart-tip    { text-align:right; font-size:11px; color:#8B9AB5; margin-top:-6px; margin-bottom:10px; }
.factor-bar-bg { background:#1E2A45; border-radius:6px; height:10px; overflow:hidden; margin-top:4px; }
.sb-row { display:flex; justify-content:space-between; font-size:11px; color:#B0BCC8; padding:4px 0; border-bottom:1px solid #1E2A45; }
.sb-val { color:#7EB8F7; font-weight:600; }
.sb-source { font-size:9px; color:#4A6A8A; font-style:italic; }
.quality-bar-bg { background:#1E2A45; border-radius:8px; height:14px; overflow:hidden; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

from scraper import fetch_screener_data, fetch_peer_data
from ratios import calculate_ratios, build_peer_comparison, last_valid
from red_flags import detect_red_flags, flags_summary
from dcf import (run_three_scenarios, run_monte_carlo, run_sensitivity,
                 derive_assumptions_from_history)
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
    except Exception:
        return na

def chart_hint():
    st.markdown(
        '<div class="chart-tip">💡 <em>Click ⛶ for fullscreen &nbsp;|&nbsp; '
        'Pan &amp; Zoom to explore</em></div>',
        unsafe_allow_html=True,
    )

def plotly(fig):
    st.plotly_chart(fig, use_container_width=True)
    chart_hint()

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def score_bar_html(score: float, label: str, weight_pct: int) -> str:
    pct   = int((score + 1) / 2 * 100)
    color = "#27AE60" if score > 0.1 else ("#E74C3C" if score < -0.1 else "#F39C12")
    return (
        f'<div style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#B0BCC8;">'
        f'<span>{label} <span style="color:#8B9AB5;font-size:10px;">(weight {weight_pct}%)</span></span>'
        f'<span style="color:{color};font-weight:700;">{score:+.2f}</span></div>'
        f'<div class="factor-bar-bg">'
        f'<div style="height:10px;width:{pct}%;background:{color};border-radius:6px;"></div>'
        f'</div></div>'
    )

def quality_bar_html(score: float, max_score: float, label: str) -> str:
    """Render a filled quality score bar (0 to max_score scale)."""
    pct   = int(score / max_score * 100) if max_score > 0 else 0
    color = "#27AE60" if pct >= 66 else ("#F39C12" if pct >= 33 else "#E74C3C")
    return (
        f'<div style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#B0BCC8;">'
        f'<span>{label}</span>'
        f'<span style="color:{color};font-weight:700;">{score:.1f} / {max_score:.0f}</span></div>'
        f'<div class="quality-bar-bg">'
        f'<div style="height:14px;width:{pct}%;background:{color};border-radius:8px;"></div>'
        f'</div></div>'
    )

BIZ_TYPE_LABELS = {
    "high-margin-stable": ("🏆 High-Margin Stable",  "#27AE60"),
    "stable":             ("✅ Stable",               "#4A90D9"),
    "cyclical":           ("🔄 Cyclical",             "#F39C12"),
    "commodity":          ("🪨 Commodity",            "#E74C3C"),
}
BIZ_TYPE_TOOLTIPS = {
    "high-margin-stable": "High OPM (≥18%), low revenue volatility, consistent FCF — e.g. IT, pharma, FMCG.",
    "stable":             "Moderate-to-good margins, predictable demand — e.g. consumer, banks.",
    "cyclical":           "Revenue highly sensitive to economic cycles — e.g. auto, infra, metals.",
    "commodity":          "Thin margins, price-taker dynamics — e.g. steel, cement, oil & gas.",
}


# ── Session state: flat keys for assumptions + report cache ──
_DEFAULT_AA = {
    "aa_base_growth":        0.12,
    "aa_bear_growth":        0.06,
    "aa_bull_growth":        0.20,
    "aa_base_ebitda_margin": 15.0,
    "aa_base_margin_delta":  0.002,
    "aa_bear_margin_delta":  -0.003,
    "aa_bull_margin_delta":  0.007,
    "aa_base_wacc":          0.11,
    "aa_base_tgr":           0.04,
    "aa_bear_tgr":           0.03,
    "aa_bull_tgr":           0.05,
    "aa_beta":               1.0,
    "aa_tax_rate":           0.25,
    "aa_capex_pct":          0.07,
    "aa_wc_pct":             0.015,
    "aa_risk_free_rate":     0.07,
    "aa_erp":                0.055,
    "aa_cost_of_debt":       0.09,
    "aa_de_ratio":           0.0,
    "aa_raw_growth_pct":     12.0,
    "aa_ticker":             "",
    "aa_is_derived":         False,
    "aa_business_type":      "stable",
    "aa_quality_score":      0.0,
    # signal cache
    "sig_signal":            "",
    "sig_conviction":        "",
    "sig_composite_score":   0.0,
    "sig_mos_required":      20.0,
    "sig_quality_score":     0.0,
    "sig_business_type":     "stable",
    # cagr cache for sidebar display
    "cagr3":                 None,
    "cagr5":                 None,
    # last ticker
    "last_ticker":           "",
    # full report cache
    "report_data":           None,
}
for k, v in _DEFAULT_AA.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _read_aa() -> dict:
    """Read assumptions from session state into a plain dict."""
    prefix = "aa_"
    keys = [k[len(prefix):] for k in _DEFAULT_AA if k.startswith(prefix)]
    return {k: st.session_state[f"aa_{k}"] for k in keys}


def _write_aa(d: dict):
    """Write a derive_assumptions_from_history() result into session state."""
    mapping = {
        "base_growth":        "aa_base_growth",
        "bear_growth":        "aa_bear_growth",
        "bull_growth":        "aa_bull_growth",
        "base_ebitda_margin": "aa_base_ebitda_margin",
        "base_margin_delta":  "aa_base_margin_delta",
        "bear_margin_delta":  "aa_bear_margin_delta",
        "bull_margin_delta":  "aa_bull_margin_delta",
        "base_wacc":          "aa_base_wacc",
        "base_tgr":           "aa_base_tgr",
        "bear_tgr":           "aa_bear_tgr",
        "bull_tgr":           "aa_bull_tgr",
        "beta":               "aa_beta",
        "tax_rate":           "aa_tax_rate",
        "capex_pct":          "aa_capex_pct",
        "wc_pct":             "aa_wc_pct",
        "risk_free_rate":     "aa_risk_free_rate",
        "erp":                "aa_erp",
        "cost_of_debt":       "aa_cost_of_debt",
        "de_ratio":           "aa_de_ratio",
        "raw_growth_pct":     "aa_raw_growth_pct",
        "business_type":      "aa_business_type",
        "quality_score":      "aa_quality_score",
    }
    for src, dst in mapping.items():
        if src in d:
            st.session_state[dst] = d[src]


# ═══════════════════════════════════════════════════
# SIDEBAR  — reads exclusively from session state
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:20px 0 10px;">'
        '<div style="font-size:36px;">📊</div>'
        '<div style="font-size:20px;font-weight:700;color:#E2E8F0;">EquityLens</div>'
        '<div style="font-size:12px;color:#8B9AB5;margin-top:4px;">NSE Research Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    ticker_input = st.text_input(
        "🔍 NSE Ticker Symbol",
        value=st.session_state["last_ticker"],
        placeholder="e.g. RELIANCE, HDFCBANK, INFY",
    ).upper().strip()

    st.markdown("**⚙️ Options**")
    fetch_peers  = st.checkbox("Fetch peer comparison", value=True)
    run_dcf_opt  = st.checkbox("Run DCF valuation",     value=True)
    run_mc_opt   = st.checkbox("Run Monte Carlo (10K)", value=True)
    n_peer_limit = st.slider("Max peers", 1, 5, 3)

    generate_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)

    st.divider()

    # ── Business type + Quality badge ──
    aa           = _read_aa()
    is_derived   = st.session_state["aa_is_derived"]
    derived_for  = st.session_state["aa_ticker"]

    if is_derived and derived_for:
        biz_type   = st.session_state.get("aa_business_type", "stable")
        qs         = st.session_state.get("aa_quality_score",  0.0)
        biz_label, biz_color = BIZ_TYPE_LABELS.get(biz_type, ("Stable", "#4A90D9"))
        qs_color = "#27AE60" if qs >= 70 else ("#F39C12" if qs >= 45 else "#E74C3C")

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
            f'<span style="font-size:14px;font-weight:700;color:#E2E8F0;">🤖 DCF Assumptions</span>'
            f'<span style="font-size:10px;background:#1B3A5C;color:#7EB8F7;'
            f'border-radius:10px;padding:2px 8px;">Auto · {derived_for}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Business type + quality row
        st.markdown(
            f'<div style="display:flex;gap:6px;margin-bottom:8px;">'
            f'<span style="font-size:11px;background:{biz_color}22;color:{biz_color};'
            f'border:1px solid {biz_color}44;border-radius:10px;padding:2px 9px;">{biz_label}</span>'
            f'<span style="font-size:11px;background:{qs_color}22;color:{qs_color};'
            f'border:1px solid {qs_color}44;border-radius:10px;padding:2px 9px;">'
            f'Quality {qs:.0f}/100</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("**⚙️ DCF Assumptions**")
        st.markdown(
            '<div style="font-size:10px;color:#8B9AB5;margin-bottom:6px;">'
            'Defaults shown — generate a report to auto-derive from real data.</div>',
            unsafe_allow_html=True,
        )

    allow_override = st.toggle("✏️ Manual override", value=False, key="override_toggle")

    if allow_override:
        ov_base_growth = st.slider("Base Revenue Growth %", 3, 35,
                                   int(round(aa["base_growth"] * 100))) / 100
        ov_bear_growth = st.slider("Bear Revenue Growth %", 0, 15,
                                   max(0, int(round(aa["bear_growth"] * 100)))) / 100
        ov_bull_growth = st.slider("Bull Revenue Growth %", 10, 40,
                                   min(40, max(10, int(round(aa["bull_growth"] * 100))))) / 100
        ov_ebitda_m    = st.slider("Base EBITDA Margin %", 3, 60,
                                   int(round(aa["base_ebitda_margin"])))
        ov_wacc        = st.slider("WACC %", 6, 18,
                                   int(round(aa["base_wacc"] * 100))) / 100
        ov_tgr         = st.slider("Terminal Growth %", 1, 6,
                                   int(round(aa["base_tgr"] * 100))) / 100
        ov_beta        = st.slider("Beta", 0.3, 2.5,
                                   float(round(aa["beta"], 1)), step=0.1)
        ov_tax         = st.slider("Tax Rate %", 15, 35,
                                   int(round(aa["tax_rate"] * 100))) / 100

        dcf_assumptions = {
            **aa,
            "base_growth":        ov_base_growth,
            "bear_growth":        ov_bear_growth,
            "bull_growth":        ov_bull_growth,
            "base_ebitda_margin": ov_ebitda_m,
            "base_wacc":          ov_wacc,
            "base_tgr":           ov_tgr,
            "bear_tgr":           max(ov_tgr - 0.01, 0.02),
            "bull_tgr":           min(ov_tgr + 0.01, 0.06),
            "beta":               ov_beta,
            "tax_rate":           ov_tax,
            "auto_derived":       False,
        }

    else:
        # ── Read-only table with source annotations ──
        def _sb_row(label, val, source=""):
            src_html = f' <span class="sb-source">({source})</span>' if source else ""
            return (
                f'<div class="sb-row">'
                f'<span>{label}{src_html}</span>'
                f'<span class="sb-val">{val}</span>'
                f'</div>'
            )

        c3   = fmt(st.session_state["cagr3"], "%") if st.session_state["cagr3"] is not None else "—"
        c5   = fmt(st.session_state["cagr5"], "%") if st.session_state["cagr5"] is not None else "—"
        de_r = aa["de_ratio"]

        if is_derived:
            biz_type = aa.get("business_type", "stable")
            rows_html = (
                _sb_row("Business Type",  BIZ_TYPE_LABELS.get(biz_type, ("stable",""))[0]) +
                _sb_row("Base Growth",    f"{aa['base_growth']*100:.1f}%",
                        f"3Y/5Y CAGR {c3}/{c5}") +
                _sb_row("Bear Growth",    f"{aa['bear_growth']*100:.1f}%",  "50% of base") +
                _sb_row("Bull Growth",    f"{aa['bull_growth']*100:.1f}%",  "160% of base") +
                _sb_row("EBITDA Margin",  f"{aa['base_ebitda_margin']:.1f}%","5Y median") +
                _sb_row("WACC",           f"{aa['base_wacc']*100:.1f}%",
                        f"β={aa['beta']:.2f}, D/E={de_r:.2f}x") +
                _sb_row("Terminal Growth",f"{aa['base_tgr']*100:.1f}%", "GDP/2, type-adj") +
                _sb_row("Capex % Rev",    f"{aa['capex_pct']*100:.1f}%",   "5Y median") +
                _sb_row("Tax Rate",       f"{aa['tax_rate']*100:.0f}%",    "effective rate") +
                _sb_row("Cost of Debt",   f"{aa['cost_of_debt']*100:.1f}%","interest/debt")
            )
        else:
            rows_html = (
                _sb_row("Base Growth",    f"{aa['base_growth']*100:.1f}%") +
                _sb_row("Bear Growth",    f"{aa['bear_growth']*100:.1f}%") +
                _sb_row("Bull Growth",    f"{aa['bull_growth']*100:.1f}%") +
                _sb_row("EBITDA Margin",  f"{aa['base_ebitda_margin']:.1f}%") +
                _sb_row("WACC",           f"{aa['base_wacc']*100:.1f}%") +
                _sb_row("Terminal Growth",f"{aa['base_tgr']*100:.1f}%") +
                _sb_row("Capex % Rev",    f"{aa['capex_pct']*100:.1f}%") +
                _sb_row("Tax Rate",       f"{aa['tax_rate']*100:.0f}%")
            )

        st.markdown(
            f'<div style="background:#0A0F1E;border:1px solid #1E2A45;border-radius:8px;'
            f'padding:10px 14px;margin-bottom:6px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )

        # Mini signal badge (only when a report has been run)
        if is_derived:
            signal  = st.session_state["sig_signal"]
            conv    = st.session_state["sig_conviction"]
            score   = st.session_state["sig_composite_score"]
            mos_req = st.session_state.get("sig_mos_required", 20.0)
            if signal:
                sig_color = {"BUY":"#27AE60","STRONG BUY":"#1a7a3c","SELL":"#E74C3C","HOLD":"#F39C12"}.get(signal,"#95A5A6")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'background:#0A0F1E;border:1px solid {sig_color}44;border-radius:8px;'
                    f'padding:8px 12px;margin-top:2px;">'
                    f'<span style="background:{sig_color};color:white;border-radius:12px;'
                    f'padding:2px 10px;font-weight:700;font-size:13px;">{signal}</span>'
                    f'<div style="font-size:10px;color:#8B9AB5;line-height:1.6;">'
                    f'{conv} conviction<br>'
                    f'<span style="color:{sig_color};font-weight:600;">Score {score:+.2f}</span><br>'
                    f'<span style="color:#8B9AB5;">MOS reqd: {mos_req:.0f}%</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        dcf_assumptions = {**aa, "auto_derived": True}

    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#8B9AB5;text-align:center;">'
        'Data: Screener.in<br>Not investment advice</div>',
        unsafe_allow_html=True,
    )


# ── LANDING PAGE (no report yet) ──
if not generate_btn and st.session_state["report_data"] is None:
    st.markdown(
        '<div style="text-align:center;padding:60px 0 40px;">'
        '<div style="font-size:54px;margin-bottom:12px;">📊</div>'
        '<div style="font-size:38px;font-weight:800;color:#E2E8F0;letter-spacing:-1px;">EquityLens</div>'
        '<div style="font-size:18px;color:#8B9AB5;margin-top:8px;">Professional equity research, '
        'DCF valuation &amp; Monte Carlo analysis for any NSE-listed company.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1,"📈","5-Year Financials","Revenue, PAT, EBITDA, EPS, CFO & FCF"),
        (c2,"🧮","20+ Ratios","Profitability, liquidity, solvency & valuation"),
        (c3,"⚠️","Red Flags","10 automated checks with explanations"),
        (c4,"🎯","Smart DCF","Quality-scored, business-type-aware signals"),
    ]:
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:28px;margin-bottom:8px;">{icon}</div>'
                f'<div style="font-size:14px;font-weight:700;color:#E2E8F0;margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:12px;color:#8B9AB5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:#8B9AB5;font-size:14px;">'
        '← Enter a ticker in the sidebar and click '
        '<strong style="color:#4A90D9;">Generate Report</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Popular tickers:**")
    tcols = st.columns(8)
    for col, ex in zip(tcols, ["RELIANCE","HDFCBANK","INFY","TCS","WIPRO","ICICIBANK","KOTAKBANK","AXISBANK"]):
        with col:
            st.code(ex, language=None)
    st.stop()


# ═══════════════════════════════════════════════════
# GENERATION BLOCK — runs only when button pressed
# ═══════════════════════════════════════════════════
if generate_btn and ticker_input:
    ticker = ticker_input
    prog   = st.progress(0, text="Starting...")
    stat   = st.empty()

    def upd(pct, msg):
        prog.progress(pct, text=msg)
        stat.markdown(f'<div style="color:#8B9AB5;font-size:13px;">⟳ {msg}</div>',
                      unsafe_allow_html=True)

    try:
        upd(10, f"Fetching {ticker} from Screener.in...")
        data = fetch_screener_data(ticker)

        upd(25, "Calculating ratios...")
        r = calculate_ratios(data)

        # Fetch peers first so peer_df is available for margin benchmarking
        peer_data_list, peer_df = [], None
        if fetch_peers and data.get("peers"):
            upd(35, f"Fetching peers: {', '.join(data['peers'][:n_peer_limit])}...")
            peer_data_list = fetch_peer_data(data["peers"][:n_peer_limit], delay=1.0)
            peer_df = build_peer_comparison(ticker, data, r, peer_data_list)

        upd(45, "Deriving DCF assumptions from historical data...")
        auto_assum = derive_assumptions_from_history(data, r, peer_df)

        # Write derived assumptions into session state
        _write_aa(auto_assum)
        st.session_state["aa_ticker"]     = ticker
        st.session_state["aa_is_derived"] = True
        st.session_state["cagr3"]         = r.get("revenue_cagr_3y")
        st.session_state["cagr5"]         = r.get("revenue_cagr_5y")
        st.session_state["last_ticker"]   = ticker

        # Use override sliders if toggle is on, else use auto
        final_assumptions = (dcf_assumptions if allow_override
                             else {**auto_assum, "auto_derived": True})

        upd(55, "Red flag analysis...")
        flags     = detect_red_flags(r)
        flag_summ = flags_summary(flags)

        dcf_result = None
        if run_dcf_opt:
            upd(65, "Running DCF valuation...")
            dcf_result = run_three_scenarios(data, r, final_assumptions)
            # Cache signal for sidebar badge
            st.session_state["sig_signal"]          = dcf_result.get("signal", "")
            st.session_state["sig_conviction"]      = dcf_result.get("conviction", "")
            st.session_state["sig_composite_score"] = dcf_result.get("composite_score", 0.0)
            st.session_state["sig_mos_required"]    = dcf_result.get("mos_required", 20.0)
            st.session_state["sig_quality_score"]   = dcf_result.get("quality_score", 0.0)
            st.session_state["sig_business_type"]   = dcf_result.get("business_type", "stable")

        mc_result = None
        if run_mc_opt and run_dcf_opt:
            upd(77, "Monte Carlo simulation (10,000 runs)...")
            mc_result = run_monte_carlo(data, r, final_assumptions)

        sens_df = None
        if run_dcf_opt:
            upd(85, "Sensitivity analysis...")
            sens_df = run_sensitivity(data, r, final_assumptions)

        upd(93, "Generating PDF & Excel...")
        pdf_bytes   = generate_pdf(ticker, data, r, flags, peer_df, dcf_result)
        excel_bytes = export_excel(ticker, data, r, flags, peer_df, dcf_result)

        # Store entire report in session state
        st.session_state["report_data"] = {
            "ticker": ticker, "data": data, "r": r,
            "flags": flags, "flag_summ": flag_summ,
            "peer_df": peer_df, "dcf_result": dcf_result,
            "mc_result": mc_result, "sens_df": sens_df,
            "pdf_bytes": pdf_bytes, "excel_bytes": excel_bytes,
            "final_assumptions": final_assumptions,
            "years": r.get("years", []),
        }

        upd(100, "Done!")
        time.sleep(0.3)
        prog.empty()
        stat.empty()

        # Rerun so sidebar reads updated session state
        st.rerun()

    except Exception as e:
        prog.empty()
        stat.empty()
        st.error(f"❌ Error for **{ticker}**: {e}")
        st.info("Check the ticker is valid on Screener.in (e.g. RELIANCE, HDFCBANK, INFY)")
        st.stop()


# ═══════════════════════════════════════════════════
# DISPLAY REPORT — always from session state cache
# ═══════════════════════════════════════════════════
rd = st.session_state["report_data"]
if rd is None:
    st.stop()

ticker            = rd["ticker"]
data              = rd["data"]
r                 = rd["r"]
flags             = rd["flags"]
flag_summ         = rd["flag_summ"]
peer_df           = rd["peer_df"]
dcf_result        = rd["dcf_result"]
mc_result         = rd["mc_result"]
sens_df           = rd["sens_df"]
pdf_bytes         = rd["pdf_bytes"]
excel_bytes       = rd["excel_bytes"]
final_assumptions = rd["final_assumptions"]
years             = rd["years"]


# ── COMPANY HEADER ──
company   = data.get("company_name", ticker)
price     = data.get("current_price")
mktcap    = data.get("market_cap")
price_str = f"₹{price:,.2f}" if price else "—"
w52_str   = f"52W: {fmt(data.get('low_52w'), prefix='₹', dec=0)} – {fmt(data.get('high_52w'), prefix='₹', dec=0)}"
sector_str = f" &nbsp;•&nbsp; {data.get('sector','')}" if data.get('sector') else ""

# Quality + Business type badge for header
_biz      = final_assumptions.get("business_type", "stable")
_qs       = final_assumptions.get("quality_score", 0.0)
_biz_lbl, _biz_col = BIZ_TYPE_LABELS.get(_biz, ("Stable", "#4A90D9"))
_qs_col   = "#27AE60" if _qs >= 70 else ("#F39C12" if _qs >= 45 else "#E74C3C")
_biz_badges = (
    f'<span style="font-size:11px;background:{_biz_col}22;color:{_biz_col};'
    f'border:1px solid {_biz_col}44;border-radius:10px;padding:2px 9px;margin-right:6px;">'
    f'{_biz_lbl}</span>'
    f'<span style="font-size:11px;background:{_qs_col}22;color:{_qs_col};'
    f'border:1px solid {_qs_col}44;border-radius:10px;padding:2px 9px;">'
    f'Quality {_qs:.0f}/100</span>'
) if final_assumptions.get("auto_derived") else ""

st.markdown(
    f'<div class="card" style="display:flex;justify-content:space-between;'
    f'align-items:center;padding:24px 28px;">'
    f'<div>'
    f'<div style="font-size:26px;font-weight:800;color:#E2E8F0;">{company}</div>'
    f'<div style="font-size:14px;color:#8B9AB5;margin-top:4px;">NSE: {ticker}{sector_str}</div>'
    f'<div style="margin-top:8px;">{_biz_badges}</div>'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div style="font-size:32px;font-weight:800;color:#4A90D9;">{price_str}</div>'
    f'<div style="font-size:12px;color:#8B9AB5;margin-top:2px;">{w52_str}</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
for col, label, val in [
    (m1,"Mkt Cap",   fmt(mktcap, suffix=" Cr", prefix="₹", dec=0)),
    (m2,"P/E",       fmt(data.get("pe_ratio"), "x")),
    (m3,"P/B",       fmt(r.get("pb_ratio"), "x", dec=2)),
    (m4,"EV/EBITDA", fmt(r.get("ev_ebitda"), "x")),
    (m5,"Div Yield", fmt(data.get("dividend_yield"), "%")),
    (m6,"ROCE",      fmt(last_valid(r.get("roce",[])), "%")),
    (m7,"ROE",       fmt(last_valid(r.get("roe",[])), "%")),
]:
    with col:
        st.metric(label, val)

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
        signal     = dcf_result.get("signal","N/A")
        conviction = dcf_result.get("conviction","")
        base_iv    = dcf_result.get("base_iv")
        upside     = dcf_result.get("upside_pct")
        comp_score = dcf_result.get("composite_score", 0)
        mos_req    = dcf_result.get("mos_required", 20.0)
        auto_tag   = "🤖 Auto" if final_assumptions.get("auto_derived") else "✏️ Manual"
        sig_color  = {"BUY":"#27AE60","STRONG BUY":"#1a7a3c","SELL":"#E74C3C","HOLD":"#F39C12"}.get(signal,"#95A5A6")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;padding:8px 0;">'
            f'<span class="signal-{signal.lower()}">{signal}</span>'
            f'<div>'
            f'<div style="color:#E2E8F0;font-size:15px;font-weight:600;">'
            f'Base IV: {fmt(base_iv,prefix="₹",dec=0)} '
            f'<span style="font-size:11px;color:#8B9AB5;">({conviction} conviction)</span></div>'
            f'<div style="color:#8B9AB5;font-size:12px;">'
            f'Composite: {comp_score:+.2f} &nbsp;|&nbsp; '
            f'DCF upside: {fmt(upside,"%")} &nbsp;|&nbsp; '
            f'MOS reqd: {mos_req:.0f}% &nbsp;|&nbsp; {auto_tag}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.divider()


# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
(tab_fin, tab_rat, tab_cf, tab_flags, tab_quality,
 tab_dupont, tab_peers, tab_dcf, tab_mc) = st.tabs([
    "📈 Financials","🧮 Ratios","📊 Cash Flow","⚠️ Red Flags",
    "🏅 Business Quality","🔄 DuPont","🏢 Peers","🎯 DCF Valuation","🎲 Monte Carlo",
])

# ── TAB 1: FINANCIALS ──
with tab_fin:
    sec("5-Year Financial Summary (₹ Cr)")
    try:
        fin_df = pd.DataFrame({
            "Revenue": r.get("revenue",[]),
            "EBITDA":  r.get("ebitda",[]),
            "PAT":     r.get("pat",[]),
            "EPS (₹)": r.get("eps",[]),
        }, index=years).T
        fin_df = fin_df.applymap(
            lambda v: f"{float(v):,.1f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(fin_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build financial table: {e}")
    plotly(chart_financial_trend(r, ticker))
    sec("Growth Rates (CAGR)")
    gc = st.columns(6)
    for col, label, val in [
        (gc[0],"Revenue 3Y CAGR",r.get("revenue_cagr_3y")),
        (gc[1],"Revenue 5Y CAGR",r.get("revenue_cagr_5y")),
        (gc[2],"PAT 3Y CAGR",    r.get("pat_cagr_3y")),
        (gc[3],"PAT 5Y CAGR",    r.get("pat_cagr_5y")),
        (gc[4],"EPS 3Y CAGR",    r.get("eps_cagr_3y")),
        (gc[5],"EPS 5Y CAGR",    r.get("eps_cagr_5y")),
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
# ── TAB 2: RATIOS ──
with tab_rat:
    sec("Key Ratio Snapshot (Latest Year)")
    rc = st.columns(6)
    for col, label, key in [
        (rc[0],"Operating Margin","operating_margin"),
        (rc[1],"Net Margin","net_margin"),
        (rc[2],"ROE","roe"),
        (rc[3],"ROCE","roce"),
        (rc[4],"Current Ratio","current_ratio"),
        (rc[5],"Debt/Equity","debt_equity"),
    ]:
        with col:
            suffix = "%" if "margin" in key.lower() or key in ("roe","roce") else "x"
            st.metric(label, fmt(last_valid(r.get(key,[])), suffix))
    st.markdown("<br>", unsafe_allow_html=True)
    sec("All Ratios — 5-Year Trend")
    try:
        ratio_df = pd.DataFrame({
            "Operating Margin %": r.get("operating_margin",[]),
            "Net Margin %":       r.get("net_margin",[]),
            "ROE %":              r.get("roe",[]),
            "ROA %":              r.get("roa",[]),
            "ROCE %":             r.get("roce",[]),
            "Current Ratio":      r.get("current_ratio",[]),
            "Quick Ratio":        r.get("quick_ratio",[]),
            "Debt/Equity":        r.get("debt_equity",[]),
            "Interest Coverage":  r.get("interest_cover",[]),
            "Asset Turnover":     r.get("asset_turnover",[]),
            "Receivable Days":    r.get("receivable_days",[]),
            "Inventory Days":     r.get("inventory_days",[]),
            "Payable Days":       r.get("payable_days",[]),
            "Cash Conversion":    r.get("cash_conversion",[]),
        }, index=years).T
        ratio_df = ratio_df.applymap(
            lambda v: f"{float(v):,.2f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(ratio_df, use_container_width=True, height=420)
    except Exception as e:
        st.warning(f"Could not build ratio table: {e}")
    plotly(chart_margins(r, ticker))

# ── TAB 3: CASH FLOW ──
with tab_cf:
    sec("Cash Flow Analysis")
    cfc = st.columns(4)
    for col, label, key in [
        (cfc[0],"Latest CFO","cfo"),(cfc[1],"Latest CFI","cfi"),
        (cfc[2],"Latest CFF","cff"),(cfc[3],"Latest FCF","fcf"),
    ]:
        with col:
            st.metric(label, fmt(last_valid(r.get(key,[])), suffix=" Cr", prefix="₹", dec=0))
    plotly(chart_cashflow_trend(r, ticker))
    sec("Cash Flow Summary")
    try:
        cfo_pat = []
        for c, p in zip(r.get("cfo",[]), r.get("pat",[])):
            try:
                fc, fp = float(c), float(p)
                cfo_pat.append(f"{fc/fp:.2f}x" if fp!=0 and not np.isnan(fc) and not np.isnan(fp) else "—")
            except Exception:
                cfo_pat.append("—")
        cf_df = pd.DataFrame({
            "CFO (₹ Cr)":r.get("cfo",[]),"CFI (₹ Cr)":r.get("cfi",[]),
            "CFF (₹ Cr)":r.get("cff",[]),"FCF (₹ Cr)":r.get("fcf",[]),
            "PAT (₹ Cr)":r.get("pat",[]),"CFO / PAT":cfo_pat,
        }, index=years).T
        cf_df = cf_df.applymap(lambda v: v if isinstance(v,str)
                               else (f"{float(v):,.0f}" if v is not None and not np.isnan(float(v)) else "—"))
        st.dataframe(cf_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build CF table: {e}")
    with st.expander("📋 Raw Cash Flow Statement"):
        raw_cf = data.get("cf", pd.DataFrame())

        # FIX
        if not raw_cf.empty:
            st.dataframe(raw_cf, use_container_width=True)
        else:
            st.info("Not available.")# ── TAB 4: RED FLAGS ──
with tab_flags:
    sec("Red Flag Analysis")
    rfc = st.columns(3)
    for col, label, val, color in [
        (rfc[0],"High Severity",  flag_summ["high"],  "#E74C3C"),
        (rfc[1],"Medium Severity",flag_summ["medium"],"#F39C12"),
        (rfc[2],"Checks Passed",  flag_summ["ok"],    "#27AE60"),
    ]:
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;border-color:{color};">'
                f'<div style="font-size:36px;font-weight:800;color:{color};">{val}</div>'
                f'<div style="color:#8B9AB5;font-size:13px;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    for flag in flags:
        cls  = ("flag-high" if flag["severity"]=="high" else "flag-medium") if flag["triggered"] else "flag-ok"
        icon = ("🔴" if flag["severity"]=="high" else "🟡") if flag["triggered"] else "✅"
        st.markdown(
            f'<div class="{cls}">'
            f'<div style="font-size:14px;font-weight:700;color:#E2E8F0;margin-bottom:4px;">'
            f'{icon} {flag["name"]} — {flag["check"]}</div>'
            f'<div style="font-size:12px;color:#B0BCC8;margin-bottom:4px;">{flag["detail"]}</div>'
            f'<div style="font-size:11px;color:#8B9AB5;">{flag["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── TAB 5: BUSINESS QUALITY ──
with tab_quality:
    qs = final_assumptions.get("quality_score", 0.0)
    qc = final_assumptions.get("quality_components", {})
    biz_type = final_assumptions.get("business_type", "stable")
    biz_label, biz_color = BIZ_TYPE_LABELS.get(biz_type, ("Stable","#4A90D9"))
    biz_tooltip = BIZ_TYPE_TOOLTIPS.get(biz_type, "")
    qs_color = "#27AE60" if qs >= 70 else ("#F39C12" if qs >= 45 else "#E74C3C")

    sec("Business Classification & Quality Score")

    # Business type card
    st.markdown(
        f'<div class="card" style="border-color:{biz_color}44;">'
        f'<div style="font-size:11px;color:#8B9AB5;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Business Type</div>'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:22px;font-weight:800;color:{biz_color};">{biz_label}</span>'
        f'</div>'
        f'<div style="font-size:12px;color:#B0BCC8;margin-top:8px;">{biz_tooltip}</div>'
        f'<div style="margin-top:12px;font-size:11px;color:#8B9AB5;line-height:1.8;">'
        f'<strong style="color:#B0BCC8;">Growth ceiling:</strong> '
        f'{"30%" if biz_type=="high-margin-stable" else "25%" if biz_type=="stable" else "20%" if biz_type=="cyclical" else "18%"}'
        f' &nbsp;|&nbsp; '
        f'<strong style="color:#B0BCC8;">MOS premium:</strong> '
        f'{"No extra" if biz_type in ("high-margin-stable","stable") else "+3%" if biz_type=="cyclical" else "+5%"}'
        f' &nbsp;|&nbsp; '
        f'<strong style="color:#B0BCC8;">Beta prior:</strong> '
        f'{"0.70" if biz_type=="high-margin-stable" else "0.80" if biz_type=="stable" else "1.00" if biz_type=="cyclical" else "1.10"}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Overall quality score
    st.markdown(
        f'<div class="card" style="text-align:center;border-color:{qs_color};">'
        f'<div style="font-size:11px;color:#8B9AB5;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Overall Business Quality Score</div>'
        f'<div style="font-size:52px;font-weight:800;color:{qs_color};">{qs:.0f}</div>'
        f'<div style="font-size:14px;color:#8B9AB5;">out of 100</div>'
        f'<div style="margin-top:12px;font-size:12px;color:#B0BCC8;">'
        f'{"🏆 High quality — deserves a premium valuation" if qs >= 70 else "⚠️ Moderate quality — invest with appropriate margin of safety" if qs >= 45 else "🔴 Low quality — avoid BUY; monitor improvement"}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Component breakdown
    sec("Quality Component Breakdown")
    q_col1, q_col2 = st.columns(2)

    component_meta = {
        "financial_health": ("💰 Financial Health", "Debt/Equity trend, interest coverage, current ratio"),
        "profitability":    ("📈 Profitability",     "ROCE level, operating margin stability & trend"),
        "cash_flow_quality":("💵 Cash Flow Quality", "CFO/PAT ratio, FCF conversion, positive FCF years"),
        "growth_quality":   ("🚀 Growth Quality",    "Revenue CAGR consistency, EPS vs Revenue growth, PAT consistency"),
    }

    sub_detail_labels = {
        # financial_health
        "debt_equity":         "Debt/Equity score",
        "interest_coverage":   "Interest coverage score",
        "current_ratio":       "Current ratio score",
        # profitability
        "roce":                "ROCE score",
        "margin_stability":    "Margin stability score",
        "margin_trend":        "Margin trend score",
        # cash_flow_quality
        "cfo_pat_ratio":       "CFO/PAT ratio score",
        "fcf_conversion":      "FCF/EBITDA conversion score",
        "positive_fcf_years":  "Positive FCF years score",
        # growth_quality
        "revenue_growth":      "Revenue growth score",
        "margin_on_growth":    "EPS vs Revenue CAGR score",
        "pat_consistency":     "PAT CAGR consistency score",
    }

    comp_items = list(component_meta.items())
    for i, (comp_key, (comp_label, comp_desc)) in enumerate(comp_items):
        target_col = q_col1 if i % 2 == 0 else q_col2
        with target_col:
            comp_data  = qc.get(comp_key, {})
            comp_score = comp_data.get("score", 0.0)
            comp_max   = comp_data.get("max",   30)
            comp_detail= comp_data.get("detail", {})

            sub_rows = "".join(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:11px;color:#8B9AB5;padding:2px 0;">'
                f'<span>{sub_detail_labels.get(k, k)}</span>'
                f'<span style="color:#7EB8F7;font-weight:600;">{v:.1f}</span>'
                f'</div>'
                for k, v in comp_detail.items()
            )

            pct   = int(comp_score / comp_max * 100) if comp_max > 0 else 0
            color = "#27AE60" if pct >= 66 else ("#F39C12" if pct >= 33 else "#E74C3C")
            st.markdown(
                f'<div class="card" style="padding:14px 16px;">'
                f'<div style="font-size:13px;font-weight:700;color:#E2E8F0;margin-bottom:4px;">{comp_label}</div>'
                f'<div style="font-size:11px;color:#8B9AB5;margin-bottom:8px;">{comp_desc}</div>'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#B0BCC8;">'
                f'<span>Score</span><span style="color:{color};font-weight:700;">{comp_score:.1f} / {comp_max}</span></div>'
                f'<div class="quality-bar-bg">'
                f'<div style="height:14px;width:{pct}%;background:{color};border-radius:8px;"></div>'
                f'</div>'
                f'<div style="margin-top:8px;">{sub_rows}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # MOS requirement explanation
    sec("Margin of Safety Requirements")
    mos_req_dcf = dcf_result.get("mos_required", 20.0) if dcf_result else 20.0
    st.markdown(
        f'<div class="card" style="font-size:12px;color:#B0BCC8;line-height:2.0;">'
        f'<div style="font-size:13px;font-weight:700;color:#E2E8F0;margin-bottom:10px;">'
        f'🎯 MOS Required for {ticker}: <span style="color:{qs_color};">{mos_req_dcf:.0f}%</span></div>'
        f'<div>Quality ≥ 80 &nbsp;→&nbsp; <strong style="color:#7EB8F7;">25%+</strong> upside required for BUY</div>'
        f'<div>Quality 60–80 &nbsp;→&nbsp; <strong style="color:#7EB8F7;">20%+</strong> upside required</div>'
        f'<div>Quality 40–60 &nbsp;→&nbsp; <strong style="color:#7EB8F7;">15%+</strong> upside required</div>'
        f'<div>Quality &lt; 40 &nbsp;→&nbsp; <strong style="color:#E74C3C;">No BUY signal</strong> regardless of upside</div>'
        f'<div style="margin-top:8px;color:#8B9AB5;">'
        f'Business type premium: Cyclical +3% | Commodity +5%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── TAB 6: DUPONT ──
with tab_dupont:
    sec("DuPont ROE Decomposition (3-Step)")
    st.markdown(
        '<div class="card" style="font-size:13px;color:#B0BCC8;">'
        '<strong style="color:#E2E8F0;">ROE = Net Profit Margin × Asset Turnover × Equity Multiplier</strong><br>'
        'Breaks down <em>why</em> ROE changed — profitability, efficiency, or leverage?'
        '</div>', unsafe_allow_html=True,
    )
    try:
        du_df = pd.DataFrame({
            "Net Margin %":          r.get("dupont_net_margin",[]),
            "Asset Turnover (×)":    r.get("dupont_asset_turnover",[]),
            "Equity Multiplier (×)": r.get("dupont_equity_mult",[]),
            "DuPont ROE %":          r.get("dupont_roe",[]),
        }, index=years).T
        du_df = du_df.applymap(
            lambda v: f"{float(v):,.2f}" if v is not None and not np.isnan(float(v)) else "—")
        st.dataframe(du_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not build DuPont table: {e}")
    plotly(chart_dupont(r, ticker))
    nm = last_valid(r.get("dupont_net_margin",[]))
    at = last_valid(r.get("dupont_asset_turnover",[]))
    em = last_valid(r.get("dupont_equity_mult",[]))
    if not any(np.isnan(v) for v in [nm, at, em]):
        st.markdown(
            '<div class="card">'
            '<div style="font-size:14px;font-weight:700;color:#E2E8F0;margin-bottom:8px;">Latest Year Interpretation</div>'
            f'<div style="font-size:13px;color:#B0BCC8;line-height:1.9;">'
            f'• <strong style="color:#E2E8F0;">Net Margin {nm:.1f}%</strong> — keeps ₹{nm:.1f} of every ₹100 revenue.<br>'
            f'• <strong style="color:#E2E8F0;">Asset Turnover {at:.2f}×</strong> — generates ₹{at:.2f} per ₹1 of assets.<br>'
            f'• <strong style="color:#E2E8F0;">Equity Multiplier {em:.2f}×</strong> — assets are {em:.2f}× larger than equity.'
            f'</div></div>', unsafe_allow_html=True,
        )

# ── TAB 7: PEERS ──
with tab_peers:
    sec("Peer Comparison")
    if peer_df is not None and len(peer_df) > 1:
        display_cols = [c for c in ["Company","Ticker","Mkt Cap (Cr)","CMP (₹)","P/E","P/B",
                                     "Net Margin %","ROE %","ROCE %","D/E","Current Ratio","Revenue CAGR 3Y %"]
                        if c in peer_df.columns]
        def highlight_rows(row):
            if row.get("_is_target") is True:
                return ["background-color:#1B3A5C;font-weight:bold"]*len(row)
            if row.get("_is_target") is None:
                return ["background-color:#1A3326;font-style:italic"]*len(row)
            return [""]*len(row)
        st.dataframe(peer_df[display_cols+["_is_target"]].copy().style.apply(highlight_rows,axis=1),
                     use_container_width=True, height=280)
        st.markdown('<small style="color:#8B9AB5;">★ Blue = target &nbsp;|&nbsp; Green = Industry Median</small>',
                    unsafe_allow_html=True)
        plotly(chart_peer_radar(peer_df, ticker))
    elif not fetch_peers:
        st.info("Enable 'Fetch peer comparison' in the sidebar.")
    else:
        st.warning(f"No peers found for {ticker}.")

# ── TAB 8: DCF VALUATION ──
with tab_dcf:
    if dcf_result is None:
        st.info("Enable 'Run DCF valuation' in the sidebar.")
    else:
        sec("Composite Valuation Signal")
        signal       = dcf_result.get("signal","N/A")
        conviction   = dcf_result.get("conviction","")
        comp_score   = dcf_result.get("composite_score",0.0)
        factor_sc    = dcf_result.get("factor_scores",{})
        upside       = dcf_result.get("upside_pct")
        base_iv      = dcf_result.get("base_iv")
        bear_iv      = dcf_result["scenarios"]["Bear"]["intrinsic_per_share"]
        bull_iv      = dcf_result["scenarios"]["Bull"]["intrinsic_per_share"]
        wacc_used    = dcf_result["wacc_result"]["wacc"] * 100
        mos_req      = dcf_result.get("mos_required", 20.0)
        quality_sc   = dcf_result.get("quality_score", 0.0)
        biz_type_dcf = dcf_result.get("business_type", "stable")
        sig_reason   = dcf_result.get("signal_reason", "")
        dcf_wt       = dcf_result.get("assumptions_used", {}).get("quality_score", quality_sc)
        auto_tag     = "🤖 History-derived" if final_assumptions.get("auto_derived") else "✏️ Manual override"

        sig_col, detail_col = st.columns([1,2])
        with sig_col:
            sig_color = {"BUY":"#27AE60","STRONG BUY":"#1a7a3c","SELL":"#E74C3C","HOLD":"#F39C12"}.get(signal,"#95A5A6")
            qs_color2 = "#27AE60" if quality_sc >= 70 else ("#F39C12" if quality_sc >= 45 else "#E74C3C")
            st.markdown(
                f'<div class="card" style="text-align:center;border-color:{sig_color};padding:30px 20px;">'
                f'<div style="font-size:11px;color:#8B9AB5;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Composite Signal</div>'
                f'<span class="signal-{signal.lower()}" style="font-size:28px;padding:10px 28px;">{signal}</span>'
                f'<div style="margin-top:14px;font-size:13px;color:#B0BCC8;">'
                f'{conviction} conviction<br>'
                f'<span style="color:{sig_color};font-weight:700;font-size:16px;">{comp_score:+.2f}</span> composite score</div>'
                f'<div style="margin-top:10px;border-top:1px solid #1E2A45;padding-top:10px;">'
                f'<div style="font-size:11px;color:#8B9AB5;">Business Quality</div>'
                f'<div style="font-size:20px;font-weight:800;color:{qs_color2};">{quality_sc:.0f}/100</div>'
                f'<div style="font-size:11px;color:#8B9AB5;">MOS Required: <span style="color:#7EB8F7;font-weight:600;">{mos_req:.0f}%</span></div>'
                f'</div>'
                f'<div style="margin-top:8px;font-size:10px;color:#8B9AB5;">{auto_tag}</div>'
                f'</div>', unsafe_allow_html=True,
            )

        with detail_col:
            dcf_s   = factor_sc.get("dcf",      {}).get("score",0.0)
            pe_s    = factor_sc.get("pe",        {}).get("score",0.0)
            ev_s    = factor_sc.get("ev_ebitda", {}).get("score",0.0)
            pe_data = factor_sc.get("pe",{})
            ev_data = factor_sc.get("ev_ebitda",{})

            # Dynamic weights from DCF result
            q = quality_sc
            w_dcf_pct = 65 if q >= 70 else 50 if q >= 50 else 40
            w_pe_pct  = 20 if q >= 70 else 25 if q >= 50 else 30
            w_ev_pct  = 15 if q >= 70 else 25 if q >= 50 else 30

            dcf_detail = (f"Base IV ₹{fmt(base_iv,dec=0)} | CMP ₹{fmt(data.get('current_price'),dec=0)} | Upside {fmt(upside,'%')} (MOS reqd {mos_req:.0f}%)")
            pe_detail  = (f"Current P/E {pe_data['current_pe']:.1f}x | Fair P/E {pe_data['fair_pe']:.1f}x | EPS CAGR {fmt(r.get('eps_cagr_3y'),'%')}"
                          if "current_pe" in pe_data else pe_data.get("note","Insufficient data"))
            ev_detail  = (f"Current EV/EBITDA {fmt(ev_data.get('current_ev_ebitda'),'x')} | Normalised {fmt(ev_data.get('hist_ev_ebitda_norm'),'x')}"
                          if "current_ev_ebitda" in ev_data else ev_data.get("note","Insufficient data"))
            st.markdown(
                f'<div class="card">'
                f'<div style="font-size:13px;font-weight:700;color:#E2E8F0;margin-bottom:14px;">'
                f'Factor Score Breakdown '
                f'<span style="font-size:10px;color:#8B9AB5;">(Quality {quality_sc:.0f}/100 → dynamic weights)</span></div>'
                + score_bar_html(dcf_s, f"📐 DCF Intrinsic Value vs CMP", w_dcf_pct)
                + score_bar_html(pe_s,  f"📊 P/E Mean Reversion (PEG anchor)", w_pe_pct)
                + score_bar_html(ev_s,  f"🏭 EV/EBITDA vs Historical Norm", w_ev_pct)
                + f'<div style="margin-top:12px;font-size:11px;color:#8B9AB5;line-height:1.9;">'
                f'<strong style="color:#B0BCC8;">DCF:</strong> {dcf_detail}<br>'
                f'<strong style="color:#B0BCC8;">P/E:</strong> {pe_detail}<br>'
                f'<strong style="color:#B0BCC8;">EV/EBITDA:</strong> {ev_detail}'
                f'</div></div>', unsafe_allow_html=True,
            )

        # Signal reason card
        if sig_reason:
            st.markdown(
                f'<div class="card" style="border-color:#1B4F8A44;">'
                f'<div style="font-size:12px;font-weight:700;color:#7EB8F7;margin-bottom:6px;">📋 Signal Rationale</div>'
                f'<div style="font-size:12px;color:#B0BCC8;line-height:1.8;">{sig_reason}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # How assumptions were derived
        if final_assumptions.get("auto_derived"):
            aa_d       = final_assumptions
            raw_g      = aa_d.get("raw_growth_pct", aa_d["base_growth"]*100)
            de_r       = aa_d.get("de_ratio",0)
            biz_t      = aa_d.get("business_type","stable")
            biz_lbl2, biz_c2 = BIZ_TYPE_LABELS.get(biz_t, ("Stable","#4A90D9"))
            peer_mg    = aa_d.get("peer_median_margin")
            peer_mg_str = f" + peer median {peer_mg:.1f}%" if peer_mg else ""
            st.markdown(
                f'<div class="card" style="border-color:#1B4F8A;">'
                f'<div style="font-size:12px;font-weight:700;color:#7EB8F7;margin-bottom:10px;">'
                f'🤖 How assumptions were derived for {ticker} '
                f'<span style="color:{biz_c2};">({biz_lbl2})</span></div>'
                f'<div style="font-size:11px;color:#B0BCC8;line-height:1.9;">'
                f'• <b>Revenue growth {aa_d["base_growth"]*100:.1f}%</b>: '
                f'3Y CAGR ({fmt(r.get("revenue_cagr_3y"),"%")}) ×60% + 5Y CAGR ({fmt(r.get("revenue_cagr_5y"),"%")}) ×40% '
                f'→ raw {raw_g:.1f}%, mean-reverted {int(aa_d.get("mr_weight",0.30)*100) if "mr_weight" in aa_d else "30"}% '
                f'toward {biz_t} long-run anchor, capped at {("30%" if biz_t=="high-margin-stable" else "25%" if biz_t=="stable" else "20%" if biz_t=="cyclical" else "18%")}<br>'
                f'• <b>EBITDA margin {aa_d["base_ebitda_margin"]:.1f}%</b>: '
                f'5Y historical median{peer_mg_str} | ceiling {("60%" if biz_t=="high-margin-stable" else "45%" if biz_t=="stable" else "35%" if biz_t=="cyclical" else "20%")} for {biz_t}<br>'
                f'• <b>WACC {aa_d["base_wacc"]*100:.1f}%</b>: '
                f'Hamada β={aa_d["beta"]:.2f} (unlevered prior {("0.70" if biz_t=="high-margin-stable" else "0.80" if biz_t=="stable" else "1.00" if biz_t=="cyclical" else "1.10")}) '
                f'from D/E={de_r:.2f}x | cost of debt includes credit quality floor<br>'
                f'• <b>Effective tax rate {aa_d["tax_rate"]*100:.1f}%</b>: derived from tax expense / PBT (or 25% default)<br>'
                f'• <b>Capex {aa_d["capex_pct"]*100:.1f}%</b>: median of historical capex/revenue<br>'
                f'• <b>Terminal growth {aa_d["base_tgr"]*100:.1f}%</b>: '
                f'max(GDP/2=3.5%, {biz_t} floor {("4%" if biz_t=="high-margin-stable" else "3%")}), bounded [3%,6%], sanity-checked vs WACC'
                f'</div></div>', unsafe_allow_html=True,
            )

        st.divider()
        sec("DCF Scenarios — Bull / Base / Bear")
        dh1,dh2,dh3,dh4 = st.columns(4)
        with dh1: st.metric("Base Case IV", fmt(base_iv,prefix="₹",dec=0))
        with dh2: st.metric("Bear Case IV", fmt(bear_iv,prefix="₹",dec=0))
        with dh3: st.metric("Bull Case IV", fmt(bull_iv,prefix="₹",dec=0))
        with dh4: st.metric("WACC Used",    fmt(wacc_used,"%"))

        wr = dcf_result["wacc_result"]
        with st.expander("🔍 WACC Breakdown"):
            wc = st.columns(5)
            for col, label, val in [
                (wc[0],"WACC",           wr["wacc"]*100),
                (wc[1],"Cost of Equity", wr["cost_of_equity"]*100),
                (wc[2],"After-tax CoD",  wr["cost_of_debt_after_tax"]*100),
                (wc[3],"Equity Weight",  wr["equity_weight"]*100),
                (wc[4],"Debt Weight",    wr["debt_weight"]*100),
            ]:
                col.metric(label, fmt(val,"%"))

        plotly(chart_scenario_comparison(dcf_result, data.get("current_price",0)))
        sec("Projected Cash Flows (₹ Cr)")
        sc1,sc2,sc3 = st.columns(3)
        for col, scenario, icon in [(sc1,"Bear","🔴"),(sc2,"Base","🟡"),(sc3,"Bull","🟢")]:
            with col:
                s = dcf_result["scenarios"][scenario]
                st.markdown(f"**{icon} {scenario} Case**")
                fcff_df = s.get("fcff_df")
                if fcff_df is not None:
                    st.dataframe(fcff_df[["Year","Revenue","EBITDA","FCFF"]].set_index("Year"),
                                 use_container_width=True)
                st.metric("Intrinsic Value / Share", fmt(s["intrinsic_per_share"],prefix="₹",dec=0))
                st.metric("Enterprise Value", fmt(s["enterprise_value"],suffix=" Cr",prefix="₹",dec=0))

        plotly(chart_dcf_waterfall(dcf_result,"Base"))
        if sens_df is not None and not sens_df.empty:
            sec("Sensitivity Analysis — Tornado Chart")
            plotly(chart_tornado(sens_df, base_iv or 0))
            st.dataframe(sens_df.set_index("Variable")[["Low IV","Base IV","High IV","Impact"]],
                         use_container_width=True)

# ── TAB 9: MONTE CARLO ──
with tab_mc:
    if mc_result is None:
        st.info("Enable 'Run Monte Carlo' in the sidebar.")
    elif "error" in mc_result:
        st.error(mc_result["error"])
    else:
        sec("Monte Carlo Simulation — 10,000 Iterations")
        st.markdown(
            '<div class="card" style="font-size:13px;color:#B0BCC8;">'
            'Each run randomizes inputs using a normal distribution centred on '
            '<strong style="color:#E2E8F0;">history-derived base assumptions</strong>. '
            'Growth std dev is calibrated from actual year-on-year revenue volatility.</div>',
            unsafe_allow_html=True,
        )
        mcc = st.columns(6)
        for col, label, val in [
            (mcc[0],"Mean IV",    mc_result["mean"]),
            (mcc[1],"Median IV",  mc_result["median"]),
            (mcc[2],"P10 (Bear)", mc_result["p10"]),
            (mcc[3],"P25",        mc_result["p25"]),
            (mcc[4],"P75",        mc_result["p75"]),
            (mcc[5],"P90 (Bull)", mc_result["p90"]),
        ]:
            with col:
                st.metric(label, fmt(val,prefix="₹",dec=0))
        prob = mc_result["prob_undervalued"]
        cmp  = mc_result["current_price"]
        st.markdown(
            f'<div class="card">'
            f'<span style="color:#4A90D9;font-size:20px;font-weight:700;">{prob}%</span>'
            f'<span style="color:#B0BCC8;font-size:14px;"> of simulations suggest the stock is '
            f'undervalued vs CMP {fmt(cmp,prefix="₹",dec=0)}</span>'
            f'</div>', unsafe_allow_html=True,
        )
        plotly(chart_monte_carlo(mc_result, ticker))
        dist_df = pd.DataFrame({
            "Percentile":["P10","P25","Median","Mean","P75","P90"],
            "Intrinsic Value (₹)":[mc_result["p10"],mc_result["p25"],mc_result["median"],
                                   mc_result["mean"],mc_result["p75"],mc_result["p90"]],
            "vs CMP":[f"{(v-cmp)/cmp*100:.1f}%" if cmp else "—"
                      for v in [mc_result["p10"],mc_result["p25"],mc_result["median"],
                                mc_result["mean"],mc_result["p75"],mc_result["p90"]]],
        })
        st.dataframe(dist_df.set_index("Percentile"), use_container_width=True)

# ── FOOTER ──
st.divider()
st.markdown(
    '<div style="text-align:center;color:#8B9AB5;font-size:12px;padding:10px 0;">'
    'EquityLens — Automated equity research for NSE-listed companies.<br>'
    'Data: <a href="https://www.screener.in" target="_blank" style="color:#4A90D9;">Screener.in</a>'
    ' &nbsp;|&nbsp; Not investment advice. All figures ₹ Crore unless stated.'
    '</div>', unsafe_allow_html=True,
)