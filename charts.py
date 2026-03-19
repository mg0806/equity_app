"""
charts.py — All interactive Plotly charts for the Streamlit UI.
Every function returns a plotly Figure object.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Brand colors
BLUE       = "#1B4F8A"
LIGHT_BLUE = "#4A90D9"
GREEN      = "#27AE60"
RED        = "#E74C3C"
ORANGE     = "#F39C12"
GREY       = "#95A5A6"
BG         = "#0E1117"
CARD_BG    = "#1C2333"
TEXT       = "#FAFAFA"
GRID       = "#2D3748"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig


# ─────────────────────────────────────────────
# FINANCIAL TREND CHART
# ─────────────────────────────────────────────

def chart_financial_trend(r: dict, ticker: str) -> go.Figure:
    """Revenue, EBITDA, PAT trend as grouped bars."""
    years   = r["years"]
    revenue = [v if not np.isnan(float(v or np.nan)) else None for v in r.get("revenue", [])]
    ebitda  = [v if not np.isnan(float(v or np.nan)) else None for v in r.get("ebitda", [])]
    pat     = [v if not np.isnan(float(v or np.nan)) else None for v in r.get("pat", [])]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=years, y=revenue,
                         marker_color=BLUE, opacity=0.9))
    fig.add_trace(go.Bar(name="EBITDA",  x=years, y=ebitda,
                         marker_color=LIGHT_BLUE, opacity=0.9))
    fig.add_trace(go.Bar(name="PAT",     x=years, y=pat,
                         marker_color=GREEN, opacity=0.9))

    fig.update_layout(
        barmode="group",
        title=f"{ticker} — Revenue, EBITDA & PAT (₹ Cr)",
        xaxis_title="Year",
        yaxis_title="₹ Crore",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_THEME,
    )
    return fig


def chart_cashflow_trend(r: dict, ticker: str) -> go.Figure:
    """CFO, CFI, CFF, FCF trend."""
    years = r["years"]

    def clean(lst):
        return [v if (v is not None and not np.isnan(float(v))) else None for v in lst]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="CFO (Operating)", x=years, y=clean(r.get("cfo", [])),
                         marker_color=GREEN))
    fig.add_trace(go.Bar(name="CFI (Investing)",  x=years, y=clean(r.get("cfi", [])),
                         marker_color=RED))
    fig.add_trace(go.Bar(name="CFF (Financing)",  x=years, y=clean(r.get("cff", [])),
                         marker_color=ORANGE))
    fig.add_trace(go.Scatter(name="Free Cash Flow", x=years, y=clean(r.get("fcf", [])),
                             mode="lines+markers", line=dict(color=LIGHT_BLUE, width=2.5),
                             marker=dict(size=8)))

    fig.update_layout(
        barmode="group",
        title=f"{ticker} — Cash Flow Analysis (₹ Cr)",
        xaxis_title="Year",
        yaxis_title="₹ Crore",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# MARGINS CHART
# ─────────────────────────────────────────────

def chart_margins(r: dict, ticker: str) -> go.Figure:
    """Operating margin, net margin, ROE, ROCE trend lines."""
    years = r["years"]

    def clean(lst):
        return [round(float(v), 1) if (v is not None and not np.isnan(float(v))) else None for v in lst]

    fig = go.Figure()
    metrics = [
        ("Operating Margin %", r.get("operating_margin", []), BLUE),
        ("Net Margin %",       r.get("net_margin", []),       LIGHT_BLUE),
        ("ROE %",              r.get("roe", []),              GREEN),
        ("ROCE %",             r.get("roce", []),             ORANGE),
    ]
    for name, vals, color in metrics:
        fig.add_trace(go.Scatter(
            name=name, x=years, y=clean(vals),
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title=f"{ticker} — Profitability & Return Metrics (%)",
        xaxis_title="Year",
        yaxis_title="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# DUPONT CHART
# ─────────────────────────────────────────────

def chart_dupont(r: dict, ticker: str) -> go.Figure:
    """DuPont decomposition with ROE line + component bars."""
    years = r["years"]

    def clean(lst):
        return [round(float(v), 2) if (v is not None and not np.isnan(float(v))) else None for v in lst]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        name="Net Margin %", x=years,
        y=clean(r.get("dupont_net_margin", [])),
        marker_color=BLUE, opacity=0.75,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        name="Asset Turnover ×10", x=years,
        y=[v*10 if v else None for v in clean(r.get("dupont_asset_turnover", []))],
        marker_color=LIGHT_BLUE, opacity=0.75,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        name="Equity Multiplier", x=years,
        y=clean(r.get("dupont_equity_mult", [])),
        marker_color=ORANGE, opacity=0.75,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name="DuPont ROE %", x=years,
        y=clean(r.get("dupont_roe", [])),
        mode="lines+markers",
        line=dict(color=RED, width=3),
        marker=dict(size=9, symbol="diamond"),
    ), secondary_y=True)

    fig.update_yaxes(title_text="Component Value", secondary_y=False,
                     gridcolor=GRID, color=TEXT)
    fig.update_yaxes(title_text="DuPont ROE %", secondary_y=True,
                     gridcolor=GRID, color=RED)
    fig.update_layout(
        barmode="group",
        title=f"{ticker} — DuPont ROE Decomposition",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# BALANCE SHEET COMPOSITION
# ─────────────────────────────────────────────

def chart_balance_sheet(r: dict, ticker: str) -> go.Figure:
    """Stacked bar: Equity vs Debt composition over years."""
    years = r["years"]

    def clean(lst):
        return [float(v) if (v is not None and not np.isnan(float(v))) else 0 for v in lst]

    equity = clean(r.get("equity", []))
    debt   = clean(r.get("total_debt", []))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Equity", x=years, y=equity, marker_color=GREEN))
    fig.add_trace(go.Bar(name="Total Debt", x=years, y=debt, marker_color=RED))

    fig.update_layout(
        barmode="stack",
        title=f"{ticker} — Capital Structure (₹ Cr)",
        xaxis_title="Year", yaxis_title="₹ Crore",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# DCF WATERFALL
# ─────────────────────────────────────────────

def chart_dcf_waterfall(dcf_result: dict, scenario: str = "Base") -> go.Figure:
    """DCF bridge: PV of FCFFs + Terminal Value → Enterprise Value → Equity Value."""
    s = dcf_result["scenarios"][scenario]
    pv_fcffs = s["pv_fcffs"]
    pv_tv    = s["pv_terminal_value"]
    ev       = s["enterprise_value"]

    labels = [f"PV FCF Yr{i+1}" for i in range(len(pv_fcffs))] + ["PV Terminal Value", "Enterprise Value"]
    values = pv_fcffs + [pv_tv, None]
    measures = ["relative"] * len(pv_fcffs) + ["relative", "total"]

    fig = go.Figure(go.Waterfall(
        name=scenario,
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector=dict(line=dict(color=GRID, width=1)),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)),
        text=[f"₹{v:,.0f}" if v else "" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"DCF Bridge — {scenario} Case (₹ Cr)",
        yaxis_title="₹ Crore",
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# SCENARIO COMPARISON
# ─────────────────────────────────────────────

def chart_scenario_comparison(dcf_result: dict, current_price: float) -> go.Figure:
    """Bull / Base / Bear intrinsic value vs current market price."""
    scenarios = ["Bear", "Base", "Bull"]
    colors_s  = [RED, ORANGE, GREEN]
    ivs       = [dcf_result["scenarios"][s]["intrinsic_per_share"] for s in scenarios]

    fig = go.Figure()
    for s, iv, color in zip(scenarios, ivs, colors_s):
        fig.add_trace(go.Bar(
            name=s, x=[s], y=[iv],
            marker_color=color, opacity=0.85,
            text=[f"₹{iv:,.0f}" if iv else "N/A"],
            textposition="outside",
        ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color=LIGHT_BLUE,
        annotation_text=f"CMP ₹{current_price:,.0f}",
        annotation_position="right",
    )
    fig.update_layout(
        title="Intrinsic Value — Scenario Comparison vs CMP",
        yaxis_title="₹ per Share",
        showlegend=False,
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# MONTE CARLO HISTOGRAM
# ─────────────────────────────────────────────

def chart_monte_carlo(mc_result: dict, ticker: str) -> go.Figure:
    """Histogram of 10,000 Monte Carlo intrinsic value estimates."""
    values        = mc_result["values"]
    current_price = mc_result["current_price"]
    mean_iv       = mc_result["mean"]
    p10           = mc_result["p10"]
    p90           = mc_result["p90"]

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=80,
        name="Intrinsic Value Distribution",
        marker_color=BLUE,
        opacity=0.75,
    ))

    # Vertical lines
    for x, label, color, dash in [
        (current_price, f"CMP ₹{current_price:,.0f}", LIGHT_BLUE, "dash"),
        (mean_iv,       f"Mean ₹{mean_iv:,.0f}",      GREEN,       "solid"),
        (p10,           f"P10 ₹{p10:,.0f}",           RED,         "dot"),
        (p90,           f"P90 ₹{p90:,.0f}",           GREEN,       "dot"),
    ]:
        fig.add_vline(x=x, line_dash=dash, line_color=color,
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title=f"{ticker} — Monte Carlo Simulation ({mc_result['n_simulations']:,} runs)",
        xaxis_title="Intrinsic Value per Share (₹)",
        yaxis_title="Frequency",
        showlegend=False,
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# TORNADO CHART (SENSITIVITY)
# ─────────────────────────────────────────────

def chart_tornado(sensitivity_df: pd.DataFrame, base_iv: float) -> go.Figure:
    """Horizontal tornado chart showing sensitivity of IV to each assumption."""
    df = sensitivity_df.sort_values("Impact", ascending=True)

    fig = go.Figure()

    for _, row in df.iterrows():
        low_iv  = row["Low IV"]  or base_iv
        high_iv = row["High IV"] or base_iv
        fig.add_trace(go.Bar(
            name=row["Variable"],
            y=[row["Variable"]],
            x=[high_iv - base_iv],
            orientation="h",
            base=base_iv,
            marker_color=GREEN,
            opacity=0.8,
            showlegend=False,
            text=f"+₹{high_iv-base_iv:,.0f}",
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name=row["Variable"] + "_low",
            y=[row["Variable"]],
            x=[low_iv - base_iv],
            orientation="h",
            base=base_iv,
            marker_color=RED,
            opacity=0.8,
            showlegend=False,
            text=f"₹{low_iv-base_iv:,.0f}",
            textposition="outside",
        ))

    fig.add_vline(x=base_iv, line_dash="solid", line_color=LIGHT_BLUE, line_width=2,
                  annotation_text=f"Base IV ₹{base_iv:,.0f}")

    fig.update_layout(
        barmode="overlay",
        title="Sensitivity Analysis — Impact on Intrinsic Value (₹)",
        xaxis_title="Intrinsic Value per Share (₹)",
        yaxis_title="",
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# PEER COMPARISON RADAR
# ─────────────────────────────────────────────

def chart_peer_radar(peer_df: pd.DataFrame, target_ticker: str) -> go.Figure:
    """Radar chart comparing target vs peers on key metrics."""
    metrics = ["Net Margin %", "ROE %", "ROCE %", "Current Ratio"]
    available = [m for m in metrics if m in peer_df.columns]
    if not available or len(peer_df) < 2:
        return go.Figure()

    fig = go.Figure()
    colors_list = [BLUE, GREEN, ORANGE, RED, GREY, LIGHT_BLUE]

    for i, (_, row) in enumerate(peer_df.iterrows()):
        if row.get("_is_target") is None:
            continue  # skip median row
        vals = []
        for m in available:
            v = row.get(m)
            try:
                vals.append(float(v) if v is not None and not np.isnan(float(v)) else 0)
            except:
                vals.append(0)
        vals.append(vals[0])  # close the loop

        name = row.get("Ticker", str(i))
        is_target = row.get("_is_target", False)
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=available + [available[0]],
            name=name,
            line=dict(
                color=BLUE if is_target else colors_list[i % len(colors_list)],
                width=3 if is_target else 1.5,
            ),
            opacity=1.0 if is_target else 0.65,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor=GRID, color=TEXT),
            angularaxis=dict(gridcolor=GRID, color=TEXT),
        ),
        title="Peer Comparison — Key Metrics Radar",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        **PLOTLY_THEME,
    )
    return fig


# ─────────────────────────────────────────────
# GAUGE CHART (for a single metric)
# ─────────────────────────────────────────────

def chart_gauge(value: float, title: str, min_val: float, max_val: float,
                threshold_low: float, threshold_high: float) -> go.Figure:
    """Simple gauge for displaying a key ratio."""
    if value is None or np.isnan(float(value if value else np.nan)):
        value = 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": TEXT}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": TEXT},
            "bar": {"color": BLUE},
            "bgcolor": CARD_BG,
            "steps": [
                {"range": [min_val, threshold_low], "color": RED},
                {"range": [threshold_low, threshold_high], "color": ORANGE},
                {"range": [threshold_high, max_val], "color": GREEN},
            ],
            "threshold": {
                "line": {"color": WHITE if True else GREY, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
        number={"font": {"color": TEXT}},
    ))
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT), margin=dict(l=20, r=20, t=40, b=10))
    return fig


WHITE = "#FFFFFF"
