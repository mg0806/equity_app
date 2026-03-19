"""
report_pdf.py — Generates a formatted 2-page PDF equity research report.
Uses ReportLab. Returns PDF as bytes (for Streamlit download button).
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, Image, HRFlowable)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

# ── Color palette ──
DARK_BLUE  = colors.HexColor("#1B4F8A")
MID_BLUE   = colors.HexColor("#4A90D9")
LIGHT_BLUE = colors.HexColor("#D6E8FF")
RED        = colors.HexColor("#E74C3C")
GREEN      = colors.HexColor("#1E8449")
AMBER      = colors.HexColor("#F39C12")
LIGHT_GREY = colors.HexColor("#F2F4F7")
WHITE      = colors.white
BLACK      = colors.black


def _fmt(val, suffix="", prefix="", decimals=1, na="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    try:
        return f"{prefix}{float(val):,.{decimals}f}{suffix}"
    except:
        return na


def _lv(lst, default=np.nan):
    if isinstance(lst, (int, float)):
        return float(lst)
    if not lst:
        return default
    for v in reversed(lst):
        try:
            f = float(v)
            if not np.isnan(f):
                return f
        except:
            pass
    return default


def _make_trend_chart(r: dict, ticker: str) -> io.BytesIO:
    """Matplotlib chart embedded in PDF."""
    years   = r["years"]
    revenue = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("revenue", [])]
    pat     = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("pat", [])]
    fcf     = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("fcf", [])]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x, w = np.arange(len(years)), 0.28
    ax.bar(x - w, revenue, w, label="Revenue", color="#1B4F8A", alpha=0.88)
    ax.bar(x,     pat,     w, label="PAT",     color="#4A90D9", alpha=0.88)
    ax.bar(x + w, fcf,     w, label="FCF",     color="#9DC3E6", alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(years, fontsize=9)
    ax.set_ylabel("₹ Cr", fontsize=9)
    ax.set_title(f"{ticker} — Revenue, PAT & FCF", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}K" if abs(v) >= 1000 else f"{v:.0f}"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf


def _make_dupont_chart(r: dict, ticker: str) -> io.BytesIO:
    years = r["years"]
    nm  = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("dupont_net_margin", [])]
    at  = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("dupont_asset_turnover", [])]
    em  = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("dupont_equity_mult", [])]
    roe = [float(v) if v and not np.isnan(float(v)) else 0 for v in r.get("dupont_roe", [])]
    x   = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax2 = ax.twinx()
    ax2.bar(x - 0.2, nm,           0.18, label="Net Margin %",       color="#1B4F8A", alpha=0.7)
    ax2.bar(x,       [v*10 for v in at], 0.18, label="Asset Turnover×10",color="#4A90D9", alpha=0.7)
    ax2.bar(x + 0.2, em,           0.18, label="Equity Multiplier",  color="#9DC3E6", alpha=0.7)
    ax.plot(x, roe, "o-", color="#E74C3C", linewidth=2.2, label="DuPont ROE %", markersize=7, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(years, fontsize=9)
    ax.set_ylabel("ROE %", color="#E74C3C", fontsize=9)
    ax2.set_ylabel("Component", fontsize=9)
    ax.set_title("DuPont ROE Decomposition", fontsize=10, fontweight="bold")
    lines, lbls = ax.get_legend_handles_labels()
    bars, blbls = ax2.get_legend_handles_labels()
    ax.legend(lines+bars, lbls+blbls, fontsize=7, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf


def generate_pdf(ticker: str, data: dict, r: dict, flags: list,
                 peer_df=None, dcf_result: dict = None) -> bytes:
    """
    Generate full 2-page PDF report. Returns bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=14*mm, leftMargin=14*mm,
                             topMargin=10*mm, bottomMargin=10*mm)

    title_s = ParagraphStyle("ts", fontSize=15, fontName="Helvetica-Bold",
                               textColor=WHITE, leading=18)
    sub_s   = ParagraphStyle("ss", fontSize=8.5, fontName="Helvetica",
                               textColor=colors.HexColor("#C5DCF5"))
    er_s    = ParagraphStyle("er", fontSize=9, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#C5DCF5"), alignment=TA_RIGHT)
    sec_s   = ParagraphStyle("sh", fontSize=10, fontName="Helvetica-Bold",
                               textColor=DARK_BLUE, spaceBefore=6, spaceAfter=3)
    body_s  = ParagraphStyle("bd", fontSize=8, fontName="Helvetica",
                               textColor=BLACK, leading=11)
    red_s   = ParagraphStyle("rd", fontSize=8, fontName="Helvetica",
                               textColor=RED, leading=11)
    green_s = ParagraphStyle("gn", fontSize=8, fontName="Helvetica",
                               textColor=GREEN, leading=11)
    small_s = ParagraphStyle("sm", fontSize=7, fontName="Helvetica",
                               textColor=colors.grey, leading=10)

    def tbl_base():
        return [
            ("BACKGROUND",    (0,0),(-1,0),  DARK_BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ALIGN",         (1,0),(-1,-1), "RIGHT"),
            ("ALIGN",         (0,0),(0,-1),  "LEFT"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT_GREY, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#C0C0C0")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(0,-1),  5),
        ]

    story = []
    company = data.get("company_name", ticker)
    yrs = r["years"]

    # ── PAGE 1 HEADER ──
    price   = _fmt(data.get("current_price"),  prefix="₹", decimals=2)
    mktcap  = _fmt(data.get("market_cap"),     suffix=" Cr", prefix="₹", decimals=0)
    pe_     = _fmt(data.get("pe_ratio"),       "x")
    pb_     = _fmt(r.get("pb_ratio"),          "x", decimals=2)
    dy_     = _fmt(data.get("dividend_yield"), "%", decimals=2)
    roce_   = _fmt(_lv(r.get("roce", [])),     "%")

    hdr = Table([[
        Paragraph(f"<b>{company}</b>", title_s),
        Paragraph(f"NSE: {ticker}  |  CMP: {price}  |  Mkt Cap: {mktcap}<br/>"
                  f"P/E: {pe_}  |  P/B: {pb_}  |  Div Yield: {dy_}  |  ROCE: {roce_}", sub_s),
        Paragraph("<b>EQUITY RESEARCH</b>", er_s),
    ]], colWidths=[80*mm, 85*mm, 47*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), DARK_BLUE),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(0,-1),  8),
        ("RIGHTPADDING",(-1,0),(-1,-1),8),
        ("TOPPADDING", (0,0),(-1,-1), 9),
        ("BOTTOMPADDING",(0,0),(-1,-1),9),
    ]))
    story.append(hdr); story.append(Spacer(1, 6))

    # ── 5-YEAR FINANCIAL SUMMARY ──
    story.append(Paragraph("5-Year Financial Summary (₹ Cr)", sec_s))
    fin_data = [
        ["Revenue"]  + [_fmt(v, decimals=0) for v in r.get("revenue", [])],
        ["EBITDA"]   + [_fmt(v, decimals=0) for v in r.get("ebitda", [])],
        ["PAT"]      + [_fmt(v, decimals=0) for v in r.get("pat", [])],
        ["EPS (₹)"]  + [_fmt(v, decimals=2) for v in r.get("eps", [])],
        ["CFO"]      + [_fmt(v, decimals=0) for v in r.get("cfo", [])],
        ["FCF"]      + [_fmt(v, decimals=0) for v in r.get("fcf", [])],
    ]
    cw = [38*mm] + [26*mm]*5
    ft = Table([["Metric"]+yrs] + fin_data, colWidths=cw)
    ft.setStyle(TableStyle(tbl_base()))
    story.append(ft); story.append(Spacer(1, 6))

    # ── KEY RATIOS ──
    story.append(Paragraph("Key Ratio Snapshot (Latest Year)", sec_s))
    all_r = [
        ("Operating Margin",  _fmt(_lv(r.get("operating_margin",[])), "%")),
        ("Net Profit Margin", _fmt(_lv(r.get("net_margin",[])), "%")),
        ("ROE",               _fmt(_lv(r.get("roe",[])), "%")),
        ("ROA",               _fmt(_lv(r.get("roa",[])), "%")),
        ("ROCE",              _fmt(_lv(r.get("roce",[])), "%")),
        ("Current Ratio",     _fmt(_lv(r.get("current_ratio",[])), "x")),
        ("Quick Ratio",       _fmt(_lv(r.get("quick_ratio",[])), "x")),
        ("Debt / Equity",     _fmt(_lv(r.get("debt_equity",[])), "x")),
        ("Interest Coverage", _fmt(_lv(r.get("interest_cover",[])), "x")),
        ("Asset Turnover",    _fmt(_lv(r.get("asset_turnover",[])), "x")),
        ("Receivable Days",   _fmt(_lv(r.get("receivable_days",[])), " days")),
        ("P/E",               _fmt(r.get("pe_ratio"), "x")),
    ]
    ratio_rows = []
    for i in range(0, len(all_r), 2):
        l = all_r[i]; r2 = all_r[i+1] if i+1 < len(all_r) else ("","")
        ratio_rows.append([l[0], l[1], r2[0], r2[1]])

    rt = Table(ratio_rows, colWidths=[52*mm, 26*mm, 52*mm, 26*mm])
    rt.setStyle(TableStyle([
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 2.8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2.8),
        ("LEFTPADDING",   (0,0),(0,-1),  5),
        ("LEFTPADDING",   (2,0),(2,-1),  8),
        ("FONTNAME",      (1,0),(1,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (3,0),(3,-1),  "Helvetica-Bold"),
        ("ALIGN",         (1,0),(1,-1),  "RIGHT"),
        ("ALIGN",         (3,0),(3,-1),  "RIGHT"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT_GREY, WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, colors.HexColor("#C0C0C0")),
    ]))
    story.append(rt); story.append(Spacer(1, 6))

    # ── RED FLAGS ──
    story.append(Paragraph("Red Flag Analysis", sec_s))
    triggered = [f for f in flags if f["triggered"]]
    ok_flags  = [f for f in flags if not f["triggered"]]
    for f in triggered:
        story.append(Paragraph(f"⚠ {f['check']} — {f['detail']}", red_s))
    if not triggered:
        story.append(Paragraph("✓ No major red flags detected", green_s))
    story.append(Spacer(1, 5))

    # ── TREND CHART ──
    story.append(Paragraph("5-Year Financial Trend", sec_s))
    story.append(Image(_make_trend_chart(r, ticker), width=165*mm, height=58*mm))

    # ── PAGE 2 ──
    story.append(PageBreak())
    p2h = Table([[
        Paragraph(f"<b>{company}  |  NSE: {ticker}</b>", title_s),
        Paragraph("<b>EQUITY RESEARCH — Page 2</b>", er_s),
    ]], colWidths=[130*mm, 55*mm])
    p2h.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), DARK_BLUE),
        ("VALIGN",    (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(0,-1), 8),
        ("RIGHTPADDING",(-1,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),9),
        ("BOTTOMPADDING",(0,0),(-1,-1),9),
    ]))
    story.append(p2h); story.append(Spacer(1, 7))

    # ── DUPONT TABLE ──
    story.append(Paragraph("DuPont ROE Decomposition (3-Step)", sec_s))
    du_rows = [
        ["Net Profit Margin (%)"] + [_fmt(v) for v in r.get("dupont_net_margin", [])],
        ["Asset Turnover (×)"]   + [_fmt(v, decimals=2) for v in r.get("dupont_asset_turnover", [])],
        ["Equity Multiplier (×)"]+ [_fmt(v, decimals=2) for v in r.get("dupont_equity_mult", [])],
        ["DuPont ROE (%)"]       + [_fmt(v) for v in r.get("dupont_roe", [])],
    ]
    cw_du = [55*mm] + [26*mm]*5
    du_t = Table([["Component"]+yrs] + du_rows, colWidths=cw_du)
    du_style = tbl_base()
    du_style += [("FONTNAME",(0,4),(-1,4),"Helvetica-Bold"),
                  ("BACKGROUND",(0,4),(-1,4),LIGHT_BLUE),
                  ("TEXTCOLOR",(0,4),(-1,4),DARK_BLUE)]
    du_t.setStyle(TableStyle(du_style))
    story.append(du_t); story.append(Spacer(1, 6))

    # ── DUPONT CHART ──
    story.append(Paragraph("DuPont ROE Trend", sec_s))
    story.append(Image(_make_dupont_chart(r, ticker), width=165*mm, height=55*mm))
    story.append(Spacer(1, 7))

    # ── DCF SUMMARY (if available) ──
    if dcf_result:
        story.append(Paragraph("DCF Valuation Summary", sec_s))
        base_iv  = dcf_result.get("base_iv", np.nan)
        bear_iv  = dcf_result["scenarios"]["Bear"]["intrinsic_per_share"]
        bull_iv  = dcf_result["scenarios"]["Bull"]["intrinsic_per_share"]
        signal   = dcf_result.get("signal", "N/A")
        upside   = dcf_result.get("upside_pct", np.nan)
        wacc_val = dcf_result["wacc_result"]["wacc"] * 100

        dcf_rows = [
            ["Bear Case IV",  _fmt(bear_iv, prefix="₹", decimals=0)],
            ["Base Case IV",  _fmt(base_iv, prefix="₹", decimals=0)],
            ["Bull Case IV",  _fmt(bull_iv, prefix="₹", decimals=0)],
            ["Current Price", _fmt(dcf_result.get("current_price"), prefix="₹", decimals=0)],
            ["Upside / (Downside)", _fmt(upside, "%")],
            ["WACC Used",     _fmt(wacc_val, "%")],
            ["Signal",        signal],
        ]
        dt = Table(dcf_rows, colWidths=[80*mm, 60*mm])
        ds = tbl_base()
        # Highlight signal row
        signal_idx = 6 + 1  # +1 for header (no header here)
        signal_color = GREEN if signal == "BUY" else (RED if signal == "SELL" else AMBER)
        ds += [("BACKGROUND",(1,6),(1,6), signal_color),
               ("TEXTCOLOR",(1,6),(1,6), WHITE),
               ("FONTNAME",(1,6),(1,6),"Helvetica-Bold")]
        dt.setStyle(TableStyle([
            ("FONTSIZE",      (0,0),(-1,-1), 8.5),
            ("TOPPADDING",    (0,0),(-1,-1), 3.5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3.5),
            ("LEFTPADDING",   (0,0),(0,-1),  5),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT_GREY, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#C0C0C0")),
            ("FONTNAME",      (0,6),(0,6),   "Helvetica-Bold"),
            ("BACKGROUND",    (1,6),(1,6),   signal_color),
            ("TEXTCOLOR",     (1,6),(1,6),   WHITE),
            ("FONTNAME",      (1,6),(1,6),   "Helvetica-Bold"),
        ]))
        story.append(dt); story.append(Spacer(1, 6))

    # ── PEER COMPARISON ──
    if peer_df is not None and len(peer_df) > 1:
        story.append(Paragraph("Peer Comparison", sec_s))
        show_cols = ["Company","Ticker","Net Margin %","ROE %","ROCE %","D/E","Current Ratio","P/E"]
        available = [c for c in show_cols if c in peer_df.columns]
        p_data = [available]
        for _, row in peer_df.iterrows():
            row_data = []
            for c in available:
                v = row.get(c)
                if c in ["Company","Ticker"]:
                    row_data.append(str(v)[:22] if v else "—")
                else:
                    try:
                        row_data.append(_fmt(float(v), decimals=1))
                    except:
                        row_data.append("—")
            p_data.append(row_data)

        n_cols = len(available)
        first_w = 45*mm
        other_w = (165*mm - first_w) / max(n_cols - 1, 1)
        cw_p = [first_w] + [other_w] * (n_cols - 1)
        pt = Table(p_data, colWidths=cw_p)
        ps = tbl_base()
        # Highlight target row and median
        for i, (_, row) in enumerate(peer_df.iterrows()):
            if row.get("_is_target") == True:
                ps += [("BACKGROUND",(0,i+1),(-1,i+1), colors.HexColor("#D6E8FF")),
                       ("FONTNAME",(0,i+1),(-1,i+1),"Helvetica-Bold")]
            elif row.get("_is_target") is None:
                ps += [("BACKGROUND",(0,i+1),(-1,i+1), colors.HexColor("#E9F7EF")),
                       ("FONTNAME",(0,i+1),(-1,i+1),"Helvetica-Bold")]
        pt.setStyle(TableStyle(ps + [("FONTSIZE",(0,0),(-1,-1),7.5)]))
        story.append(pt)
        story.append(Spacer(1, 6))

    # ── DISCLAIMER ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Disclaimer: This report is auto-generated for educational purposes only. "
        "Not investment advice. Data from Screener.in. All figures ₹ Crore unless stated.",
        small_s))

    doc.build(story)
    return buf.getvalue()
