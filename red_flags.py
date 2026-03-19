"""
red_flags.py — Detects 10 financial red flags from ratio data.
Each flag returns a dict with name, detail, severity (warning/ok), and description.
"""

import numpy as np


def lv(lst, n=1):
    """Get last n valid (non-NaN) values from a list."""
    if not isinstance(lst, list):
        return [np.nan] * n
    vals = []
    for x in lst:
        try:
            f = float(x)
            if not np.isnan(f):
                vals.append(f)
        except:
            pass
    return vals[-n:] if len(vals) >= n else vals + [np.nan] * (n - len(vals))


def growth(series):
    """Calculate growth from first to last valid value."""
    vals = [v for v in series if v is not None and not np.isnan(float(v))]
    if len(vals) < 2 or vals[0] == 0:
        return np.nan
    return (vals[-1] - vals[0]) / abs(vals[0])


def detect_red_flags(r: dict) -> list:
    """
    Run all 10 red flag checks. Returns list of dicts:
    {name, triggered, severity, detail, description}
    """
    flags = []

    # ── 1. Declining CFO despite rising PAT ──
    cfo3 = lv(r.get("cfo", []), 3)
    pat3 = lv(r.get("pat", []), 3)
    cfo_declining = len(cfo3) >= 2 and not np.isnan(cfo3[0]) and cfo3[-1] < cfo3[0]
    pat_rising    = len(pat3) >= 2 and not np.isnan(pat3[0]) and pat3[-1] > pat3[0]
    triggered_1   = cfo_declining and pat_rising
    flags.append({
        "name": "Earnings Quality",
        "check": "Declining CFO despite rising PAT",
        "triggered": triggered_1,
        "severity": "high" if triggered_1 else "ok",
        "detail": (f"CFO moved from ₹{cfo3[0]:,.0f} Cr to ₹{cfo3[-1]:,.0f} Cr "
                   f"while PAT grew from ₹{pat3[0]:,.0f} to ₹{pat3[-1]:,.0f} Cr")
                  if (not np.isnan(cfo3[0]) and not np.isnan(pat3[0])) else "Insufficient data",
        "description": "When cash from operations falls while reported profit rises, it may indicate "
                       "aggressive revenue recognition or working capital manipulation.",
    })

    # ── 2. Rising debt with falling ROCE ──
    debt3 = lv(r.get("total_debt", []), 3)
    roce3 = lv(r.get("roce", []), 3)
    debt_rising  = len(debt3) >= 2 and not np.isnan(debt3[0]) and debt3[-1] > debt3[0]
    roce_falling = len(roce3) >= 2 and not np.isnan(roce3[0]) and roce3[-1] < roce3[0]
    triggered_2  = debt_rising and roce_falling
    flags.append({
        "name": "Capital Allocation",
        "check": "Rising debt with falling ROCE",
        "triggered": triggered_2,
        "severity": "high" if triggered_2 else "ok",
        "detail": (f"Debt grew {((debt3[-1]-debt3[0])/abs(debt3[0])*100):.0f}% while "
                   f"ROCE fell from {roce3[0]:.1f}% to {roce3[-1]:.1f}%")
                  if (not np.isnan(debt3[0]) and not np.isnan(roce3[0])) else "Insufficient data",
        "description": "Borrowing more while generating lower returns on capital suggests inefficient "
                       "capital deployment or deteriorating business quality.",
    })

    # ── 3. Receivables growing faster than revenue ──
    rec3 = lv(r.get("receivables", []), 3)
    rev3 = lv(r.get("revenue", []), 3)
    rec_g = growth(rec3)
    rev_g = growth(rev3)
    triggered_3 = (not np.isnan(rec_g) and not np.isnan(rev_g) and rec_g > rev_g + 0.15)
    flags.append({
        "name": "Collection Risk",
        "check": "Receivables growing faster than revenue",
        "triggered": triggered_3,
        "severity": "medium" if triggered_3 else "ok",
        "detail": (f"Receivables grew {rec_g*100:.0f}% vs revenue growth of {rev_g*100:.0f}%")
                  if not (np.isnan(rec_g) or np.isnan(rev_g)) else "Insufficient data",
        "description": "When receivables grow faster than sales, the company may be booking revenue "
                       "it hasn't collected — a classic sign of channel stuffing or bad debts.",
    })

    # ── 4. Negative FCF for 3 consecutive years ──
    fcf3 = lv(r.get("fcf", []), 3)
    triggered_4 = len(fcf3) >= 3 and all(not np.isnan(v) and v < 0 for v in fcf3)
    flags.append({
        "name": "Free Cash Flow",
        "check": "Negative FCF for 3 consecutive years",
        "triggered": triggered_4,
        "severity": "high" if triggered_4 else "ok",
        "detail": f"FCF: {', '.join([f'₹{v:,.0f} Cr' for v in fcf3 if not np.isnan(v)])}",
        "description": "Sustained negative free cash flow means the company consistently spends more "
                       "than it generates. It must rely on debt or equity dilution to fund operations.",
    })

    # ── 5. Shrinking operating margins ──
    opm3 = lv(r.get("operating_margin", []), 3)
    triggered_5 = (len(opm3) >= 2 and not np.isnan(opm3[0]) and
                   not np.isnan(opm3[-1]) and opm3[-1] < opm3[0] - 3)
    flags.append({
        "name": "Margin Erosion",
        "check": "Shrinking operating margins (>3% decline)",
        "triggered": triggered_5,
        "severity": "medium" if triggered_5 else "ok",
        "detail": (f"Operating margin fell from {opm3[0]:.1f}% to {opm3[-1]:.1f}% "
                   f"({opm3[-1]-opm3[0]:.1f}pp decline)")
                  if (not np.isnan(opm3[0]) and not np.isnan(opm3[-1])) else "Insufficient data",
        "description": "Declining operating margins indicate pricing pressure, rising input costs, "
                       "or loss of competitive advantage.",
    })

    # ── 6. High leverage (D/E > 2x) ──
    de1 = lv(r.get("debt_equity", []), 1)
    triggered_6 = bool(de1 and not np.isnan(de1[0]) and de1[0] > 2.0)
    flags.append({
        "name": "Leverage Risk",
        "check": "High Debt/Equity ratio (>2x)",
        "triggered": triggered_6,
        "severity": "high" if triggered_6 else "ok",
        "detail": f"D/E = {de1[0]:.2f}x" if (de1 and not np.isnan(de1[0])) else "No data",
        "description": "High leverage amplifies risk. In a downturn, heavily indebted companies "
                       "face cash flow stress, covenant breaches, and potential insolvency.",
    })

    # ── 7. Low interest coverage (<2x) ──
    ic1 = lv(r.get("interest_cover", []), 1)
    triggered_7 = bool(ic1 and not np.isnan(ic1[0]) and 0 < ic1[0] < 2.0)
    flags.append({
        "name": "Debt Servicing",
        "check": "Low interest coverage (<2x)",
        "triggered": triggered_7,
        "severity": "high" if triggered_7 else "ok",
        "detail": f"Interest coverage = {ic1[0]:.1f}x" if (ic1 and not np.isnan(ic1[0])) else "No data",
        "description": "Interest coverage below 2x means operating profits barely cover interest "
                       "payments, leaving little buffer for business downturns.",
    })

    # ── 8. ROE declining trend ──
    roe3 = lv(r.get("roe", []), 3)
    triggered_8 = (len(roe3) >= 2 and not np.isnan(roe3[0]) and
                   not np.isnan(roe3[-1]) and roe3[-1] < roe3[0] - 5)
    flags.append({
        "name": "Return Dilution",
        "check": "ROE declining trend (>5pp fall)",
        "triggered": triggered_8,
        "severity": "medium" if triggered_8 else "ok",
        "detail": (f"ROE declined from {roe3[0]:.1f}% to {roe3[-1]:.1f}%")
                  if (not np.isnan(roe3[0]) and not np.isnan(roe3[-1])) else "Insufficient data",
        "description": "Declining ROE can mean the company is retaining profits but deploying them "
                       "at lower returns — value destruction for shareholders.",
    })

    # ── 9. Current ratio below 1 ──
    cr1 = lv(r.get("current_ratio", []), 1)
    triggered_9 = bool(cr1 and not np.isnan(cr1[0]) and 0 < cr1[0] < 1.0)
    flags.append({
        "name": "Liquidity Stress",
        "check": "Current ratio below 1.0",
        "triggered": triggered_9,
        "severity": "high" if triggered_9 else "ok",
        "detail": f"Current ratio = {cr1[0]:.2f}x" if (cr1 and not np.isnan(cr1[0])) else "No data",
        "description": "Current liabilities exceed current assets. The company may struggle to meet "
                       "short-term obligations without additional financing.",
    })

    # ── 10. Revenue contraction ──
    rev5 = lv(r.get("revenue", []), 5)
    triggered_10 = (len(rev5) >= 3 and
                    not np.isnan(rev5[0]) and not np.isnan(rev5[-1]) and
                    rev5[-1] < rev5[0])
    flags.append({
        "name": "Revenue Contraction",
        "check": "Revenue declined over last 3 years",
        "triggered": triggered_10,
        "severity": "medium" if triggered_10 else "ok",
        "detail": (f"Revenue fell from ₹{rev5[0]:,.0f} Cr to ₹{rev5[-1]:,.0f} Cr "
                   f"({((rev5[-1]-rev5[0])/rev5[0]*100):.1f}%)")
                  if (not np.isnan(rev5[0]) and not np.isnan(rev5[-1])) else "Insufficient data",
        "description": "Shrinking revenue indicates loss of market share, pricing power, or demand "
                       "weakness. Sustained contraction is a serious concern.",
    })

    return flags


def flags_summary(flags: list) -> dict:
    """Return summary counts by severity."""
    high    = sum(1 for f in flags if f["triggered"] and f["severity"] == "high")
    medium  = sum(1 for f in flags if f["triggered"] and f["severity"] == "medium")
    ok      = sum(1 for f in flags if not f["triggered"])
    return {"high": high, "medium": medium, "ok": ok, "total": len(flags)}
