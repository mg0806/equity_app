"""
dcf.py — DCF Valuation Engine with:
  - 5-year FCFF projection (Bull / Base / Bear scenarios)
  - WACC calculation (CAPM + after-tax cost of debt)
  - Terminal value (Gordon Growth Model)
  - Monte Carlo simulation (10,000 iterations)
  - Sensitivity analysis (tornado chart data)
  - Buy / Hold / Sell signal with margin of safety
"""

import numpy as np
import pandas as pd
from ratios import last_valid


# ─────────────────────────────────────────────
# WACC CALCULATOR
# ─────────────────────────────────────────────

def calculate_wacc(
    market_cap: float,          # ₹ Cr
    total_debt: float,          # ₹ Cr
    beta: float = 1.0,
    risk_free_rate: float = 0.07,    # 10Y Gsec yield (~7%)
    equity_risk_premium: float = 0.055,  # India ERP
    cost_of_debt: float = 0.09,      # ~9% for Indian cos
    tax_rate: float = 0.25,
) -> dict:
    """Calculate WACC using CAPM for cost of equity."""
    # Cost of equity via CAPM
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Capital structure weights
    total_capital = market_cap + total_debt
    if total_capital == 0:
        return {"wacc": 0.11, "cost_of_equity": cost_of_equity,
                "cost_of_debt_after_tax": cost_of_debt * (1 - tax_rate),
                "equity_weight": 1.0, "debt_weight": 0.0}

    w_equity = market_cap / total_capital
    w_debt   = total_debt / total_capital
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

    wacc = (w_equity * cost_of_equity) + (w_debt * after_tax_cost_of_debt)

    return {
        "wacc": round(wacc, 4),
        "cost_of_equity": round(cost_of_equity, 4),
        "cost_of_debt_after_tax": round(after_tax_cost_of_debt, 4),
        "equity_weight": round(w_equity, 4),
        "debt_weight": round(w_debt, 4),
        "beta": beta,
        "risk_free_rate": risk_free_rate,
        "erp": equity_risk_premium,
    }


# ─────────────────────────────────────────────
# FCF PROJECTION
# ─────────────────────────────────────────────

def project_fcff(
    base_revenue: float,
    base_ebitda_margin: float,
    revenue_growth: float,
    margin_improvement: float,     # annual change in EBITDA margin
    capex_pct_revenue: float,
    wc_change_pct_revenue: float,  # working capital change as % revenue
    tax_rate: float,
    n_years: int = 5,
) -> pd.DataFrame:
    """
    Project 5-year Free Cash Flow to Firm (FCFF).
    FCFF = EBIT(1-t) + D&A - Capex - ΔWC
    Simplified: FCFF ≈ EBITDA(1-t) - Capex - ΔWC
    """
    rows = []
    revenue = base_revenue
    ebitda_margin = base_ebitda_margin

    for yr in range(1, n_years + 1):
        revenue      = revenue * (1 + revenue_growth)
        ebitda_margin = min(ebitda_margin + margin_improvement, 0.60)  # cap at 60%
        ebitda       = revenue * ebitda_margin
        nopat        = ebitda * (1 - tax_rate)
        capex        = revenue * capex_pct_revenue
        delta_wc     = revenue * wc_change_pct_revenue
        fcff         = nopat - capex - delta_wc

        rows.append({
            "Year": f"FY+{yr}",
            "Revenue": round(revenue, 1),
            "EBITDA Margin %": round(ebitda_margin * 100, 1),
            "EBITDA": round(ebitda, 1),
            "NOPAT": round(nopat, 1),
            "Capex": round(capex, 1),
            "ΔWorking Capital": round(delta_wc, 1),
            "FCFF": round(fcff, 1),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# DCF VALUATION (SINGLE SCENARIO)
# ─────────────────────────────────────────────

def dcf_valuation(
    fcff_df: pd.DataFrame,
    wacc: float,
    terminal_growth: float,
    total_debt: float,
    cash: float,
    shares_outstanding: float,   # in Cr shares
) -> dict:
    """
    Compute intrinsic value from projected FCFFs.
    Intrinsic Value = PV(FCFFs) + PV(Terminal Value) - Debt + Cash
    """
    fcffs = fcff_df["FCFF"].tolist()
    n     = len(fcffs)

    # PV of each year's FCFF
    pv_fcffs = [fcff / ((1 + wacc) ** (i + 1)) for i, fcff in enumerate(fcffs)]
    sum_pv_fcff = sum(pv_fcffs)

    # Terminal value (Gordon Growth)
    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.01  # safety constraint
    terminal_value    = terminal_fcff / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)

    # Enterprise value → Equity value
    enterprise_value  = sum_pv_fcff + pv_terminal_value
    equity_value      = enterprise_value - total_debt + cash
    equity_value      = max(equity_value, 0)

    intrinsic_per_share = (equity_value / shares_outstanding) if shares_outstanding > 0 else np.nan

    return {
        "pv_fcffs": [round(v, 1) for v in pv_fcffs],
        "sum_pv_fcff": round(sum_pv_fcff, 1),
        "terminal_value": round(terminal_value, 1),
        "pv_terminal_value": round(pv_terminal_value, 1),
        "enterprise_value": round(enterprise_value, 1),
        "equity_value": round(equity_value, 1),
        "intrinsic_per_share": round(intrinsic_per_share, 2) if not np.isnan(intrinsic_per_share) else np.nan,
        "tv_pct_of_ev": round(pv_terminal_value / enterprise_value * 100, 1) if enterprise_value else np.nan,
    }


# ─────────────────────────────────────────────
# THREE SCENARIOS (BULL / BASE / BEAR)
# ─────────────────────────────────────────────

def run_three_scenarios(data: dict, r: dict, assumptions: dict) -> dict:
    """
    Run DCF for Bull, Base, and Bear scenarios.
    Returns scenario results + buy/hold/sell signal.
    """
    base_revenue       = last_valid(r.get("revenue", []))
    base_ebitda_margin = last_valid(r.get("operating_margin", [])) / 100
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt = 0 if (total_debt is None or (isinstance(total_debt, float) and np.isnan(total_debt))) else float(total_debt)
    cash_val           = last_valid(r.get("cash", []))
    cash_val = 0 if (cash_val is None or (isinstance(cash_val, float) and np.isnan(cash_val))) else cash_val
    market_cap         = data.get("market_cap") or np.nan
    current_price      = data.get("current_price") or np.nan

    # Approximate shares outstanding from market cap and price
    shares = (market_cap / current_price) if (market_cap and current_price) else 100.0

    wacc_result = calculate_wacc(
        market_cap=market_cap or 0,
        total_debt=total_debt or 0,
        beta=assumptions.get("beta", 1.0),
        risk_free_rate=assumptions.get("risk_free_rate", 0.07),
        equity_risk_premium=assumptions.get("erp", 0.055),
        cost_of_debt=assumptions.get("cost_of_debt", 0.09),
        tax_rate=assumptions.get("tax_rate", 0.25),
    )
    wacc = wacc_result["wacc"]

    scenarios = {
        "Bear": {
            "revenue_growth":         assumptions.get("bear_growth", 0.05),
            "margin_improvement":     assumptions.get("bear_margin_delta", -0.005),
            "capex_pct_revenue":      assumptions.get("capex_pct", 0.08),
            "wc_change_pct_revenue":  assumptions.get("wc_pct", 0.02),
            "terminal_growth":        assumptions.get("bear_tgr", 0.03),
        },
        "Base": {
            "revenue_growth":         assumptions.get("base_growth", 0.12),
            "margin_improvement":     assumptions.get("base_margin_delta", 0.002),
            "capex_pct_revenue":      assumptions.get("capex_pct", 0.07),
            "wc_change_pct_revenue":  assumptions.get("wc_pct", 0.015),
            "terminal_growth":        assumptions.get("base_tgr", 0.04),
        },
        "Bull": {
            "revenue_growth":         assumptions.get("bull_growth", 0.20),
            "margin_improvement":     assumptions.get("bull_margin_delta", 0.008),
            "capex_pct_revenue":      assumptions.get("capex_pct", 0.06),
            "wc_change_pct_revenue":  assumptions.get("wc_pct", 0.01),
            "terminal_growth":        assumptions.get("bull_tgr", 0.05),
        },
    }

    results = {}
    for name, s in scenarios.items():
        fcff_df = project_fcff(
            base_revenue=base_revenue or 10000,
            base_ebitda_margin=base_ebitda_margin or 0.15,
            revenue_growth=s["revenue_growth"],
            margin_improvement=s["margin_improvement"],
            capex_pct_revenue=s["capex_pct_revenue"],
            wc_change_pct_revenue=s["wc_change_pct_revenue"],
            tax_rate=assumptions.get("tax_rate", 0.25),
        )
        val = dcf_valuation(
            fcff_df=fcff_df,
            wacc=wacc,
            terminal_growth=s["terminal_growth"],
            total_debt=total_debt or 0,
            cash=cash_val or 0,
            shares_outstanding=shares,
        )
        results[name] = {**val, "fcff_df": fcff_df, "assumptions": s}

    # Signal based on base case
    base_iv = results["Base"]["intrinsic_per_share"]
    if not np.isnan(base_iv) and not np.isnan(current_price) and current_price > 0:
        upside = (base_iv - current_price) / current_price * 100
        mos    = upside  # margin of safety = upside %
        if upside > 20:
            signal = "BUY"
            signal_color = "green"
        elif upside > -10:
            signal = "HOLD"
            signal_color = "orange"
        else:
            signal = "SELL"
            signal_color = "red"
    else:
        upside = np.nan
        mos    = np.nan
        signal = "N/A"
        signal_color = "grey"

    return {
        "scenarios": results,
        "wacc_result": wacc_result,
        "current_price": current_price,
        "base_iv": base_iv,
        "upside_pct": round(upside, 1) if not np.isnan(upside) else np.nan,
        "margin_of_safety": round(mos, 1) if not np.isnan(mos) else np.nan,
        "signal": signal,
        "signal_color": signal_color,
        "shares": shares,
    }


# ─────────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def run_monte_carlo(data: dict, r: dict, assumptions: dict, n_simulations: int = 10000) -> dict:
    """
    Run Monte Carlo DCF simulation with randomized key inputs.
    Returns distribution of intrinsic values.
    """
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = (last_valid(r.get("operating_margin", [])) or 15) / 100
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt = 0 if (total_debt is None or (isinstance(total_debt, float) and np.isnan(total_debt))) else float(total_debt) or 0
    cash_val           = last_valid(r.get("cash", []))
    cash_val = 0 if (cash_val is None or (isinstance(cash_val, float) and np.isnan(cash_val))) else cash_val or 0
    market_cap         = data.get("market_cap") or 100000
    current_price      = data.get("current_price") or 1000
    shares             = (market_cap / current_price) if current_price else 100

    base_growth  = assumptions.get("base_growth", 0.12)
    base_tgr     = assumptions.get("base_tgr", 0.04)
    base_wacc    = assumptions.get("base_wacc", 0.11)
    capex_pct    = assumptions.get("capex_pct", 0.07)
    tax_rate     = assumptions.get("tax_rate", 0.25)

    np.random.seed(42)
    intrinsic_values = []

    for _ in range(n_simulations):
        # Randomize inputs with normal distribution (mean=base, std=range/3)
        rev_growth  = np.random.normal(base_growth, 0.05)
        rev_growth  = np.clip(rev_growth, -0.10, 0.40)

        tgr         = np.random.normal(base_tgr, 0.01)
        tgr         = np.clip(tgr, 0.01, 0.06)

        wacc        = np.random.normal(base_wacc, 0.015)
        wacc        = np.clip(wacc, 0.06, 0.20)

        margin_delta= np.random.normal(0.002, 0.003)
        cap_pct     = np.random.normal(capex_pct, 0.01)
        cap_pct     = np.clip(cap_pct, 0.02, 0.20)

        try:
            fcff_df = project_fcff(
                base_revenue=base_revenue,
                base_ebitda_margin=base_ebitda_margin,
                revenue_growth=rev_growth,
                margin_improvement=margin_delta,
                capex_pct_revenue=cap_pct,
                wc_change_pct_revenue=0.015,
                tax_rate=tax_rate,
            )
            val = dcf_valuation(
                fcff_df=fcff_df,
                wacc=wacc,
                terminal_growth=tgr,
                total_debt=total_debt,
                cash=cash_val,
                shares_outstanding=shares,
            )
            iv = val["intrinsic_per_share"]
            if iv and not np.isnan(iv) and iv > 0:
                intrinsic_values.append(iv)
        except:
            pass

    if not intrinsic_values:
        return {"error": "Monte Carlo failed — insufficient data"}

    arr = np.array(intrinsic_values)
    return {
        "values": arr.tolist(),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p10": round(float(np.percentile(arr, 10)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "std": round(float(arr.std()), 2),
        "n_simulations": len(arr),
        "current_price": current_price,
        "prob_undervalued": round(float((arr > current_price).mean() * 100), 1),
    }


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS (TORNADO CHART DATA)
# ─────────────────────────────────────────────

def run_sensitivity(data: dict, r: dict, assumptions: dict) -> pd.DataFrame:
    """
    Vary each key assumption ±1 std dev, measure impact on base IV.
    Returns DataFrame for tornado chart.
    """
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = (last_valid(r.get("operating_margin", [])) or 15) / 100
    total_debt         = last_valid(r.get("total_debt", []))
    total_debt = 0 if (total_debt is None or (isinstance(total_debt, float) and np.isnan(total_debt))) else float(total_debt) or 0
    cash_val           = last_valid(r.get("cash", []))
    cash_val = 0 if (cash_val is None or (isinstance(cash_val, float) and np.isnan(cash_val))) else cash_val or 0
    market_cap         = data.get("market_cap") or 100000
    current_price      = data.get("current_price") or 1000
    shares             = (market_cap / current_price) if current_price else 100

    base_wacc   = assumptions.get("base_wacc", 0.11)
    base_growth = assumptions.get("base_growth", 0.12)
    base_tgr    = assumptions.get("base_tgr", 0.04)
    capex_pct   = assumptions.get("capex_pct", 0.07)
    tax_rate    = assumptions.get("tax_rate", 0.25)

    def compute_iv(rev_growth, wacc, tgr, cap_pct, margin_delta=0.002):
        try:
            fcff_df = project_fcff(
                base_revenue=base_revenue,
                base_ebitda_margin=base_ebitda_margin,
                revenue_growth=rev_growth,
                margin_improvement=margin_delta,
                capex_pct_revenue=cap_pct,
                wc_change_pct_revenue=0.015,
                tax_rate=tax_rate,
            )
            val = dcf_valuation(fcff_df, wacc, tgr, total_debt, cash_val, shares)
            return val["intrinsic_per_share"] or np.nan
        except:
            return np.nan

    # Base case
    base_iv = compute_iv(base_growth, base_wacc, base_tgr, capex_pct)

    # Each variable's range
    variables = [
        ("Revenue Growth",    base_growth,  0.05,  lambda v: compute_iv(v, base_wacc, base_tgr, capex_pct)),
        ("WACC",              base_wacc,    0.02,  lambda v: compute_iv(base_growth, v, base_tgr, capex_pct)),
        ("Terminal Growth",   base_tgr,     0.01,  lambda v: compute_iv(base_growth, base_wacc, v, capex_pct)),
        ("Capex % Revenue",   capex_pct,    0.02,  lambda v: compute_iv(base_growth, base_wacc, base_tgr, v)),
        ("Tax Rate",          tax_rate,     0.05,  lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct)),
        ("EBITDA Margin Δ",   0.002,        0.005, lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct, v)),
    ]

    rows = []
    for name, base_val, delta, fn in variables:
        low_iv  = fn(base_val - delta)
        high_iv = fn(base_val + delta)
        impact  = abs((high_iv or base_iv) - (low_iv or base_iv))
        rows.append({
            "Variable":  name,
            "Base IV":   round(base_iv, 2) if not np.isnan(base_iv) else np.nan,
            "Low IV":    round(low_iv,  2) if not np.isnan(low_iv)  else np.nan,
            "High IV":   round(high_iv, 2) if not np.isnan(high_iv) else np.nan,
            "Impact":    round(impact, 2),
            "Low Label": f"{name} - {delta*100:.0f}%",
            "High Label":f"{name} + {delta*100:.0f}%",
        })

    df = pd.DataFrame(rows).sort_values("Impact", ascending=True)
    return df
