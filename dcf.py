"""
dcf.py — Enhanced DCF Valuation Engine with:
  - Business type classification (stable / cyclical / high-margin-stable / commodity)
  - Business quality scoring (0-100) across 4 dimensions
  - Context-aware assumption derivation (business-type + quality adjusted)
  - Dynamic margin-of-safety requirements based on quality
  - Enhanced composite signal with quality gating
  - 5-year FCFF projection (Bull / Base / Bear scenarios)
  - WACC calculation (CAPM + after-tax cost of debt, actual D/E)
  - Terminal value (Gordon Growth Model with sanity checks)
  - Monte Carlo simulation (10,000 iterations)
  - Sensitivity analysis (tornado chart data)
"""

import numpy as np
import pandas as pd
from ratios import last_valid


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _valid_floats(lst):
    """Return list of finite floats from a mixed/None list."""
    out = []
    for v in lst:
        try:
            f = float(v)
            if not np.isnan(f):
                out.append(f)
        except Exception:
            pass
    return out


def _safe_float(val, default=np.nan):
    """Safely cast a value to float, returning default on failure."""
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


# ─────────────────────────────────────────────
# BUSINESS TYPE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_business_type(r: dict, data: dict) -> str:
    """
    Classify business type using Screener.in financial metrics.

    Classification logic:
      - 'high-margin-stable'  : OPM >= 20% AND revenue volatility low AND positive FCF
      - 'stable'              : OPM >= 10% AND moderate volatility
      - 'cyclical'            : High revenue volatility (std/mean > 20%) OR thin margins with swing
      - 'commodity'           : OPM < 8% AND high revenue volatility OR negative FCF trend

    Returns
    -------
    str : One of 'stable', 'cyclical', 'high-margin-stable', 'commodity'
    """
    operating_margins = _valid_floats(r.get("operating_margin", []))
    revenues          = _valid_floats(r.get("revenue", []))
    fcf_vals          = _valid_floats(r.get("fcf", []))

    # Median operating margin
    median_opm = float(np.median(operating_margins)) if operating_margins else 10.0

    # Revenue volatility: coefficient of variation (std / mean)
    if len(revenues) >= 3:
        yoy_growths = [(revenues[i + 1] - revenues[i]) / abs(revenues[i])
                       for i in range(len(revenues) - 1) if revenues[i] != 0]
        rev_volatility = float(np.std(yoy_growths)) if yoy_growths else 0.10
    else:
        rev_volatility = 0.10  # default moderate volatility

    # FCF quality: ratio of positive FCF years
    fcf_positive_ratio = (sum(1 for f in fcf_vals if f > 0) / len(fcf_vals)
                          if fcf_vals else 0.5)

    # Classification rules (order matters — most specific first)
    if median_opm >= 18.0 and rev_volatility < 0.15 and fcf_positive_ratio >= 0.6:
        return "high-margin-stable"

    if median_opm < 8.0 and (rev_volatility > 0.20 or fcf_positive_ratio < 0.4):
        return "commodity"

    if rev_volatility > 0.20 or (median_opm < 10.0 and rev_volatility > 0.15):
        return "cyclical"

    return "stable"


# ─────────────────────────────────────────────
# BUSINESS QUALITY SCORING
# ─────────────────────────────────────────────

def calculate_business_quality(r: dict, peer_df=None) -> tuple:
    """
    Calculate business quality score (0-100) from Screener.in data.

    Components
    ----------
    Financial health (30%):
        - Debt/Equity trend (lower = better)
        - Interest coverage (higher = better)
        - Current ratio stability
    Profitability (30%):
        - Median ROCE / ROE levels
        - Operating margin stability (low std = better)
        - Margin trend direction
    Cash flow quality (20%):
        - CFO / PAT ratio (closer to 1 = better)
        - FCF conversion (FCF / EBITDA)
        - Ratio of positive FCF years
    Growth quality (20%):
        - Revenue growth consistency (CAGR vs volatility)
        - EPS CAGR vs revenue CAGR (margin on growth)
        - Earnings quality (PAT CAGR consistency)

    Returns
    -------
    quality_score      : float  0-100
    quality_components : dict   breakdown by component
    """

    # ── 1. Financial Health (30 pts) ──────────────────
    health_score = 0.0
    health_detail = {}

    de_vals      = _valid_floats(r.get("debt_equity", []))
    int_cov_vals = _valid_floats(r.get("interest_cover", []))
    curr_ratio   = _valid_floats(r.get("current_ratio", []))

    # Debt/Equity: score 10 pts (D/E = 0 → 10, D/E ≥ 2 → 0)
    if de_vals:
        latest_de = de_vals[-1]
        de_score  = float(np.clip(10 * (1 - latest_de / 2.0), 0, 10))
        # Bonus 2 pts if D/E is improving (declining trend)
        if len(de_vals) >= 3 and de_vals[-1] < de_vals[0]:
            de_score = min(de_score + 2, 10)
    else:
        de_score = 5.0  # neutral when unknown
    health_detail["debt_equity"] = round(de_score, 1)
    health_score += de_score

    # Interest coverage: score 10 pts (IC ≥ 5 → 10, IC ≤ 1 → 0)
    if int_cov_vals:
        latest_ic = int_cov_vals[-1]
        ic_score  = float(np.clip(10 * (latest_ic - 1) / 4.0, 0, 10))
    else:
        ic_score  = 5.0  # neutral
    health_detail["interest_coverage"] = round(ic_score, 1)
    health_score += ic_score

    # Current ratio: score 10 pts (CR ≥ 2 → 10, CR ≤ 0.5 → 0)
    if curr_ratio:
        latest_cr = curr_ratio[-1]
        cr_score  = float(np.clip(10 * (latest_cr - 0.5) / 1.5, 0, 10))
    else:
        cr_score  = 5.0
    health_detail["current_ratio"] = round(cr_score, 1)
    health_score += cr_score

    # ── 2. Profitability (30 pts) ─────────────────────
    prof_score  = 0.0
    prof_detail = {}

    roce_vals  = _valid_floats(r.get("roce", []))
    roe_vals   = _valid_floats(r.get("roe", []))
    opm_vals   = _valid_floats(r.get("operating_margin", []))

    # ROCE: 10 pts (ROCE ≥ 20% → 10, ROCE ≤ 5% → 0)
    if roce_vals:
        med_roce  = float(np.median(roce_vals))
        roce_sc   = float(np.clip(10 * (med_roce - 5) / 15.0, 0, 10))
    else:
        roce_sc   = 4.0
    prof_detail["roce"] = round(roce_sc, 1)
    prof_score += roce_sc

    # Operating margin stability: 10 pts (std < 2% → 10, std > 8% → 0)
    if len(opm_vals) >= 3:
        opm_std = float(np.std(opm_vals))
        opm_stab_sc = float(np.clip(10 * (1 - opm_std / 8.0), 0, 10))
    else:
        opm_stab_sc = 4.0
    prof_detail["margin_stability"] = round(opm_stab_sc, 1)
    prof_score += opm_stab_sc

    # Margin trend: 10 pts (improving = positive delta)
    if len(opm_vals) >= 3:
        opm_delta   = opm_vals[-1] - opm_vals[0]
        margin_tr_sc = float(np.clip(5 + opm_delta, 0, 10))  # neutral at 5
    else:
        margin_tr_sc = 5.0
    prof_detail["margin_trend"] = round(margin_tr_sc, 1)
    prof_score += margin_tr_sc

    # ── 3. Cash Flow Quality (20 pts) ─────────────────
    cf_score  = 0.0
    cf_detail = {}

    cfo_vals  = _valid_floats(r.get("cfo", []))
    pat_vals  = _valid_floats(r.get("pat", []))
    fcf_vals  = _valid_floats(r.get("fcf", []))
    ebitda_v  = _valid_floats(r.get("ebitda", []))

    # CFO / PAT ratio: 8 pts (ratio ~1 → 8, ratio < 0 → 0)
    cfo_pat_ratios = []
    for cf_v, pt_v in zip(cfo_vals, pat_vals):
        if pt_v and pt_v != 0 and not np.isnan(pt_v):
            cfo_pat_ratios.append(cf_v / pt_v)
    if cfo_pat_ratios:
        med_cfo_pat = float(np.median(cfo_pat_ratios))
        cfo_pat_sc  = float(np.clip(8 * min(med_cfo_pat, 1.5) / 1.5, 0, 8))
    else:
        cfo_pat_sc  = 4.0
    cf_detail["cfo_pat_ratio"] = round(cfo_pat_sc, 1)
    cf_score += cfo_pat_sc

    # FCF / EBITDA conversion: 7 pts
    fcf_ebitda_ratios = []
    for fv, ev in zip(fcf_vals, ebitda_v):
        if ev and ev > 0:
            fcf_ebitda_ratios.append(fv / ev)
    if fcf_ebitda_ratios:
        med_fcf_ebitda = float(np.median(fcf_ebitda_ratios))
        fcf_eb_sc      = float(np.clip(7 * med_fcf_ebitda, 0, 7))
    else:
        fcf_eb_sc = 3.5
    cf_detail["fcf_conversion"] = round(fcf_eb_sc, 1)
    cf_score += fcf_eb_sc

    # Positive FCF years: 5 pts
    if fcf_vals:
        pos_ratio  = sum(1 for f in fcf_vals if f > 0) / len(fcf_vals)
        fcf_pos_sc = float(5 * pos_ratio)
    else:
        fcf_pos_sc = 2.5
    cf_detail["positive_fcf_years"] = round(fcf_pos_sc, 1)
    cf_score += fcf_pos_sc

    # ── 4. Growth Quality (20 pts) ────────────────────
    growth_score  = 0.0
    growth_detail = {}

    cagr3_rev = _safe_float(r.get("revenue_cagr_3y"), np.nan)
    cagr5_rev = _safe_float(r.get("revenue_cagr_5y"), np.nan)
    cagr3_eps = _safe_float(r.get("eps_cagr_3y"), np.nan)

    # Revenue growth consistency: 8 pts
    # Score based on 5Y CAGR level; penalise if 3Y << 5Y (deceleration)
    if not np.isnan(cagr5_rev):
        base_g_sc = float(np.clip(8 * cagr5_rev / 20.0, 0, 8))  # 20% → full score
        if not np.isnan(cagr3_rev) and cagr3_rev < cagr5_rev * 0.5:
            base_g_sc = max(base_g_sc - 2, 0)  # deceleration penalty
    else:
        base_g_sc = 4.0
    growth_detail["revenue_growth"] = round(base_g_sc, 1)
    growth_score += base_g_sc

    # EPS CAGR vs Revenue CAGR (margin on growth): 7 pts
    # EPS should grow at least as fast as revenue for quality growth
    if not np.isnan(cagr3_eps) and not np.isnan(cagr3_rev) and cagr3_rev > 0:
        eps_rev_ratio = cagr3_eps / cagr3_rev
        mg_sc = float(np.clip(7 * min(eps_rev_ratio, 2.0) / 2.0, 0, 7))
    else:
        mg_sc = 3.5
    growth_detail["margin_on_growth"] = round(mg_sc, 1)
    growth_score += mg_sc

    # PAT CAGR consistency (PAT 3Y similar to 5Y): 5 pts
    cagr3_pat = _safe_float(r.get("pat_cagr_3y"), np.nan)
    cagr5_pat = _safe_float(r.get("pat_cagr_5y"), np.nan)
    if not np.isnan(cagr3_pat) and not np.isnan(cagr5_pat) and cagr5_pat != 0:
        consistency = 1 - abs(cagr3_pat - cagr5_pat) / abs(cagr5_pat)
        consist_sc  = float(np.clip(5 * consistency, 0, 5))
    else:
        consist_sc  = 2.5
    growth_detail["pat_consistency"] = round(consist_sc, 1)
    growth_score += consist_sc

    # ── Aggregate ──────────────────────────────────────
    # Weights: health 30, prof 30, cf 20, growth 20 = 100 max
    total = health_score + prof_score + cf_score + growth_score
    quality_score = float(np.clip(total, 0, 100))

    quality_components = {
        "financial_health": {
            "score":  round(health_score, 1),
            "max":    30,
            "detail": health_detail,
        },
        "profitability": {
            "score":  round(prof_score, 1),
            "max":    30,
            "detail": prof_detail,
        },
        "cash_flow_quality": {
            "score":  round(cf_score, 1),
            "max":    20,
            "detail": cf_detail,
        },
        "growth_quality": {
            "score":  round(growth_score, 1),
            "max":    20,
            "detail": growth_detail,
        },
    }

    return round(quality_score, 1), quality_components


# ─────────────────────────────────────────────
# HISTORY-DERIVED ASSUMPTION ENGINE (ENHANCED)
# ─────────────────────────────────────────────

def derive_assumptions_from_history(data: dict, r: dict, peer_df=None) -> dict:
    """
    Analyse 5 years of historical data to auto-derive DCF assumptions.

    Enhancements over v1:
      - Classifies business type before deriving growth / margin assumptions.
      - Applies business-type-specific growth ceilings and mean-reversion anchors.
      - Uses peer median margins when peer_df is available for margin benchmarking.
      - Derives effective tax rate from financials where possible.
      - Sets terminal growth as max(GDP_proxy / 2, 3%) with business-type floor.
      - Computes quality score and embeds it in the return dict for downstream use.
    """
    rev      = r.get("revenue", [])
    ebitda_m = r.get("operating_margin", [])
    capex    = r.get("capex", [])
    cfo      = r.get("cfo", [])
    debt     = r.get("total_debt", [])
    equity   = r.get("equity", [])

    n = r.get("n_years", 5)

    # ── Classify business & compute quality ──────────────
    biz_type = classify_business_type(r, data)
    quality_score, quality_components = calculate_business_quality(r, peer_df)

    # Business-type parameters
    # growth_ceiling : max allowed base_growth
    # mr_anchor      : long-run mean-reversion target (GDP + sector premium)
    # mr_weight      : how much we pull toward the anchor (0-1)
    # margin_ceiling : absolute cap on EBITDA margin assumption
    # tgr_floor      : minimum terminal growth rate
    BIZ_PARAMS = {
        "high-margin-stable": {
            "growth_ceiling": 0.30,
            "mr_anchor":      0.12,   # IT/pharma can sustain 12% long-run
            "mr_weight":      0.25,
            "margin_ceiling": 60.0,
            "tgr_floor":      0.04,
        },
        "stable": {
            "growth_ceiling": 0.25,
            "mr_anchor":      0.10,
            "mr_weight":      0.30,
            "margin_ceiling": 45.0,
            "tgr_floor":      0.03,
        },
        "cyclical": {
            "growth_ceiling": 0.20,
            "mr_anchor":      0.08,   # cyclicals mean-revert to lower GDP-linked growth
            "mr_weight":      0.40,   # stronger pull toward anchor
            "margin_ceiling": 35.0,
            "tgr_floor":      0.03,
        },
        "commodity": {
            "growth_ceiling": 0.18,
            "mr_anchor":      0.07,
            "mr_weight":      0.50,   # heavy mean-reversion for commodities
            "margin_ceiling": 20.0,
            "tgr_floor":      0.03,
        },
    }
    bp = BIZ_PARAMS[biz_type]

    # ── Revenue growth ────────────────────────────────────
    cagr3 = _safe_float(r.get("revenue_cagr_3y"))
    cagr5 = _safe_float(r.get("revenue_cagr_5y"))
    c3    = None if np.isnan(cagr3) else cagr3
    c5    = None if np.isnan(cagr5) else cagr5

    if c3 is not None and c5 is not None:
        raw_growth = (0.60 * c3 + 0.40 * c5) / 100
    elif c3 is not None:
        raw_growth = c3 / 100
    elif c5 is not None:
        raw_growth = c5 / 100
    else:
        raw_growth = 0.10

    # Business-type–adjusted mean reversion
    anchor    = bp["mr_anchor"]
    mr_weight = bp["mr_weight"]
    base_growth = (1 - mr_weight) * raw_growth + mr_weight * anchor
    base_growth = float(np.clip(base_growth, 0.03, bp["growth_ceiling"]))

    # ── EBITDA / operating margin ─────────────────────────
    valid_margins = _valid_floats(ebitda_m)

    # Industry peer median margin (if peer data available)
    peer_median_margin = None
    if peer_df is not None and "Net Margin %" in peer_df.columns:
        peer_margins = _valid_floats(peer_df["Net Margin %"].dropna().tolist())
        if peer_margins:
            peer_median_margin = float(np.median(peer_margins))

    if valid_margins:
        hist_median_margin = float(np.median(valid_margins))
    else:
        hist_median_margin = 15.0

    # Blend historical with peer median if available
    if peer_median_margin is not None:
        base_ebitda_margin_pct = 0.70 * hist_median_margin + 0.30 * peer_median_margin
    else:
        base_ebitda_margin_pct = hist_median_margin

    # Apply business-type ceiling and absolute floor
    base_ebitda_margin_pct = float(
        np.clip(base_ebitda_margin_pct, 3.0, bp["margin_ceiling"])
    )

    # Margin trend slope (pp per year) — capped tighter for cyclicals/commodities
    if len(valid_margins) >= 3:
        raw_delta = (valid_margins[-1] - valid_margins[0]) / (len(valid_margins) - 1) / 100
        if biz_type in ("cyclical", "commodity"):
            margin_delta = float(np.clip(raw_delta, -0.015, 0.010))
        else:
            margin_delta = float(np.clip(raw_delta, -0.020, 0.020))
    else:
        margin_delta = 0.001

    # ── Capex % revenue ───────────────────────────────────
    rev_vals   = _valid_floats(rev)
    capex_vals = _valid_floats(capex)
    capex_ratios = []
    for c_v, r_v in zip(capex_vals[-n:], rev_vals[-n:]):
        if r_v > 0:
            capex_ratios.append(abs(c_v) / r_v)
    capex_pct = (float(np.clip(np.median(capex_ratios), 0.02, 0.25))
                 if capex_ratios else 0.07)

    # ── Working capital % revenue ─────────────────────────
    pat_vals   = _valid_floats(r.get("pat", []))
    cfo_vals   = _valid_floats(cfo)
    wc_changes = []
    for cf_v, pt_v, rv in zip(cfo_vals[-n:], pat_vals[-n:], rev_vals[-n:]):
        if rv > 0 and pt_v != 0:
            wc_changes.append(abs(pt_v - cf_v) / rv)
    wc_pct = (float(np.clip(np.median(wc_changes), 0.005, 0.05))
              if wc_changes else 0.015)

    # ── Effective tax rate ────────────────────────────────
    # Use actual tax expense / PBT where possible
    tax_vals = _valid_floats(r.get("tax", []))
    pbt_vals = _valid_floats(r.get("pbt", []))
    eff_tax_rates = []
    for tx, pb in zip(tax_vals, pbt_vals):
        if pb > 0:
            eff_tax_rates.append(tx / pb)
    if eff_tax_rates:
        tax_rate_est = float(np.clip(np.median(eff_tax_rates), 0.15, 0.35))
    else:
        tax_rate_est = 0.25  # standard Indian corporate tax

    # ── WACC via Hamada equation ──────────────────────────
    debt_vals     = _valid_floats(debt)
    equity_vals   = _valid_floats(equity)
    latest_debt   = debt_vals[-1]   if debt_vals   else 0.0
    latest_equity = equity_vals[-1] if equity_vals else 1.0
    de_ratio = latest_debt / latest_equity if latest_equity > 0 else 0.5

    # Unlevered beta prior adjusted by business type
    UNLEV_BETA = {
        "high-margin-stable": 0.70,  # lower systematic risk (IT, pharma)
        "stable":             0.80,
        "cyclical":           1.00,  # higher systematic risk
        "commodity":          1.10,
    }
    unlevered_beta = UNLEV_BETA[biz_type]
    levered_beta   = unlevered_beta * (1 + (1 - tax_rate_est) * de_ratio)
    levered_beta   = float(np.clip(levered_beta, 0.5, 2.5))

    rfr = 0.07    # 10Y G-Sec yield
    erp = 0.055   # India ERP
    cost_of_equity = rfr + levered_beta * erp

    # Cost of debt from interest / total debt; credit quality adjustment
    interest_vals = _valid_floats(r.get("interest", []))
    if interest_vals and debt_vals and debt_vals[-1] > 0:
        raw_cod = interest_vals[-1] / debt_vals[-1]
        # Apply D/E-based credit quality floor (higher D/E → higher floor)
        credit_floor = 0.06 + 0.02 * min(de_ratio, 2.0)
        cost_of_debt = float(np.clip(raw_cod, credit_floor, 0.16))
    else:
        cost_of_debt = 0.09

    market_cap = float(data.get("market_cap") or 0)
    total_cap  = market_cap + latest_debt
    w_eq   = market_cap / total_cap if total_cap > 0 else 0.8
    w_dt   = 1 - w_eq
    base_wacc = w_eq * cost_of_equity + w_dt * cost_of_debt * (1 - tax_rate_est)
    base_wacc = float(np.clip(base_wacc, 0.07, 0.18))

    # ── Terminal growth ───────────────────────────────────
    # GDP proxy: India nominal GDP growth ~7% → real ~4%
    # Terminal growth = max(GDP/2, business_type_floor), capped at 6%
    gdp_proxy = 0.07
    base_tgr  = float(np.clip(
        max(gdp_proxy / 2, bp["tgr_floor"], base_growth / 2),
        bp["tgr_floor"], 0.06
    ))

    # Sanity check: terminal growth must be < WACC
    if base_tgr >= base_wacc:
        base_tgr = base_wacc - 0.02

    # ── Bull / Bear scenarios ─────────────────────────────
    bear_growth = float(np.clip(base_growth * 0.50, 0.02, 0.15))
    bull_growth = float(np.clip(base_growth * 1.60, base_growth + 0.03, 0.40))

    return {
        # Core growth
        "base_growth":          round(base_growth, 4),
        "bear_growth":          round(bear_growth, 4),
        "bull_growth":          round(bull_growth, 4),
        # Margins
        "base_ebitda_margin":   round(base_ebitda_margin_pct, 2),   # in %
        "base_margin_delta":    round(margin_delta, 4),
        "bear_margin_delta":    round(margin_delta - 0.005, 4),
        "bull_margin_delta":    round(margin_delta + 0.005, 4),
        # Capital structure / WACC
        "base_wacc":            round(base_wacc, 4),
        "beta":                 round(levered_beta, 2),
        "risk_free_rate":       rfr,
        "erp":                  erp,
        "cost_of_debt":         round(cost_of_debt, 4),
        "tax_rate":             round(tax_rate_est, 4),
        # Capex / WC
        "capex_pct":            round(capex_pct, 4),
        "wc_pct":               round(wc_pct, 4),
        # Terminal growth
        "base_tgr":             round(base_tgr, 4),
        "bear_tgr":             round(max(base_tgr - 0.01, 0.02), 4),
        "bull_tgr":             round(min(base_tgr + 0.01, 0.06), 4),
        # For display / downstream
        "de_ratio":             round(de_ratio, 2),
        "raw_growth_pct":       round(raw_growth * 100, 2),
        # Business context
        "business_type":        biz_type,
        "quality_score":        quality_score,
        "quality_components":   quality_components,
        "peer_median_margin":   round(peer_median_margin, 2) if peer_median_margin else None,
    }


# ─────────────────────────────────────────────
# WACC CALCULATOR
# ─────────────────────────────────────────────

def calculate_wacc(
    market_cap: float,
    total_debt: float,
    beta: float = 1.0,
    risk_free_rate: float = 0.07,
    equity_risk_premium: float = 0.055,
    cost_of_debt: float = 0.09,
    tax_rate: float = 0.25,
) -> dict:
    """Compute WACC using market-value weights."""
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    total_capital  = market_cap + total_debt
    if total_capital == 0:
        return {
            "wacc":                  0.11,
            "cost_of_equity":        cost_of_equity,
            "cost_of_debt_after_tax":cost_of_debt * (1 - tax_rate),
            "equity_weight":         1.0,
            "debt_weight":           0.0,
        }

    w_equity = market_cap / total_capital
    w_debt   = total_debt / total_capital
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc = (w_equity * cost_of_equity) + (w_debt * after_tax_cost_of_debt)

    return {
        "wacc":                  round(wacc, 4),
        "cost_of_equity":        round(cost_of_equity, 4),
        "cost_of_debt_after_tax":round(after_tax_cost_of_debt, 4),
        "equity_weight":         round(w_equity, 4),
        "debt_weight":           round(w_debt, 4),
        "beta":                  beta,
        "risk_free_rate":        risk_free_rate,
        "erp":                   equity_risk_premium,
    }


# ─────────────────────────────────────────────
# FCF PROJECTION
# ─────────────────────────────────────────────

def project_fcff(
    base_revenue: float,
    base_ebitda_margin: float,      # decimal (e.g. 0.18)
    revenue_growth: float,
    margin_improvement: float,
    capex_pct_revenue: float,
    wc_change_pct_revenue: float,
    tax_rate: float,
    n_years: int = 5,
) -> pd.DataFrame:
    """Project 5-year FCFF from base assumptions."""
    rows = []
    revenue       = base_revenue
    ebitda_margin = base_ebitda_margin

    for yr in range(1, n_years + 1):
        revenue       = revenue * (1 + revenue_growth)
        ebitda_margin = min(ebitda_margin + margin_improvement, 0.60)
        ebitda   = revenue * ebitda_margin
        nopat    = ebitda * (1 - tax_rate)
        capex    = revenue * capex_pct_revenue
        delta_wc = revenue * wc_change_pct_revenue
        fcff     = nopat - capex - delta_wc

        rows.append({
            "Year":             f"FY+{yr}",
            "Revenue":          round(revenue, 1),
            "EBITDA Margin %":  round(ebitda_margin * 100, 1),
            "EBITDA":           round(ebitda, 1),
            "NOPAT":            round(nopat, 1),
            "Capex":            round(capex, 1),
            "ΔWorking Capital": round(delta_wc, 1),
            "FCFF":             round(fcff, 1),
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
    shares_outstanding: float,
) -> dict:
    """Discount FCFF stream + terminal value to equity per share."""
    fcffs = fcff_df["FCFF"].tolist()
    n     = len(fcffs)

    pv_fcffs    = [fcff / ((1 + wacc) ** (i + 1)) for i, fcff in enumerate(fcffs)]
    sum_pv_fcff = sum(pv_fcffs)

    # Sanity guard: terminal growth < wacc
    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.01

    terminal_fcff     = fcffs[-1] * (1 + terminal_growth)
    terminal_value    = terminal_fcff / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)

    enterprise_value    = sum_pv_fcff + pv_terminal_value
    equity_value        = max(enterprise_value - total_debt + cash, 0)
    intrinsic_per_share = (equity_value / shares_outstanding
                           if shares_outstanding > 0 else np.nan)

    return {
        "pv_fcffs":           [round(v, 1) for v in pv_fcffs],
        "sum_pv_fcff":        round(sum_pv_fcff, 1),
        "terminal_value":     round(terminal_value, 1),
        "pv_terminal_value":  round(pv_terminal_value, 1),
        "enterprise_value":   round(enterprise_value, 1),
        "equity_value":       round(equity_value, 1),
        "intrinsic_per_share":round(intrinsic_per_share, 2) if not np.isnan(intrinsic_per_share) else np.nan,
        "tv_pct_of_ev":       round(pv_terminal_value / enterprise_value * 100, 1) if enterprise_value else np.nan,
    }


# ─────────────────────────────────────────────
# INVESTMENT SIGNAL GENERATOR
# ─────────────────────────────────────────────

def generate_investment_signal(
    quality_score: float,
    dcf_upside_pct: float,
    composite_score: float,
    business_type: str,
) -> dict:
    """
    Generate investment signal with quality-gated margin of safety.

    MOS requirements (based on quality score):
      Quality ≥ 80  → 25%+ MOS for BUY  (high quality = demand premium)
      Quality 60-80 → 20%+ MOS
      Quality 40-60 → 15%+ MOS
      Quality < 40  → No BUY signals (SELL / HOLD only)

    Additional rules:
      - Commodity businesses require +5% extra MOS due to earnings unpredictability.
      - Cyclical businesses require +3% extra MOS.
      - STRONG BUY only for quality ≥ 70 AND upside ≥ 40% AND composite ≥ 0.45.

    Returns
    -------
    dict with keys: signal, conviction, reason, mos_required, upside
    """
    upside = dcf_upside_pct if (dcf_upside_pct and not np.isnan(dcf_upside_pct)) else 0.0

    # Determine MOS requirement
    if quality_score >= 80:
        mos_required = 25.0
    elif quality_score >= 60:
        mos_required = 20.0
    elif quality_score >= 40:
        mos_required = 15.0
    else:
        mos_required = 999.0  # effectively blocks BUY

    # Business-type MOS premium
    if business_type == "commodity":
        mos_required += 5.0
    elif business_type == "cyclical":
        mos_required += 3.0

    # Derive signal
    if quality_score < 40:
        if composite_score < -0.20:
            signal    = "SELL"
            conviction = "Moderate"
            reason    = (f"Quality score {quality_score:.0f}/100 is below the minimum threshold "
                         f"for BUY consideration. Composite score {composite_score:+.2f} is bearish.")
        else:
            signal    = "HOLD"
            conviction = "Low"
            reason    = (f"Quality score {quality_score:.0f}/100 is too low to generate a BUY. "
                         f"Business quality must improve before accumulating.")
    elif upside >= mos_required and composite_score > 0.20:
        if quality_score >= 70 and upside >= 40.0 and composite_score >= 0.45:
            signal    = "STRONG BUY"
            conviction = "High"
            reason    = (f"High-quality business (score {quality_score:.0f}/100, {business_type}) "
                         f"with {upside:.1f}% DCF upside — well above the {mos_required:.0f}% "
                         f"margin of safety requirement. Composite score {composite_score:+.2f}.")
        else:
            signal    = "BUY"
            conviction = "High" if composite_score > 0.40 else "Moderate"
            reason    = (f"{business_type.title()} business (quality {quality_score:.0f}/100) "
                         f"with {upside:.1f}% upside exceeds {mos_required:.0f}% MOS requirement. "
                         f"Composite score {composite_score:+.2f}.")
    elif composite_score < -0.20:
        signal    = "SELL"
        conviction = "Strong" if composite_score < -0.50 else "Moderate"
        reason    = (f"Composite score {composite_score:+.2f} is bearish. "
                     f"DCF upside {upside:.1f}% is below the {mos_required:.0f}% MOS required "
                     f"for the quality score {quality_score:.0f}/100.")
    else:
        signal    = "HOLD"
        conviction = "Moderate" if abs(composite_score) > 0.10 else "Low"
        reason    = (f"Insufficient margin of safety. {upside:.1f}% upside < "
                     f"{mos_required:.0f}% required for {business_type} business "
                     f"(quality {quality_score:.0f}/100). Composite {composite_score:+.2f}.")

    return {
        "signal":       signal,
        "conviction":   conviction,
        "reason":       reason,
        "mos_required": mos_required,
        "upside":       round(upside, 1),
    }


# ─────────────────────────────────────────────
# COMPOSITE VALUATION SIGNAL (ENHANCED)
# ─────────────────────────────────────────────

def compute_composite_signal(
    current_price: float,
    base_iv: float,
    r: dict,
    data: dict,
    assumptions: dict = None,
) -> dict:
    """
    Quality-adjusted three-factor composite signal.

    Factors
    -------
    Factor 1 — DCF Upside (base weight 50%, rises to 65% for quality ≥ 70)
        score = clamp((IV/CMP - 1) / 0.30, -1, +1)

    Factor 2 — P/E Mean Reversion (weight 25%)
        PEG = 1 anchor: fair P/E = EPS CAGR × 1.2, bounded [8, 45]

    Factor 3 — EV/EBITDA vs Historical (weight 25%, drops to 10% for quality ≥ 70)
        Lower current vs normalised = undervalued.

    Signal is then passed through generate_investment_signal() which applies
    quality-gated MOS requirements.
    """
    assumptions = assumptions or {}
    signals      = {}

    quality_score  = float(assumptions.get("quality_score",  50.0))
    business_type  = assumptions.get("business_type", "stable")

    # Dynamic DCF weight: higher quality business → trust DCF more
    if quality_score >= 70:
        w_dcf, w_pe, w_ev = 0.65, 0.20, 0.15
    elif quality_score >= 50:
        w_dcf, w_pe, w_ev = 0.50, 0.25, 0.25
    else:
        # Low quality: give more weight to market-based signals
        w_dcf, w_pe, w_ev = 0.40, 0.30, 0.30

    # ── Factor 1: DCF upside ──────────────────────────────
    dcf_score      = 0.0
    dcf_upside_pct = np.nan
    if (base_iv and not np.isnan(base_iv) and
            current_price and not np.isnan(current_price) and current_price > 0):
        dcf_upside_pct = (base_iv - current_price) / current_price * 100
        dcf_score      = float(np.clip((base_iv / current_price - 1) / 0.30, -1.0, 1.0))
    signals["dcf"] = {
        "score":      round(dcf_score, 3),
        "upside_pct": round(dcf_upside_pct, 1) if not np.isnan(dcf_upside_pct) else np.nan,
    }

    # ── Factor 2: P/E mean reversion ─────────────────────
    pe_score   = 0.0
    current_pe = data.get("pe_ratio")
    eps_cagr3  = r.get("eps_cagr_3y")

    if (current_pe and eps_cagr3 is not None and
            not np.isnan(_safe_float(current_pe)) and
            not np.isnan(_safe_float(eps_cagr3))):
        current_pe = float(current_pe)
        eps_growth = float(eps_cagr3)
        fair_pe    = float(np.clip(eps_growth * 1.2, 8.0, 45.0))
        if fair_pe > 0 and current_pe > 0:
            pe_score = float(np.clip((fair_pe - current_pe) / fair_pe, -1.0, 1.0))
        signals["pe"] = {
            "score":      round(pe_score, 3),
            "current_pe": round(current_pe, 1),
            "fair_pe":    round(fair_pe, 1),
        }
    else:
        signals["pe"] = {"score": 0.0, "note": "insufficient data"}

    # ── Factor 3: EV/EBITDA vs historical norm ────────────
    ev_score          = 0.0
    ebitda_vals       = [v for v in r.get("ebitda", [])
                         if v is not None and not np.isnan(float(v))]
    market_cap        = float(data.get("market_cap") or 0)
    latest_debt       = float(last_valid(r.get("total_debt", [])) or 0)
    latest_cash       = float(last_valid(r.get("cash", [])) or 0)
    ev_now            = market_cap + latest_debt - latest_cash

    if len(ebitda_vals) >= 3 and ev_now > 0:
        median_ebitda          = float(np.median(ebitda_vals))
        latest_ebitda          = float(ebitda_vals[-1])
        hist_ev_ebitda_implied = ev_now / median_ebitda if median_ebitda > 0 else np.nan
        curr_ev_ebitda_calc    = ev_now / latest_ebitda  if latest_ebitda > 0 else np.nan

        if (not np.isnan(hist_ev_ebitda_implied) and
                not np.isnan(curr_ev_ebitda_calc) and
                hist_ev_ebitda_implied > 0):
            ev_score = float(np.clip(
                (hist_ev_ebitda_implied - curr_ev_ebitda_calc) / hist_ev_ebitda_implied,
                -1.0, 1.0,
            ))
        signals["ev_ebitda"] = {
            "score":               round(ev_score, 3),
            "current_ev_ebitda":   round(curr_ev_ebitda_calc, 1) if not np.isnan(curr_ev_ebitda_calc) else np.nan,
            "hist_ev_ebitda_norm": round(hist_ev_ebitda_implied, 1) if not np.isnan(hist_ev_ebitda_implied) else np.nan,
        }
    else:
        signals["ev_ebitda"] = {"score": 0.0, "note": "insufficient data"}

    # ── Composite score (quality-adjusted weights) ────────
    composite = (w_dcf * dcf_score +
                 w_pe  * pe_score  +
                 w_ev  * ev_score)

    # ── Quality-gated signal via generate_investment_signal ──
    inv_signal = generate_investment_signal(
        quality_score   = quality_score,
        dcf_upside_pct  = dcf_upside_pct if not np.isnan(dcf_upside_pct) else 0.0,
        composite_score = composite,
        business_type   = business_type,
    )

    signal     = inv_signal["signal"]
    conviction = inv_signal["conviction"]

    # Map to color
    signal_color = {
        "STRONG BUY": "green",
        "BUY":        "green",
        "HOLD":       "orange",
        "SELL":       "red",
    }.get(signal, "grey")

    return {
        "signal":            signal,
        "signal_color":      signal_color,
        "composite_score":   round(composite, 3),
        "conviction":        conviction,
        "upside_pct":        round(dcf_upside_pct, 1) if not np.isnan(dcf_upside_pct) else np.nan,
        "factor_scores":     signals,
        "quality_score":     quality_score,
        "business_type":     business_type,
        "mos_required":      inv_signal["mos_required"],
        "signal_reason":     inv_signal["reason"],
        "dcf_weight":        w_dcf,
    }


# ─────────────────────────────────────────────
# THREE SCENARIOS (BULL / BASE / BEAR)
# ─────────────────────────────────────────────

def run_three_scenarios(data: dict, r: dict, assumptions: dict) -> dict:
    """
    Run DCF for Bull, Base, Bear.  If assumptions contain 'auto_derived'=True
    they came from derive_assumptions_from_history(); otherwise user overrides.
    """
    base_revenue       = last_valid(r.get("revenue", []))
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt  = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None or np.isnan(float(total_debt or np.nan))) else float(total_debt)
    cash_val    = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None or np.isnan(float(cash_val   or np.nan))) else float(cash_val)
    market_cap  = data.get("market_cap") or np.nan
    current_price = data.get("current_price") or np.nan

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

    scenarios_params = {
        "Bear": {
            "revenue_growth":        assumptions.get("bear_growth", 0.05),
            "margin_improvement":    assumptions.get("bear_margin_delta", -0.005),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.08),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.02),
            "terminal_growth":       assumptions.get("bear_tgr", 0.03),
        },
        "Base": {
            "revenue_growth":        assumptions.get("base_growth", 0.12),
            "margin_improvement":    assumptions.get("base_margin_delta", 0.002),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.07),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.015),
            "terminal_growth":       assumptions.get("base_tgr", 0.04),
        },
        "Bull": {
            "revenue_growth":        assumptions.get("bull_growth", 0.20),
            "margin_improvement":    assumptions.get("bull_margin_delta", 0.008),
            "capex_pct_revenue":     assumptions.get("capex_pct", 0.06),
            "wc_change_pct_revenue": assumptions.get("wc_pct", 0.01),
            "terminal_growth":       assumptions.get("bull_tgr", 0.05),
        },
    }

    results = {}
    for name, s in scenarios_params.items():
        fcff_df = project_fcff(
            base_revenue=base_revenue or 10000,
            base_ebitda_margin=base_ebitda_margin,
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

    base_iv = results["Base"]["intrinsic_per_share"]

    # ── Composite signal (quality-aware) ──────────────────
    composite = compute_composite_signal(
        current_price=current_price,
        base_iv=base_iv,
        r=r,
        data=data,
        assumptions=assumptions,
    )

    return {
        "scenarios":        results,
        "wacc_result":      wacc_result,
        "current_price":    current_price,
        "base_iv":          base_iv,
        "upside_pct":       composite["upside_pct"],
        "margin_of_safety": composite["upside_pct"],
        "signal":           composite["signal"],
        "signal_color":     composite["signal_color"],
        "composite_score":  composite["composite_score"],
        "conviction":       composite["conviction"],
        "factor_scores":    composite["factor_scores"],
        "quality_score":    composite["quality_score"],
        "business_type":    composite["business_type"],
        "mos_required":     composite["mos_required"],
        "signal_reason":    composite["signal_reason"],
        "shares":           shares,
        "assumptions_used": assumptions,
    }


# ─────────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def run_monte_carlo(data: dict, r: dict, assumptions: dict, n_simulations: int = 10000) -> dict:
    """Run 10,000 Monte Carlo iterations with assumption randomisation."""
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt  = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None or np.isnan(float(total_debt or np.nan))) else float(total_debt) or 0
    cash_val    = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None or np.isnan(float(cash_val   or np.nan))) else float(cash_val) or 0
    market_cap  = data.get("market_cap") or 100000
    current_price = data.get("current_price") or 1000
    shares      = (market_cap / current_price) if current_price else 100

    base_growth = assumptions.get("base_growth", 0.12)
    base_tgr    = assumptions.get("base_tgr", 0.04)
    base_wacc   = assumptions.get("base_wacc", assumptions.get("wacc", 0.11))
    capex_pct   = assumptions.get("capex_pct", 0.07)
    tax_rate    = assumptions.get("tax_rate", 0.25)

    # Std devs calibrated from historical revenue volatility
    rev_vals = [v for v in r.get("revenue", [])
                if v is not None and not np.isnan(float(v))]
    if len(rev_vals) >= 3:
        yoy_growths = [(rev_vals[i + 1] - rev_vals[i]) / abs(rev_vals[i])
                       for i in range(len(rev_vals) - 1) if rev_vals[i] != 0]
        growth_std = float(np.clip(np.std(yoy_growths), 0.03, 0.12)) if yoy_growths else 0.06
    else:
        growth_std = 0.06

    np.random.seed(42)
    intrinsic_values = []

    for _ in range(n_simulations):
        rev_growth   = float(np.clip(np.random.normal(base_growth, growth_std), -0.10, 0.40))
        tgr          = float(np.clip(np.random.normal(base_tgr,    0.01),       0.01,  0.06))
        wacc         = float(np.clip(np.random.normal(base_wacc,   0.015),      0.06,  0.20))
        margin_delta = float(np.random.normal(assumptions.get("base_margin_delta", 0.002), 0.003))
        cap_pct      = float(np.clip(np.random.normal(capex_pct, 0.01), 0.02, 0.20))

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
            iv  = val["intrinsic_per_share"]
            if iv and not np.isnan(iv) and iv > 0:
                intrinsic_values.append(iv)
        except Exception:
            pass

    if not intrinsic_values:
        return {"error": "Monte Carlo failed — insufficient data"}

    arr = np.array(intrinsic_values)
    return {
        "values":           arr.tolist(),
        "mean":             round(float(arr.mean()), 2),
        "median":           round(float(np.median(arr)), 2),
        "p10":              round(float(np.percentile(arr, 10)), 2),
        "p25":              round(float(np.percentile(arr, 25)), 2),
        "p75":              round(float(np.percentile(arr, 75)), 2),
        "p90":              round(float(np.percentile(arr, 90)), 2),
        "std":              round(float(arr.std()), 2),
        "n_simulations":    len(arr),
        "current_price":    current_price,
        "prob_undervalued": round(float((arr > current_price).mean() * 100), 1),
    }


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS (TORNADO CHART DATA)
# ─────────────────────────────────────────────

def run_sensitivity(data: dict, r: dict, assumptions: dict) -> pd.DataFrame:
    """Compute one-at-a-time sensitivity of intrinsic value to key inputs."""
    base_revenue       = last_valid(r.get("revenue", [])) or 10000
    base_ebitda_margin = assumptions.get("base_ebitda_margin",
                         last_valid(r.get("operating_margin", [])) or 15.0) / 100.0
    total_debt  = last_valid(r.get("total_debt", []))
    total_debt  = 0 if (total_debt is None or np.isnan(float(total_debt or np.nan))) else float(total_debt) or 0
    cash_val    = last_valid(r.get("cash", []))
    cash_val    = 0 if (cash_val   is None or np.isnan(float(cash_val   or np.nan))) else float(cash_val) or 0
    market_cap  = data.get("market_cap") or 100000
    current_price = data.get("current_price") or 1000
    shares      = (market_cap / current_price) if current_price else 100

    base_wacc   = assumptions.get("base_wacc", assumptions.get("wacc", 0.11))
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
        except Exception:
            return np.nan

    base_iv = compute_iv(base_growth, base_wacc, base_tgr, capex_pct)

    variables = [
        ("Revenue Growth",  base_growth, 0.05, lambda v: compute_iv(v, base_wacc, base_tgr, capex_pct)),
        ("WACC",            base_wacc,   0.02, lambda v: compute_iv(base_growth, v, base_tgr, capex_pct)),
        ("Terminal Growth", base_tgr,    0.01, lambda v: compute_iv(base_growth, base_wacc, v, capex_pct)),
        ("Capex % Revenue", capex_pct,   0.02, lambda v: compute_iv(base_growth, base_wacc, base_tgr, v)),
        ("Tax Rate",        tax_rate,    0.05, lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct)),
        ("EBITDA Margin Δ", 0.002,       0.005,lambda v: compute_iv(base_growth, base_wacc, base_tgr, capex_pct, v)),
    ]

    rows = []
    for name, base_val, delta, fn in variables:
        low_iv  = fn(base_val - delta)
        high_iv = fn(base_val + delta)
        impact  = abs((high_iv or base_iv) - (low_iv or base_iv))
        rows.append({
            "Variable":   name,
            "Base IV":    round(base_iv, 2) if not np.isnan(base_iv) else np.nan,
            "Low IV":     round(low_iv,  2) if not np.isnan(low_iv)  else np.nan,
            "High IV":    round(high_iv, 2) if not np.isnan(high_iv) else np.nan,
            "Impact":     round(impact,  2),
            "Low Label":  f"{name} - {delta*100:.0f}%",
            "High Label": f"{name} + {delta*100:.0f}%",
        })

    return pd.DataFrame(rows).sort_values("Impact", ascending=True)