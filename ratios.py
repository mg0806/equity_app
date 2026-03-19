"""
ratios.py — Fixed version. All extraction bugs resolved.
"""

import numpy as np
import pandas as pd
from scraper import extract_series, get_year_labels


def safe_div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((b != 0) & ~np.isnan(b) & ~np.isnan(a), a / b, np.nan)
    return result.tolist()

def pct(a, b):
    return safe_div(np.array(a, dtype=float) * 100, b)

def last_valid(lst, default=np.nan):
    if isinstance(lst, (int, float)):
        v = float(lst)
        return v if not np.isnan(v) else default
    if lst is None:
        return default
    for v in reversed(lst):
        try:
            f = float(v)
            if not np.isnan(f):
                return f
        except:
            pass
    return default

def calculate_ratios(data: dict, n_years: int = 5) -> dict:
    pl = data.get("pl", pd.DataFrame())
    bs = data.get("bs", pd.DataFrame())
    cf = data.get("cf", pd.DataFrame())
    years = get_year_labels(pl, n_years)
    r = {"years": years, "n_years": n_years}

    # ── P&L ──
    # FIX: Use "^Sales" not "Sales$" — Screener row is "Sales +"
    revenue      = extract_series(pl, [r"^Sales", r"Revenue from Operations", r"Revenue"], n_years)
    ebitda       = extract_series(pl, [r"Operating Profit", r"EBITDA", r"PBDIT"], n_years)
    # FIX: "Net Profit +" won't match "Net Profit$" — remove anchor
    pat          = extract_series(pl, [r"Net Profit", r"^PAT$", r"Profit after tax"], n_years)
    interest     = extract_series(pl, [r"^Interest", r"Finance Cost"], n_years)
    depreciation = extract_series(pl, [r"^Depreciation", r"Amortisation"], n_years)
    tax          = extract_series(pl, [r"^Tax", r"Income Tax"], n_years)
    eps          = extract_series(pl, [r"EPS", r"Earnings Per Share"], n_years)
    raw_material = extract_series(pl, [r"Raw Material", r"Material Cost"], n_years)

    # ── Balance Sheet ──
    total_assets   = extract_series(bs, [r"Total Assets", r"Total Liabilities", r"Balance Sheet"], n_years)
    # FIX: Equity = Equity Capital + Reserves (Screener splits them; no combined row)
    equity_capital = extract_series(bs, [r"^Equity Capital", r"Share Capital"], n_years)
    reserves       = extract_series(bs, [r"^Reserves", r"Reserves and Surplus"], n_years)
    networth       = extract_series(bs, [r"Net Worth", r"Shareholders.*Equity", r"Total Equity"], n_years)
    ec_arr = np.array(equity_capital, dtype=float)
    rs_arr = np.array(reserves, dtype=float)
    nw_arr = np.array(networth, dtype=float)
    equity = np.where(
        ~np.isnan(nw_arr), nw_arr,
        np.where(~np.isnan(ec_arr) & ~np.isnan(rs_arr), ec_arr + rs_arr,
        np.where(~np.isnan(rs_arr), rs_arr, ec_arr))
    ).tolist()

    total_debt   = extract_series(bs, [r"^Borrowings", r"Long Term Borrow", r"Total.*Debt"], n_years)
    # FIX: Screener detailed BS may not have "Current Assets" as a row
    # Try explicit names, then fall back to "Other Assets" (which is current assets in compact view)
    current_assets = extract_series(bs, [r"Current Assets", r"Total Current Asset", r"Other Assets"], n_years)
    current_liab   = extract_series(bs, [r"Current Liabilities", r"Total Current Liab", r"Other Liabilities"], n_years)
    inventory    = extract_series(bs, [r"Inventories", r"^Inventory", r"Stock"], n_years)
    receivables  = extract_series(bs, [r"Debtors", r"Trade Receivable", r"Receivables", r"Sundry Debtor"], n_years)
    payables     = extract_series(bs, [r"Creditors", r"Trade Payable", r"Payables", r"Sundry Creditor"], n_years)
    cash         = extract_series(bs, [r"Cash", r"Bank Balance"], n_years)
    fixed_assets = extract_series(bs, [r"Fixed Assets", r"Net Block", r"Tangible Asset"], n_years)

    # ── Cash Flow ──
    cfo  = extract_series(cf, [r"Cash from Operating", r"Operating Activit", r"Net Cash.*Operat"], n_years)
    cfi  = extract_series(cf, [r"Cash from Investing", r"Investing Activit"], n_years)
    cff  = extract_series(cf, [r"Cash from Financing", r"Financing Activit"], n_years)
    # FIX: Capex rarely explicit in Screener CF — estimate as 70% of |CFI|
    capex_explicit = extract_series(cf, [r"Capital Expenditure", r"Purchase.*Fixed", r"Capex", r"Fixed Assets.*Purchased"], n_years)
    cfi_abs = np.abs(np.array(cfi, dtype=float))
    capex_arr = np.array(capex_explicit, dtype=float)
    capex = np.where(~np.isnan(capex_arr), capex_arr, cfi_abs * 0.70).tolist()

    r.update({
        "revenue": revenue, "ebitda": ebitda, "pat": pat, "interest": interest,
        "depreciation": depreciation, "tax": tax, "eps": eps,
        "total_assets": total_assets, "total_debt": total_debt, "equity": equity,
        "current_assets": current_assets, "current_liab": current_liab,
        "inventory": inventory, "receivables": receivables, "payables": payables,
        "cash": cash, "fixed_assets": fixed_assets,
        "cfo": cfo, "cfi": cfi, "cff": cff, "capex": capex,
    })

    r["fcf"] = (np.array(cfo, dtype=float) - np.array(capex, dtype=float)).tolist()

    # ── Profitability ──
    rev_arr = np.array(revenue, dtype=float)
    pat_arr = np.array(pat, dtype=float)
    ebt_arr = np.array(ebitda, dtype=float)
    eq_arr  = np.array(equity, dtype=float)
    ta_arr  = np.array(total_assets, dtype=float)
    dep_arr = np.array(depreciation, dtype=float)
    cl_arr  = np.array(current_liab, dtype=float)
    int_arr = np.array(interest, dtype=float)

    r["gross_margin"]     = pct(rev_arr - np.array(raw_material, dtype=float), rev_arr)
    r["operating_margin"] = pct(ebt_arr, rev_arr)
    r["net_margin"]       = pct(pat_arr, rev_arr)
    r["roe"]              = pct(pat_arr, eq_arr)
    r["roa"]              = pct(pat_arr, ta_arr)
    ebit = (ebt_arr - dep_arr).tolist()
    cap_emp = np.where(np.isnan(cl_arr), ta_arr, ta_arr - cl_arr)
    r["ebit"] = ebit
    r["roce"] = pct(ebit, cap_emp)

    # ── Liquidity ──
    ca_arr   = np.array(current_assets, dtype=float)
    inv_arr  = np.array(inventory, dtype=float)
    cash_arr = np.array(cash, dtype=float)
    # Fallback estimates when current assets/liab not available on compact BS
    ca_safe  = np.where(np.isnan(ca_arr),  ta_arr * 0.25, ca_arr)
    cl_safe  = np.where(np.isnan(cl_arr),  ta_arr * 0.20, cl_arr)
    inv_safe = np.where(np.isnan(inv_arr), 0, inv_arr)
    cash_safe = np.where(np.isnan(cash_arr), 0, cash_arr)
    r["current_ratio"] = safe_div(ca_safe, cl_safe)
    r["quick_ratio"]   = safe_div(ca_safe - inv_safe, cl_safe)
    r["cash_ratio"]    = safe_div(cash_safe, cl_safe)

    # ── Solvency ──
    r["debt_equity"]    = safe_div(total_debt, equity)
    r["debt_ebitda"]    = safe_div(total_debt, ebitda)
    r["interest_cover"] = safe_div(ebt_arr, int_arr)

    # ── Efficiency ──
    rec_arr = np.array(receivables, dtype=float)
    pay_arr = np.array(payables, dtype=float)
    r["asset_turnover"]  = safe_div(rev_arr, ta_arr)
    r["receivable_days"] = safe_div(rec_arr * 365, rev_arr)
    r["payable_days"]    = safe_div(pay_arr * 365, rev_arr)
    r["inventory_days"]  = safe_div(inv_arr * 365, rev_arr)
    rec_d = np.array(r["receivable_days"], dtype=float)
    inv_d = np.array(r["inventory_days"], dtype=float)
    pay_d = np.array(r["payable_days"], dtype=float)
    r["cash_conversion"] = (rec_d + np.where(np.isnan(inv_d), 0, inv_d) - pay_d).tolist()

    # ── Valuation ──
    r["pe_ratio"]       = data.get("pe_ratio")
    r["current_price"]  = data.get("current_price")
    r["book_value"]     = data.get("book_value")
    r["market_cap"]     = data.get("market_cap")
    r["dividend_yield"] = data.get("dividend_yield")
    r["high_52w"]       = data.get("high_52w")
    r["low_52w"]        = data.get("low_52w")
    bv = data.get("book_value"); cp = data.get("current_price")
    r["pb_ratio"] = float(cp)/float(bv) if (cp and bv and float(bv)!=0) else np.nan
    mktcap = float(data.get("market_cap") or 0)
    ldebt  = float(last_valid(total_debt) or 0)
    lcash  = float(last_valid(cash) or 0)
    lebitda = last_valid(ebitda)
    ev = mktcap + ldebt - lcash
    r["ev_ebitda"] = ev/lebitda if (lebitda and not np.isnan(lebitda) and lebitda!=0) else np.nan

    # ── DuPont ──
    r["dupont_net_margin"]     = r["net_margin"]
    r["dupont_asset_turnover"] = r["asset_turnover"]
    r["dupont_equity_mult"]    = safe_div(ta_arr, eq_arr)
    r["dupont_roe"] = (
        np.array(r["dupont_net_margin"], dtype=float) / 100 *
        np.array(r["dupont_asset_turnover"], dtype=float) *
        np.array(r["dupont_equity_mult"], dtype=float) * 100
    ).tolist()

    # ── CAGR ──
    def cagr(series, n):
        vals = [float(v) for v in series if v is not None]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= n+1 and vals[-(n+1)] > 0:
            return ((vals[-1]/vals[-(n+1)])**(1/n)-1)*100
        return np.nan

    r["revenue_cagr_3y"] = cagr(revenue, 3)
    r["revenue_cagr_5y"] = cagr(revenue, 5)
    r["pat_cagr_3y"]     = cagr(pat, 3)
    r["pat_cagr_5y"]     = cagr(pat, 5)
    r["eps_cagr_3y"]     = cagr(eps, 3)
    r["eps_cagr_5y"]     = cagr(eps, 5)

    return r


def build_peer_comparison(target_ticker, target_data, target_r, peer_data_list):
    rows = []
    rows.append({
        "Company": target_data.get("company_name", target_ticker),
        "Ticker": target_ticker,
        "Mkt Cap (Cr)": target_data.get("market_cap"),
        "CMP (₹)": target_data.get("current_price"),
        "P/E": target_data.get("pe_ratio"),
        "P/B": target_r.get("pb_ratio"),
        "EV/EBITDA": target_r.get("ev_ebitda"),
        "Net Margin %": last_valid(target_r.get("net_margin", [])),
        "ROE %": last_valid(target_r.get("roe", [])),
        "ROCE %": last_valid(target_r.get("roce", [])),
        "D/E": last_valid(target_r.get("debt_equity", [])),
        "Current Ratio": last_valid(target_r.get("current_ratio", [])),
        "Revenue CAGR 3Y %": target_r.get("revenue_cagr_3y"),
        "_is_target": True,
    })
    for pdata in peer_data_list:
        try:
            pr = calculate_ratios(pdata)
            rows.append({
                "Company": pdata.get("company_name", pdata.get("ticker","")),
                "Ticker": pdata.get("ticker",""),
                "Mkt Cap (Cr)": pdata.get("market_cap"),
                "CMP (₹)": pdata.get("current_price"),
                "P/E": pdata.get("pe_ratio"),
                "P/B": pr.get("pb_ratio"),
                "EV/EBITDA": pr.get("ev_ebitda"),
                "Net Margin %": last_valid(pr.get("net_margin",[])),
                "ROE %": last_valid(pr.get("roe",[])),
                "ROCE %": last_valid(pr.get("roce",[])),
                "D/E": last_valid(pr.get("debt_equity",[])),
                "Current Ratio": last_valid(pr.get("current_ratio",[])),
                "Revenue CAGR 3Y %": pr.get("revenue_cagr_3y"),
                "_is_target": False,
            })
        except Exception as e:
            print(f"Peer error {pdata.get('ticker','?')}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num_cols = ["P/E","P/B","EV/EBITDA","Net Margin %","ROE %","ROCE %","D/E","Current Ratio","Revenue CAGR 3Y %"]
    peer_only = df[df["_is_target"]==False][num_cols]
    if not peer_only.empty:
        med = {}
        for c in num_cols:
            try:
                v = peer_only[c].astype(float).median()
                med[c] = round(float(v),2) if not np.isnan(v) else np.nan
            except:
                med[c] = np.nan
        df = pd.concat([df, pd.DataFrame([{"Company":"Industry Median","Ticker":"—",**med,"_is_target":None}])], ignore_index=True)
    return df
