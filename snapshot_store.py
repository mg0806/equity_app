"""
snapshot_store.py

Point-in-time snapshot database for EquityLens.

The goal is to remove look-ahead bias from historical validation. Each saved
snapshot stores the exact company data, calculated ratios, peer table and model
metadata available on a capture date. Replays then select only snapshots whose
snapshot_date is <= the historical as_of date.
"""

import json
import sqlite3
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from dcf import derive_assumptions_from_screener, run_three_scenarios
from market_ready import build_market_ready_report
from ratios import calculate_ratios
from red_flags import detect_red_flags
from validation_engine import TRADING_DAYS


DEFAULT_DB_PATH = "data/equitylens_snapshots.sqlite"
SCHEMA_VERSION = 2
DEFAULT_MODEL_VERSION = "equitylens_v1.1_sector_nse_bse_events"
DEFAULT_SOURCE_VERSION = "screener_yahoo_nse_bse_v1"


def _now_date():
    return date.today().isoformat()


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def _df_to_payload(df):
    if df is None or not hasattr(df, "empty"):
        return {"__type__": "dataframe", "empty": True, "data": None}
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    return {
        "__type__": "dataframe",
        "empty": bool(safe.empty),
        "data": safe.to_json(orient="split", date_format="iso"),
    }


def _payload_to_df(payload):
    if not isinstance(payload, dict) or payload.get("__type__") != "dataframe" or payload.get("empty"):
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(payload["data"]), orient="split")
    except Exception:
        return pd.DataFrame()


def _encode(obj):
    if isinstance(obj, pd.DataFrame):
        return _df_to_payload(obj)
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if isinstance(obj, tuple):
        return [_encode(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _decode(obj):
    if isinstance(obj, dict) and obj.get("__type__") == "dataframe":
        return _payload_to_df(obj)
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _dumps(obj):
    return json.dumps(_encode(obj), default=_json_default, separators=(",", ":"))


def _loads(text):
    if not text:
        return None
    return _decode(json.loads(text))


class SnapshotStore:
    """SQLite-backed point-in-time snapshot repository."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

    def close(self):
        self.conn.close()

    def init_db(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS company_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                company_name TEXT,
                snapshot_date TEXT NOT NULL,
                fiscal_period TEXT,
                source_version TEXT,
                model_version TEXT,
                data_json TEXT NOT NULL,
                ratios_json TEXT NOT NULL,
                peer_df_json TEXT,
                assumptions_json TEXT,
                dcf_json TEXT,
                market_ready_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, snapshot_date)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date
                ON company_snapshots(ticker, snapshot_date);

            CREATE TABLE IF NOT EXISTS replay_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                signal TEXT,
                target_price REAL,
                current_price REAL,
                upside_pct REAL,
                confidence_score REAL,
                risk_score REAL,
                forward_1m_return_pct REAL,
                forward_3m_return_pct REAL,
                forward_6m_return_pct REAL,
                forward_12m_return_pct REAL,
                forward_max_drawdown_pct REAL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("company_snapshots", "model_version", "TEXT")
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def _ensure_column(self, table, column, column_type):
        cols = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def save_snapshot(
        self,
        ticker,
        data,
        ratios=None,
        peer_df=None,
        assumptions=None,
        dcf_result=None,
        market_ready=None,
        snapshot_date=None,
        fiscal_period=None,
        source_version=DEFAULT_SOURCE_VERSION,
        model_version=DEFAULT_MODEL_VERSION,
    ):
        """Save one point-in-time snapshot."""
        ticker = str(ticker or data.get("ticker") or "").upper().strip()
        if not ticker:
            raise ValueError("ticker is required")
        snapshot_date = snapshot_date or _now_date()
        ratios = ratios or calculate_ratios(data)
        created_at = datetime.utcnow().isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO company_snapshots(
                ticker, company_name, snapshot_date, fiscal_period, source_version,
                model_version, data_json, ratios_json, peer_df_json, assumptions_json, dcf_json,
                market_ready_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, snapshot_date) DO UPDATE SET
                company_name=excluded.company_name,
                fiscal_period=excluded.fiscal_period,
                source_version=excluded.source_version,
                model_version=excluded.model_version,
                data_json=excluded.data_json,
                ratios_json=excluded.ratios_json,
                peer_df_json=excluded.peer_df_json,
                assumptions_json=excluded.assumptions_json,
                dcf_json=excluded.dcf_json,
                market_ready_json=excluded.market_ready_json,
                created_at=excluded.created_at
            """,
            (
                ticker,
                data.get("company_name", ticker),
                snapshot_date,
                fiscal_period,
                source_version,
                model_version,
                _dumps(data),
                _dumps(ratios),
                _dumps(peer_df if peer_df is not None else pd.DataFrame()),
                _dumps(assumptions or {}),
                _dumps(dcf_result or {}),
                _dumps(market_ready or {}),
                created_at,
            ),
        )
        self.conn.commit()

    def latest_snapshot(self, ticker, as_of=None):
        """Return latest snapshot for ticker on or before as_of."""
        ticker = str(ticker).upper().strip()
        params = [ticker]
        where = "ticker = ?"
        if as_of:
            where += " AND snapshot_date <= ?"
            params.append(str(as_of))
        row = self.conn.execute(
            f"""
            SELECT * FROM company_snapshots
            WHERE {where}
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def snapshots(self, ticker=None, start=None, end=None):
        """Return saved snapshots, optionally filtered."""
        where = []
        params = []
        if ticker:
            where.append("ticker = ?")
            params.append(str(ticker).upper().strip())
        if start:
            where.append("snapshot_date >= ?")
            params.append(str(start))
        if end:
            where.append("snapshot_date <= ?")
            params.append(str(end))
        sql = "SELECT * FROM company_snapshots"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ticker, snapshot_date"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def inventory(self):
        rows = self.conn.execute(
            """
            SELECT ticker, company_name, COUNT(*) AS snapshots,
                   MIN(snapshot_date) AS first_snapshot,
                   MAX(snapshot_date) AS latest_snapshot
            FROM company_snapshots
            GROUP BY ticker, company_name
            ORDER BY ticker
            """
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def _row_to_snapshot(self, row):
        if row is None:
            return None
        return {
            "id": row["id"],
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "snapshot_date": row["snapshot_date"],
            "fiscal_period": row["fiscal_period"],
            "source_version": row["source_version"],
            "model_version": row["model_version"] or DEFAULT_MODEL_VERSION,
            "data": _loads(row["data_json"]),
            "ratios": _loads(row["ratios_json"]),
            "peer_df": _loads(row["peer_df_json"]),
            "assumptions": _loads(row["assumptions_json"]) or {},
            "dcf_result": _loads(row["dcf_json"]) or {},
            "market_ready": _loads(row["market_ready_json"]) or {},
            "created_at": row["created_at"],
        }

    def save_replay_result(self, run_name, result):
        created_at = datetime.utcnow().isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO replay_results(
                run_name, ticker, snapshot_date, signal, target_price,
                current_price, upside_pct, confidence_score, risk_score,
                forward_1m_return_pct, forward_3m_return_pct,
                forward_6m_return_pct, forward_12m_return_pct,
                forward_max_drawdown_pct, result_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                result.get("ticker"),
                result.get("snapshot_date"),
                result.get("signal"),
                result.get("target_price"),
                result.get("current_price"),
                result.get("upside_pct"),
                result.get("confidence_score"),
                result.get("risk_score"),
                result.get("forward_1m_return_pct"),
                result.get("forward_3m_return_pct"),
                result.get("forward_6m_return_pct"),
                result.get("forward_12m_return_pct"),
                result.get("forward_max_drawdown_pct"),
                _dumps(result),
                created_at,
            ),
        )
        self.conn.commit()


def _price_at_or_after(history, snapshot_date, days=0):
    if history is None or not hasattr(history, "empty") or history.empty or "Close" not in history:
        return np.nan
    df = history.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        return np.nan
    start_date = pd.to_datetime(snapshot_date)
    start_idx = df.index[df["Date"] >= start_date]
    if len(start_idx) == 0:
        return np.nan
    idx = int(start_idx[0]) + days
    if idx >= len(df):
        return np.nan
    return float(df.loc[idx, "Close"])


def _forward_return(history, snapshot_date, days):
    start = _price_at_or_after(history, snapshot_date, 0)
    end = _price_at_or_after(history, snapshot_date, days)
    if np.isnan(start) or np.isnan(end) or start == 0:
        return np.nan
    return (end / start - 1) * 100


def _forward_drawdown(history, snapshot_date, days=252):
    if history is None or not hasattr(history, "empty") or history.empty or "Close" not in history:
        return np.nan
    df = history.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    idxs = df.index[df["Date"] >= pd.to_datetime(snapshot_date)]
    if len(idxs) == 0:
        return np.nan
    idx = int(idxs[0])
    end = min(idx + days, len(df) - 1)
    window = df.loc[idx:end, "Close"]
    if window.empty:
        return np.nan
    dd = (window / window.cummax() - 1) * 100
    return float(dd.min())


def replay_snapshot(snapshot, run_monte_carlo=False, return_price_history=None):
    """
    Replay the full DCF and market-ready signal using only stored snapshot data.

    This is the key point-in-time operation: no live fetch occurs here.
    """
    data = snapshot["data"]
    ratios = snapshot.get("ratios") or calculate_ratios(data)
    peer_df = snapshot.get("peer_df")
    if peer_df is None or not hasattr(peer_df, "empty"):
        peer_df = pd.DataFrame()
    flags = detect_red_flags(ratios)
    assumptions = snapshot.get("assumptions") or derive_assumptions_from_screener(data, ratios, peer_df)
    dcf_result = snapshot.get("dcf_result") or {}
    if not dcf_result:
        try:
            dcf_result = run_three_scenarios(data, ratios, assumptions)
        except Exception as exc:
            dcf_result = {"error": str(exc), "signal": "N/A"}
    mc_result = None
    market_ready = build_market_ready_report(data, ratios, flags, peer_df, dcf_result, mc_result, assumptions)
    target = market_ready.get("target", {})
    confidence = market_ready.get("confidence", {})
    risk = market_ready.get("risk", {})
    hist = return_price_history
    if hist is None:
        hist = (data.get("market_data") or {}).get("price_history")
    result = {
        "ticker": snapshot["ticker"],
        "company_name": snapshot.get("company_name"),
        "snapshot_date": snapshot["snapshot_date"],
        "signal": dcf_result.get("signal", "N/A"),
        "target_price": target.get("target_price"),
        "current_price": data.get("current_price"),
        "upside_pct": target.get("upside_pct"),
        "confidence_score": confidence.get("score"),
        "risk_score": risk.get("score"),
        "forward_1m_return_pct": _forward_return(hist, snapshot["snapshot_date"], TRADING_DAYS["1M"]),
        "forward_3m_return_pct": _forward_return(hist, snapshot["snapshot_date"], TRADING_DAYS["3M"]),
        "forward_6m_return_pct": _forward_return(hist, snapshot["snapshot_date"], TRADING_DAYS["6M"]),
        "forward_12m_return_pct": _forward_return(hist, snapshot["snapshot_date"], TRADING_DAYS["12M"]),
        "forward_max_drawdown_pct": _forward_drawdown(hist, snapshot["snapshot_date"], TRADING_DAYS["12M"]),
        "market_ready": market_ready,
        "dcf_result": dcf_result,
    }
    return result


def replay_snapshot_backtest(store, tickers=None, start=None, end=None, run_name=None, persist=True):
    """Replay all saved snapshots and summarize point-in-time signal evidence."""
    run_name = run_name or f"snapshot_replay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    snapshots = []
    if tickers:
        for ticker in tickers:
            snapshots.extend(store.snapshots(ticker=ticker, start=start, end=end))
    else:
        snapshots = store.snapshots(start=start, end=end)

    latest_price_history = {}
    for snapshot in snapshots:
        ticker = snapshot["ticker"]
        if ticker not in latest_price_history:
            latest = store.latest_snapshot(ticker)
            latest_price_history[ticker] = ((latest or {}).get("data", {}).get("market_data") or {}).get("price_history")

    results = []
    for snapshot in snapshots:
        result = replay_snapshot(snapshot, return_price_history=latest_price_history.get(snapshot["ticker"]))
        results.append(result)
        if persist:
            store.save_replay_result(run_name, result)

    df = pd.DataFrame([
        {k: v for k, v in result.items() if k not in ("market_ready", "dcf_result")}
        for result in results
    ])
    return {
        "run_name": run_name,
        "results": results,
        "table": df,
        "summary": summarize_replay_results(df),
    }


def summarize_replay_results(df):
    if df is None or df.empty or "signal" not in df.columns:
        return pd.DataFrame()
    rows = []
    for signal, group in df.groupby("signal", dropna=False):
        row = {"Signal": signal, "Observations": len(group)}
        for label, col in [
            ("1M", "forward_1m_return_pct"),
            ("3M", "forward_3m_return_pct"),
            ("6M", "forward_6m_return_pct"),
            ("12M", "forward_12m_return_pct"),
        ]:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"Median {label} Return %"] = vals.median()
            row[f"{label} Hit Rate %"] = (vals > 0).mean() * 100
        row["Median Forward Max Drawdown %"] = pd.to_numeric(
            group["forward_max_drawdown_pct"], errors="coerce"
        ).median()
        rows.append(row)
    return pd.DataFrame(rows).round(1)
