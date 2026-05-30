# Deployment Runbook

## Local Demo

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Demo Checklist

Use 3-4 tickers to show breadth:

- `RELIANCE`: diversified large-cap.
- `BAJAJHFL`: NBFC/housing finance.
- `BSE`: exchange-platform valuation.
- `INFY` or `TCS`: high-margin stable IT services.

For each demo:

1. Generate report.
2. Open Decision tab.
3. Explain blended target bridge.
4. Show DCF assumptions and Monte Carlo.
5. Show peer comparison and peer ranking.
6. Open Report History in the sidebar and show the saved model-versioned snapshot.
7. Download PDF and Excel.

## Production Deployment Options

### Option 1: Internal Streamlit Server

Best for early pilots.

- Deploy on one VM.
- Restrict access through VPN.
- Add reverse proxy and HTTPS.
- Add basic authentication or SSO proxy.

### Option 2: Containerized App

Recommended for enterprise pilots.

- Dockerize app.
- Deploy behind Nginx.
- Keep SQLite on persistent disk for pilots, or move report snapshots to managed Postgres for multi-user deployments.
- Store generated PDF/Excel artifacts in object storage only if long-term binary retention is required.
- Add monitoring and logs.

### Option 3: Multi-User SaaS

Requires product hardening:

- Authentication.
- Tenant separation.
- Database.
- Job queue for report generation.
- Rate limits.
- API keys and source-provider management.

## Environment Variables To Add Later

```text
APP_ENV=production
DATA_CACHE_TTL_HOURS=24
REPORT_STORAGE_PATH=...
SNAPSHOT_DB_PATH=data/equitylens_snapshots.sqlite
AUTH_PROVIDER=...
```

## Operational Risks

- Screener page structure can change.
- Source websites can block or throttle requests.
- yfinance can fail intermittently.
- Long Monte Carlo jobs can slow demos.
- SQLite is suitable for a local/internal pilot but should be migrated to managed database storage for multi-user SaaS.
- Public NSE/BSE pages are best-effort sources; licensed feeds are needed for commercial redistribution.

## Production Hardening Backlog

- Add centralized request cache beyond local `.cache/`.
- Add background report jobs.
- Add user auth.
- Add source timestamps.
- Add assumption override audit log.
- Add in-app validation dashboard for snapshot replay.
- Add API provider abstraction for licensed feeds.
