# Security And Privacy

## Current Security Posture

EquityLens is currently a local/internal Streamlit application. It includes a local SQLite report/snapshot database, but it does not yet include enterprise authentication, authorization, or multi-user access controls.

## Data Handled

- Public company financial data.
- Public market price data.
- Public NSE/BSE corporate actions and announcements.
- Generated reports.
- Local SQLite report snapshots with model/source versions.
- Analyst-entered ticker symbols and assumption overrides.

The app does not intentionally collect personal investor data.

## Enterprise Requirements Before Sale

- Add authentication.
- Add role-based access control.
- Serve over HTTPS.
- Store secrets outside code.
- Log report generation events.
- Protect or migrate the local SQLite snapshot database for multi-user deployments.
- Store generated report metadata in a managed database for SaaS deployments.
- Add data-source timestamps for every field.
- Add assumption override audit logging.
- Disable arbitrary file access in deployment.

## Recommended Architecture

```text
Browser
  -> HTTPS reverse proxy
  -> Auth layer / SSO
  -> Streamlit app
  -> SQLite report history for pilots / managed DB for SaaS
  -> Local `.cache/` for exchange-event responses
  -> Data provider adapters
```

## Secrets

Do not commit:

- API keys.
- Client files.
- Generated reports.
- `data/equitylens_snapshots.sqlite`.
- `.cache/` contents if they contain client-specific usage traces.
- `.streamlit/secrets.toml`.
- `.env` files.

## Privacy Statement Draft

EquityLens processes public market and company financial data for research automation. It does not provide personalized investment advice and does not require personal portfolio holdings to generate reports. Users should consult a SEBI-registered investment advisor or qualified financial professional before making investment decisions.
