# Forecast Evidence Website

This folder contains the runnable website artefact for the dissertation submission.

## Run Locally

From the repository root:

```bash
npm run dev -- --port 4285
```

Then open:

```text
http://127.0.0.1:4285/
```

Do not open `website/index.html` directly as a file. The page needs the
local server for sign in, create account, sessions, and logout. If the file is
opened directly, it redirects to the local server URL.

## Current Scope

Current implemented slice:

- sign in;
- create account;
- dynamic sign-in/create-account page copy;
- signed-in session display;
- signed-in workspace landing page;
- workspace-level project summary and headline findings;
- graph-led data analysis page;
- main results page;
- protected forecast workflow skeleton;
- premium-gated question capture;
- query-led forecast API bridge;
- separate retrieved-market evidence review page;
- separate forecast result comparison page;
- historical backtesting evidence page;
- sign out.

The MVP deliberately keeps methodological explanation in the dissertation
writeup rather than adding a separate Methods or Preview page.

## Access Model

The final website uses a sign-in-first flow. Unauthenticated users can only use
the account access page. After sign-in, free users can open the landing page,
Data Analysis, Results, and Backtesting. Premium users can use those pages and
also run custom forecast questions through the Forecast -> Evidence -> Result
workflow.

The premium distinction is enforced server-side by `/api/forecast/query`, which
returns `403` for authenticated free users and runs the query-led retrieval and
aggregation workflow for premium users.

## Local Auth State

The local server stores created accounts in:

```text
.local/website_users.json
```

That directory is ignored by git. The initial local user store is seeded from
`config/demo_users.json`.

## Authentication Behaviour

The local authentication server exposes:

- `GET /api/session`
- `POST /api/login`
- `POST /api/register`
- `POST /api/logout`
- `GET /api/overview`
- `GET /api/results`
- `POST /api/forecast/query`

Passwords are hashed with PBKDF2-SHA256, a random salt, and 210,000 iterations.
Successful sign in and registration create a signed HttpOnly session cookie.
The overview endpoint reads the generated analytics data used by the existing
dashboard and returns a compact internal summary for the final website pages.
The results endpoint combines that analytics data with the historical
aggregation backtest summary. The forecast query
endpoint is
premium-gated and calls the Python query-led retrieval/aggregation
service through `scripts/stage_07_website/query_forecast_api.py`.

This is local prototype authentication for the dissertation artefact. It is not
yet production authentication.

## Current Test Checks

Current checks for this first slice:

- `node --check website/app.js`
- `node --check scripts/stage_07_website/auth_server.mjs`
- local page request returns `200`
- demo premium login returns an authenticated session
- registration creates a local free account
- signed-in users can open `/workspace`
- anonymous users are redirected away from `/workspace`
- signed-in free users can open `/data-analysis`, `/results`, and
  `/backtesting`
- anonymous users are redirected away from `/data-analysis`, `/results`, and
  `/backtesting`
- signed-in users can open `/forecast`
- anonymous users are redirected away from `/forecast`
- free users receive `403` from `/api/forecast/query`
- premium users receive forecast-query payloads from `/api/forecast/query`
- logout clears the session
