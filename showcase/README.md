# EQD Risk Engine — public showcase site

A public-facing, static Next.js site for the [eqd-risk-engine](../README.md) project —
built for linking from LinkedIn/personal-website, not for daily operational use (that's
the Streamlit dashboard at `app/dashboard.py`, one level up). Design/scope decisions are
locked in `../planning/showcase_site_plan.md`.

Five pages: Home, Daily P&L Explain, AI Risk Analyst, Hedging Strategy Lab, Market-Consistent
Pricing. Every number is read from a static JSON snapshot of real curated data —
there is no live backend, no API route, and no computation happens in the browser.

## How the data gets here

```
data/curated/*.parquet  --[scripts/export_showcase_data.py]-->  showcase/src/data/*.json
```

`../scripts/export_showcase_data.py` reads a handful of deliberately-chosen real dates
(see the plan doc for which, and why) from the project's curated Parquet tables and
writes flat JSON into `src/data/`. Those JSON files are committed — the site itself does
no data fetching, at build time or runtime.

To refresh the snapshot after a new pipeline run:

```bash
cd ..                                   # project root
source .venv/bin/activate
python scripts/export_showcase_data.py
cd showcase && npm run build            # verify it still builds
```

## Local development

```bash
npm install
npm run dev       # http://localhost:3000, hot reload, reads the committed JSON in src/data/
npm run build     # static export -> out/
npm run lint
```

## Deployment

Static export (`next.config.ts` sets `output: "export"`) — deployable to any static
host. Intended target: Vercel free tier, "root directory" set to `showcase/`, connected
to this repo. Deploys are triggered by pushing updated `src/data/*.json` (manual refresh
step above), not by any live pipeline hook.
