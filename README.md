# Prediction Market Analytics Dashboard

This repository contains the source-code artefact for an MSc Data Analytics
dissertation on prediction-market analytics using Polymarket data.

The project builds a local dashboard for exploring prediction-market evidence,
including dataset analysis, retrieval, historical backtesting and a role-gated
premium forecast workflow.

## Structure

- `config/` contains small configuration files and labelled query data.
- `data/` contains placeholder folders for local databases and generated
  outputs. Large data files are not stored in Git.
- `scripts/` contains the reproducible project pipeline, ordered by stage.
- `website/` contains the final dashboard interface.
- `tests/` contains automated checks for key project behaviour.
- `docs/` contains short run and testing notes for the submitted artefact.

## Data

The full SQLite database and generated analysis outputs are excluded from
version control because they are large generated artefacts. The folder structure
is kept so the pipeline can be run with a local database snapshot.

## Website

Run the local dashboard server with:

```bash
npm run dev
```

Demo users are provided in `config/demo_users.json`.
