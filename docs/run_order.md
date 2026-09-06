# Run Order

This guide describes the intended order for running the submitted artefact. The
repository is staged so examiners can inspect the source code, run the local
website, and understand how each pipeline stage fits into the dissertation
workflow.

Large generated datasets and the full SQLite database are not committed to Git.
Stages that depend on collected data require either a fresh local collection run
or an equivalent local snapshot.

## 1. Environment Setup

Clone the repository, create a Python environment, and install the Python
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Node.js is required for the local website server:

```bash
npm run dev
```

## 2. Stage 01: Data Collection

Scripts:

- `scripts/stage_01_collection/collect_events_snapshot.py`
- `scripts/stage_01_collection/collect_price_history.py`

This stage requires live Polymarket API access through the Gamma and CLOB API
helpers in `scripts/stage_01_collection/api/`.

Expected outputs are local raw event and price-history snapshots used by later
stages. These generated files are not committed to Git.

## 3. Stage 02: Database Build

Scripts:

- `scripts/stage_02_database/create_event_database.py`
- `scripts/stage_02_database/load_events_snapshot.py`
- `scripts/stage_02_database/audit_event_database.py`

This stage creates the event-first SQLite database, loads a collected event
snapshot, and audits the resulting database.

Expected output:

- local SQLite database under `data/database/`

## 4. Stage 03: Analysis Dataset Preparation

Scripts:

- `scripts/stage_03_preparation/build_market_analysis_dataset.py`
- `scripts/stage_03_preparation/build_market_category_layer.py`
- `scripts/stage_03_preparation/qa_market_analysis_dataset.py`
- `scripts/stage_03_preparation/build_event_structure_dataset.py`

This stage converts the database into analysis-ready market, category, and
event-structure datasets.

Expected outputs are generated files under `data/outputs/`.

## 5. Stage 04: Descriptive Analysis

Scripts:

- `scripts/stage_04_analysis/build_descriptive_statistics.py`
- `scripts/stage_04_analysis/build_market_eda_summary.py`
- `scripts/stage_04_analysis/build_market_eda_figures.py`

This stage creates dissertation-ready descriptive statistics, EDA summaries, and
figures from the prepared analysis datasets.

Expected outputs are generated summary tables and figures under `data/outputs/`.

## 6. Stage 05: Retrieval Evaluation

Scripts:

- `scripts/stage_05_retrieval/build_retrieval_corpus.py`
- `scripts/stage_05_retrieval/analyse_retrieval_readiness.py`
- `scripts/stage_05_retrieval/run_tfidf_baseline.py`
- `scripts/stage_05_retrieval/run_event_aware_retrieval.py`
- `scripts/stage_05_retrieval/run_embedding_labelled_eval.py`

The labelled retrieval evaluation uses:

- `config/retrieval_eval_queries.csv`

This stage builds the retrieval corpus, checks retrieval readiness, compares a
TF-IDF baseline, runs event-aware retrieval, and evaluates labelled retrieval
queries.

Expected outputs are retrieval metrics, readiness checks, and evaluation
results under `data/outputs/`.

## 7. Stage 06: Forecasting And Backtesting

Scripts:

- `scripts/stage_06_forecasting/build_reliability_dataset.py`
- `scripts/stage_06_forecasting/build_reliability_figures.py`
- `scripts/stage_06_forecasting/build_winner_dataset.py`
- `scripts/stage_06_forecasting/build_winner_figures.py`
- `scripts/stage_06_forecasting/build_historical_aggregation_backtest.py`
- `scripts/stage_06_forecasting/summarise_aggregation_backtest.py`
- `scripts/stage_06_forecasting/build_event_normalisation_backtest.py`
- `scripts/stage_06_forecasting/build_backtest_example_table.py`
- `scripts/stage_06_forecasting/build_current_forecast_examples.py`
- `scripts/stage_06_forecasting/validate_forecasting_outputs.py`

This stage builds reliability and winner-analysis datasets, evaluates historical
aggregation methods, checks event-family normalisation, creates current forecast
examples, and validates the forecasting outputs.

Run `validate_forecasting_outputs.py` after the other forecasting scripts.

Expected outputs are forecast evidence, reliability outputs, backtest summaries,
and validation artefacts under `data/outputs/`.

## 8. Stage 07: Website Artefact

The website source is in `website/`. Generated JavaScript data assets used by
the interface are staged under `website/data/`.

Run the local dashboard server with:

```bash
npm run dev
```

Then open:

```text
http://127.0.0.1:4285/
```

Demo accounts:

```text
free@example.test / free-demo
premium@example.test / premium-demo
```

The premium workflow is implemented as a local dissertation prototype. Demo user
accounts are stored in `config/demo_users.json`.

## 9. Validation Checks

Python syntax check:

```bash
python3 -m compileall -q scripts
```

Node syntax checks:

```bash
node --check scripts/stage_07_website/auth_server.mjs
for f in website/*.js website/data/*.js; do node --check "$f" || exit 1; done
```

Website smoke-test expectations:

- `/` returns `200`
- anonymous `/workspace` redirects to `/`
- `/api/session` returns unauthenticated session JSON

## 10. Known Limitations

- Large generated datasets and the SQLite database are excluded from Git.
- Full end-to-end reproduction requires generated local data.
- Stage 01 depends on live API availability and may produce different snapshots
  over time.
- The website authentication is local prototype authentication, not production
  authentication.
- The premium forecast workflow is a dissertation artefact/prototype, not a
  commercial service.
