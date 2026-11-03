# Data Factory — Stage 3 (Machine learning)

Stage 1 built a resumable, deterministic factory for NYC 311 data. Stage 2 added PostgreSQL
and the Olist data. Stage 3 adds a file source (Kaggle "Give Me Some Credit") through the same
package and database, and a default model whose decision threshold minimises a 12:1 cost.

## Run it

Requirements: **Python 3.11+** and a **PostgreSQL server** (measured on 18.6; no Docker needed).

```bash
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

# Connection: no password or port is stored in the repository.
set PGPASSWORD=<postgres password>          # macOS/Linux: export PGPASSWORD=...
set PGPORT=5432                             # only if the server is not on 5432
# or replace the whole URL: set DATABASE_URL=postgresql+psycopg://user@host:port/dbname

data-factory check-db                       # connection test
# Stage 3 input: download cs-training.csv from
# https://www.kaggle.com/competitions/GiveMeSomeCredit/data (accept the rules) into data/raw/credit/
data-factory run                            # one command: empty server -> full database
jupyter nbconvert --to notebook --execute --inplace solution.ipynb   # stage-3 notebook (~5 min)
data-factory predict --input new.csv --output scores.csv            # score a new file (same schema)
python -m unittest discover -s tests -t .   # tests (database tests are skipped without a server)
```

`data-factory run` creates the database `dba_stage2` if it does not exist, then builds both
datasets: **nyc311** (fetch → validate → clean → enrich → publish → load), **olist** and
**credit** (fetch → validate → load; for credit, fetch checks the pinned sha256 of the local file). Each load runs in one transaction: all or nothing, safe to repeat.

```bash
data-factory run --dataset olist                 # one dataset
data-factory run --dataset nyc311 --stages load  # one stage
data-factory benchmark --label now --set snapshot # time the dashboard queries
data-factory index-evidence                      # index plans, sizes, write cost
```

## Stage-3 result (from `solution.ipynb`)

| | cost per 1000 applicants |
|---|---|
| approve everyone | 802.1 (CV) |
| one-column rule: reject if any past-due (baseline) | 419.9 (CV), 412.8 (test) |
| HistGradientBoosting, threshold 0.0689 chosen on cost | 371.5 ± 10.6 (5-fold CV), **366.1 (held-out test, seen once)** |

Costs: approving a defaulter 12, rejecting a good applicant 1. Accuracy is reported only to
show why it misleads (approve everyone: 0.9332; chosen model: 0.8040).

## Project layout

| Path | Content |
|---|---|
| `factory.toml` | every setting (sources, pinned checksums, schemas, SQL files) |
| `src/data_factory/` | the package: sources, contracts, validation, sinks, pipelines, CLI |
| `sql/nyc311/`, `sql/olist/`, `sql/credit/` | all DDL (and the view `credit.model_input`), run by the load stage |
| `data/` | raw downloads, intermediate and published files (not in Git) |
| `solution.ipynb`, `report.md` | stage-3 notebook (run after `data-factory run`) and report |
| `docs/stage1/`, `docs/stage2/` | earlier notebooks and reports |
| `docs/decisions.md` | every decision with its evidence |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 10 | configuration error |
| 20–21 | network error / invalid response |
| 30–34 | data error (validation, quarantine limit, source schema change, reference data, checksum mismatch) |
| 40 | input of an earlier stage is missing |
| 50 | database error (connection, constraint, load rolled back) |
| 1 | unexpected error (bug) |
