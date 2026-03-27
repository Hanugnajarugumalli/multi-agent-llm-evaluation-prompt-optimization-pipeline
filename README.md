# BillSum multi-agent summarization project

Final project for comparing a LangGraph setup (3 steps: notes → summary → extra “decision” blurb) vs a boring one-shot OpenAI prompt on the same bills. Scoring is ROUGE + a dumb word-overlap score I wrote because we didn’t have time to run human eval. MLflow logs runs under `./mlruns` if you don’t pass `--skip-mlflow`.

Data is the US test jsonl in `billsum_v4_1/` — thousands of rows so you’re not evaluating on like 10 examples.

## Setup

Need Python 3.10+ (I used 3.11). From the folder:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `cp .env.example .env` if you want to put your API key in a file instead of exporting it every time.

**No key:** leave `OPENAI_API_KEY` blank and it uses a fake LLM that just chops the text up so the pipeline still runs (good for checking you didn’t break imports). Not useful for real numbers obviously.

**With OpenAI:**

```
export OPENAI_API_KEY=...
PYTHONPATH=src python scripts/run_experiment.py --limit 50 --run-name try1
```

No `--limit` = whole test file (takes a while with the API, mock mode is quick).

## Where stuff is

- `src/pipeline/graph/workflow.py` — the LangGraph
- `src/pipeline/baselines/single_agent.py` — baseline
- `src/pipeline/evaluation/metrics.py` — ROUGE + overlap F1
- `scripts/run_experiment.py` — actually runs everything

Main script flags: `--data` (path to jsonl), `--limit` (first N rows), `--run-name` (shows up in MLflow), `--skip-mlflow` (only print JSON to stdout).

Env vars are in `.env.example` — model name, max chars to truncate bills, mlflow experiment name, etc.

## MLflow

```
mlflow ui --backend-store-uri file:./mlruns
```

Then open whatever port it says (usually 5000). There’s also a jsonl artifact per run with per-bill scores if something looks wrong.

## Docker

```
docker build -t bill-eval .
docker run --rm -v "$(pwd)/mlruns:/app/mlruns" bill-eval
```

Pass `-e OPENAI_API_KEY=...` if you’re not using mock mode. There’s also `docker-compose.yml` if you want the volume mounted without typing it every time.

## Running on EC2

Basically: Ubuntu instance, install docker, copy the repo over, same docker commands. Use tmux or screen for long jobs. I didn’t bother with S3 for mlflow but you can change `MLFLOW_TRACKING_URI` if your class requires it.

## Resume / portfolio

If you’re writing this up: multi-agent code is under `graph/`, baseline under `baselines/`, metrics under `evaluation/`. Don’t copy fake % improvements from anywhere — run it yourself and put whatever your run actually got.

Dataset format is described in `billsum_v4_1/README.md` (it’s the official BillSum jsonlines).

## What to put on GitHub

**Include in the repo:** `README.md`, `requirements.txt`, `.env.example`, `.gitignore`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, everything under `src/`, `scripts/`, and `billsum_v4_1/README.md` plus the jsonl files you actually use (the default eval only needs `us_test_data_final_OFFICIAL.jsonl`).

**Do not commit:** `.env` (API keys), `.venv/` or `venv/`, `mlruns/` (MLflow output — regenerates when you run experiments).

**Train file:** `us_train_data_final_OFFICIAL.jsonl` is huge (~213MB) and hits GitHub’s file size limit, so it’s listed in `.gitignore`. This project’s script uses the **test** split only; if you need train data, download it from the [BillSum source](https://github.com/FiscalNote/BillSum) separately.

**Upload** — GitHub profile: [github.com/Hanugnajarugumalli](https://github.com/Hanugnajarugumalli). This repo is already on branch `main` with commits; `origin` is set to:

`https://github.com/Hanugnajarugumalli/billsum-multi-agent-eval.git`

1. On GitHub (logged in as you): [**New repository**](https://github.com/new) → name it **`billsum-multi-agent-eval`** → create **empty** (no README / .gitignore / license).
2. In the project folder:

```
git push -u origin main
```

If you used a different repo name, run:

```
git remote set-url origin https://github.com/Hanugnajarugumalli/OTHER_REPO_NAME.git
git push -u origin main
```
