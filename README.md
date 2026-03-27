# billsum thing (multi agent vs single prompt)

built this for class — theres a langgraph with 3 steps (rough notes on the bill, then a paragraph summary, then a short “what to double check” part) and a baseline thats just one big prompt to the same model. i compare them on the billSum US test jsonl in `billsum_v4_1/`.

for metrics i used rouge and also a really simple word overlap f1 against the reference summary bc we didnt do human ratings. mlflow saves runs in `./mlruns` unless you add `--skip-mlflow`.

---

how to run it

python 3.10 or newer is fine, i used 3.11.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

you can `cp .env.example .env` and put your key there.

if you leave `OPENAI_API_KEY` empty it uses a fake “llm” that just cuts up the text — good for making sure nothing crashes, bad if you want real scores.

with a real key:

```
export OPENAI_API_KEY=...
PYTHONPATH=src python scripts/run_experiment.py --limit 50 --run-name whatever
```

no `--limit` runs the whole test file. with the api that costs money; mock mode is fast but dumb.

---

where i put code

langgraph: `src/pipeline/graph/workflow.py`  
baseline: `src/pipeline/baselines/single_agent.py`  
scoring: `src/pipeline/evaluation/metrics.py`  
main script: `scripts/run_experiment.py`

flags i actually used: `--data` (jsonl path), `--limit`, `--run-name`, `--skip-mlflow`. env stuff is in `.env.example`.

---

mlflow ui

```
mlflow ui --backend-store-uri file:./mlruns
```

opens on some localhost port. each run has a jsonl with per-bill numbers if you need to debug.

---

docker

```
docker build -t bill-eval .
docker run --rm -v "$(pwd)/mlruns:/app/mlruns" bill-eval
```

add `-e OPENAI_API_KEY=...` if not using mock. theres `docker-compose.yml` too.

---

ec2

same idea as laptop but ubuntu + docker + scp the folder. i used tmux for long runs. didnt set up s3 for mlflow, you can point `MLFLOW_TRACKING_URI` somewhere else if your prof wants that.

---

github

my profile: https://github.com/Hanugnajarugumalli

dont commit `.env` or `.venv` or `mlruns` — `.gitignore` already ignores those. the train jsonl is massive so its gitignored too; this project only needs the test split for `run_experiment.py`. bill format is explained in `billsum_v4_1/README.md`.

remote i used:

```
https://github.com/Hanugnajarugumalli/billsum-multi-agent-eval.git
```

create that repo empty on github then:

```
git push -u origin main
```

if the repo name is different:

```
git remote set-url origin https://github.com/Hanugnajarugumalli/OTHER_NAME.git
git push -u origin main
```
