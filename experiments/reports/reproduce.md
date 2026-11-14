# How to reproduce the Qasper run

From the project root, with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The official dev archive is already at `experiments/dataset/qasper-train-dev-v0.3.tgz`. To rebuild the frozen 80-question subset:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from research_assistant.qasper_data import write_freeze; write_freeze(Path('experiments/dataset/qasper-train-dev-v0.3.tgz'), Path('experiments/dataset'), seed=20261107)"
```

To repeat the model run, set `GEMINI_API_KEY` in `.env`. Completed rows are skipped. Rows that stored an API error are retried.

```powershell
.\.venv\Scripts\python.exe -m research_assistant.experiment --output experiments/runs
.\.venv\Scripts\python.exe -m research_assistant.reporting
```

Ask one question from the frozen papers:

```powershell
.\.venv\Scripts\python.exe -m research_assistant --papers experiments/dataset/papers.json --question "YOUR QUESTION" --system proposed --gemini
```

Without `--gemini`, the reply is an extractive quote of the top passage, not a Gemini answer.

The calibration file is `experiments/dataset/calibration_questions.jsonl`. It is not passed to the experiment command. The threshold in `experiments/runs/experiment_config.json` stays null.
