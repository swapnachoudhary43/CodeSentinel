# Implementation audit

Recorded after `python -m pytest -q` reported 68 passed, before the Qasper run. Later tests cover the evaluation code and the suite count is higher.

## Working before this extension

- `research_assistant` answers from a closed corpus, cites chunk ids, drops unsupported sentences in the proposed system, and abstains when hybrid retrieval is empty or a threshold is set.
- B0 calls the generator with no passages. B1 retrieves dense top-k and does not verify. Proposed uses BM25 plus the hashing embedder, reciprocal rank fusion, and a support check.
- Gemini `gemini-3.5-flash-lite` answered the fixture fusion question with a supported citation and abstained on the France question because the notes do not contain it.
- The command-line tool, sample corpus, and 25 research-assistant tests exercised that behavior. The other 43 tests belong to CodeSentinel and were left in place.

## Gaps found

- The Qasper helper expected a columnar Hugging Face row. Official v0.3 is a dict of arXiv id to paper, with `full_text` as a list of sections and `qas` as a list of questions. `datasets` 5.1 also refuses the old dataset script.
- No frozen question file, no saved model outputs, and no metric table existed. Fixture runs were not a benchmark.
- B1 with zero hits was passed to the generator as "no passages," the same signal as B0. The result is now still labeled dense retrieval, and the prompt says retrieval returned nothing.
- The free-tier Gemini quota is 15 requests per minute for `gemini-3.5-flash-lite`. A first full run hit HTTP 429. Those failures are kept in the jsonl until a later run replaces them. Calls are now spaced by 4.5 seconds, and a 429 waits about 65 seconds.
- Dense retrieval in this repository is `HashingEmbedder`, not a neural model. The experiment config says so. It was not relabeled as MiniLM.
- Human ratings do not exist. The sheet is blank on purpose.
- There is no claim of a significance test or of beating PaperQA2 or OpenScholar.

## Files changed rather than replaced

- `pipeline.py` and `llm.py` keep B0, B1, and proposed. Ablations are a separate function. Empty B1 retrieval no longer pretends no retrieval ran. Gemini can cache and retry.
- `cli.py` accepts a papers file and the ablation names.
- `types.py` adds verification records with a default, so older result objects still construct.

## Files added

- `qasper_data.py`, `metrics.py`, `experiment.py`, `reporting.py`
- `experiments/dataset` for the official tarball, manifest, and frozen questions
- `experiments/runs` for raw outputs
- `experiments/reports` for the audit, literature matrix, and generated chapter 6
