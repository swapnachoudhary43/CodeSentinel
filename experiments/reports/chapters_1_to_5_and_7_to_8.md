# Generative AI-Based Research Assistant

The measured comparison is in `chapter6_results.md`, which is written from `experiments/runs/metric_results.json`. This file does not restate those scores.

## Chapter 1. Introduction

Researchers ask questions that are answered in the body of a paper, not only in the abstract. A general language model can answer without showing the passage it used. This project studies a smaller, closed-corpus assistant: given the papers that were indexed, retrieve passages, require a chunk id on a factual sentence, drop a sentence the cited passage does not support, and abstain when retrieval finds nothing or the check deletes every sentence.

The objectives are:

1. Retrieve passages for an information-seeking question.
2. Generate an answer whose kept factual sentences cite a retrieved chunk.
3. Remove sentences the cited chunk does not support, and abstain when no supported sentence remains.
4. Compare that pipeline with the same generator using no retrieval, and with dense retrieval without a support check, on one frozen question list.

The run uses the Qasper validation split, version 0.3. It does not use the hidden test split. It does not claim a better score than PaperQA2 or OpenScholar.

## Chapter 2. Literature review

The 15-paper matrix is `literature_matrix.md`. The gap used for this dissertation is narrower than those systems. General models can emit citations that were not retrieved. Naive retrieval pastes passages into a prompt and does not check each sentence. PaperQA2 and OpenScholar search very large literature collections. This project indexes only the papers attached to an 80-question Qasper slice and records every system output.

## Chapter 3. Research methodology

The independent variable is the pipeline. B0 receives the question and no passages. B1 receives the top five hashing-cosine chunks and does not verify sentences. The proposed system fuses BM25 and that hashing rank with reciprocal rank fusion (constant 60), generates with citations, and runs a second call that must begin with SUPPORTED or the sentence is dropped.

The same generator, temperature 0, questions, and paper collection are used for every system. The collection is the set of validation papers that contain at least one selected evaluation question. A question can therefore retrieve a passage from a different paper in the slice. That is intentional for a collection assistant, and it is not the original single-paper Qasper protocol.

Eighty evaluation questions and fifteen calibration questions were drawn with seed 20261107. The split is stratified on the first annotator's unanswerable flag. The calibration questions were not scored and were not used to set a threshold. The threshold argument is null, so score-based abstention is off. Abstention still happens when retrieval is empty or the verifier leaves no sentence.

Reference text is the first annotator in the v0.3 file order. Token F1 also takes the maximum over answerable annotators. Evidence strings are the first annotator's `evidence` field. Figure images were not downloaded. Captions are indexed.

Metrics, with denominators, are defined in `research_assistant/metrics.py` and repeated in chapter 6. RAGAS was not run. It would add another judge model and a dependency the scoring script does not use. The verifier score is reported separately and is not called ground truth.

Threats: the slice is 80 validation questions, not the full dev set or the official test set. The dense channel is not a trained neural retriever. The verifier is the same model family as the generator. No significance test is applied.

## Chapter 4. System analysis and design

The indexed path is: load papers, split each section into 200-word windows with a 40-word overlap, build a BM25 index and a hashed-token cosine index, retrieve, generate, and, for the proposed system only, verify.

B0 does not search. B1 searches in dense mode and keeps the raw generation. The proposed system searches in hybrid mode. If that search returns no chunk, it abstains before the generator is called. Otherwise each sentence must cite a retrieved chunk id. A missing citation or a citation outside the retrieved set is dropped. A second model call labels the pair SUPPORTED or NOT_SUPPORTED. Any other first token counts as not supported.

Ablations, which do not replace the three systems:

- No verifier: hybrid retrieval, raw answer kept.
- BM25 only: keyword list only, verifier kept.
- Dense only: hashing cosine only, verifier kept.

Empty retrieval still abstains in those ablations, because that gate belongs to retrieval rather than to the verifier.

## Chapter 5. Implementation

The package is `research_assistant`. CodeSentinel's review workflow is unchanged. The generator is Gemini through `google-genai`, model id recorded in the experiment config, temperature 0. The API key stays in `.env`. Responses are cached by a hash of the model id and the prompt. Calls are spaced by 4.5 seconds because the free-tier limit observed in this project is 15 requests per minute. A 429 response is stored if retries are exhausted, and a later run replaces that error row instead of deleting it quietly.

The command-line tool still runs the two fixture notes. `--papers` points it at the frozen Qasper papers. `--gemini` is required for a model answer. Without it, the tool quotes the top passage and says so.

## Chapter 7. Conclusion and future scope

The implementation meets the pipeline objectives: three systems share one generator, citations are checked only in the proposed system and in the ablations that keep the verifier, and refusals are recorded. Whether those checks improve token F1 or evidence recall on this slice is a measured result and is stated only in chapter 6.

The run does not establish a new state of the art. The dense retriever is a hash of tokens, the question list is a validation subset, human reviewers have not filled the score sheet, and no comparison was executed against PaperQA2 or OpenScholar.

Future work that this repository does not claim to have done: swap the hashing embedder for a neural model such as `all-MiniLM-L6-v2`, fit an abstention threshold on the calibration split only, and collect the two-reviewer sheet.

## Chapter 8. References

The reference list for the background papers is the link column of `literature_matrix.md`. The dataset citation is Dasigi et al., NAACL 2021, DOI 10.18653/v1/2021.naacl-main.365. Qasper version 0.3 was read from `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`. The Hugging Face card for `allenai/qasper` states the license CC BY 4.0. The README inside that tarball does not repeat the license line.
