# Generative AI-Based Research Assistant

Completion guide for MST III topic 53. This is a writing and experiment plan for the M.Tech report. It is separate from the CodeSentinel code-review application in this folder.

The assignment (MST III) asks every student to pick one topic, read at least 10–15 papers, build a literature matrix, state a research gap and objectives, implement a system, evaluate it, and submit eight chapters. Topic 53 is **Generative AI-Based Research Assistant**.

Do not copy published scores from PaperQA2 or OpenScholar into your results chapter. Those numbers are background. Chapter 6 may contain only measurements you produce.

Before the viva, open each paper and check the author line against the reference list at the end of this guide.

---

## Working title

**A Citation-Grounded Generative Research Assistant Using Hybrid Retrieval and Claim Verification**

Short name you can use on the title page: Generative AI-Based Research Assistant.

---

## Abstract you can adapt

Students and researchers must read many papers, extract methods and results, and write a literature review. A general large language model answers these questions fluently, but it often cites papers it did not read. This project builds a research assistant that answers only from a fixed set of research papers. The system splits each paper into short passages, retrieves candidates with keyword search and dense embeddings, and asks a generative model to write an answer in which every factual sentence cites a passage. A second check drops any sentence the cited passage does not support. If retrieval is too weak, the system abstains. On a frozen subset of the Qasper dataset, the assistant is compared with the same generator used with no retrieval and with naive dense retrieval. The evaluation uses faithfulness, answer relevancy, context precision, citation accuracy, abstention, and a small human rating. The aim is a reproducible assistant that also fills a literature matrix of method, dataset, metric, and limitation.

Replace the last two sentences after you have real numbers. Example shape, with blanks: “On 80 Qasper questions the proposed system reached faithfulness ___, compared with ___ for naive RAG and ___ for the generator alone. Citation accuracy on a 30-sentence audit was ___.”

---

## Chapter 1 — Introduction

### 1.1 Background

Scientific literature grows faster than a reader can survey it. Question answering over papers is harder than factoid web search: the answer may sit in a methods or results section, and the reader needs the supporting span, not only a sentence that sounds right. Retrieval-augmented generation (Lewis et al., 2020) attaches retrieved text to a generator so the answer can be traced to a source. Later agents such as PaperQA (Lala et al., 2023) and PaperQA2 (Skarlinski et al., 2024) search full-text papers. OpenScholar (Asai et al., 2025) synthesizes cited answers over a store of about 45 million open-access papers and reports that a strong general model hallucinates citations most of the time.

Those systems set the research direction. They also assume infrastructure an M.Tech dissertation cannot rebuild. This project studies the same problem on a closed, public corpus: given papers the system has actually indexed, can a generative assistant answer, cite, and refuse when the evidence is missing?

### 1.2 Problem statement

General-purpose generative models produce research answers without a checkable link to a passage. Naive retrieval-augmented pipelines paste the top passages into the prompt and still emit sentences those passages do not support. Existing research assistants also do not produce the structured literature matrix (paper, method, dataset, metric, limitation) required in an academic review. This project asks whether hybrid retrieval plus an explicit support check improves faithfulness and citation accuracy over a no-retrieval generator and over naive dense retrieval, on a fixed subset of research papers.

### 1.3 Objectives

1. Retrieve passages that contain evidence for an information-seeking question about a research paper.
2. Generate an answer in which every factual sentence cites a retrieved passage.
3. Remove sentences the cited passage does not support, and abstain when retrieval confidence is low.
4. Compare the proposed pipeline with a no-retrieval generator and with naive dense RAG using the same generator, the same questions, and the same metrics.

### 1.4 Scope

Included:

- English NLP research papers already packaged in Qasper.
- A frozen test subset of about 80 questions, chosen before prompt tuning.
- A separate set of 15 questions used only to set the abstention threshold.
- One generator (Gemini) shared by all three systems.
- Local retrieval with BM25 and `all-MiniLM-L6-v2` embeddings.
- An optional literature-matrix view for the demonstration.

Excluded:

- Training a new language model.
- Indexing the open web or tens of millions of papers.
- A claim that the system exceeds PaperQA2 or OpenScholar.
- Advice that a user should treat as medical, legal, or financial fact.

### 1.5 Contribution

The contribution is a small, fully specified pipeline: hybrid retrieval, citation-bearing generation, a support check, and abstention, evaluated against two baselines on a public paper QA dataset, together with a matrix extraction prompt that turns the same evidence into a review row. The contribution is the combination and the controlled comparison, not a new foundation model.

### 1.6 Report organization

Chapter 2 reviews retrieval, scientific language models, research-assistant agents, and evaluation. Chapter 3 fixes the dataset, baselines, and metrics. Chapter 4 describes the architecture and algorithms. Chapter 5 records the implementation. Chapter 6 reports measurements and failure cases. Chapter 7 states what the experiments support and what they do not.

---

## Chapter 2 — Literature review

Write the chapter in four groups, then one gap section. Do not summarize papers in the order you downloaded them.

### Group A — Retrieval and embeddings

Lewis et al. (2020) combine a sequence-to-sequence generator with a dense index of Wikipedia. The generator conditions on retrieved passages, which improves specificity and factuality on knowledge-intensive tasks relative to a parametric model alone. The index is not scientific full text, and the model does not verify each generated claim.

Reimers and Gurevych (2019) train sentence embeddings so that similar sentences lie near each other. That representation is the practical retriever for this project (`all-MiniLM-L6-v2`), with the constraint that inputs longer than 256 tokens are truncated. Karpukhin et al. (2020) show dense passage retrieval can outperform keyword search on open-domain QA when question-passage pairs are available for training. This project does not train a new retriever; it uses a frozen embedding model and BM25 together, because keyword search still helps when the question contains a method name or a metric.

Beltagy, Lo, and Cohan (2019) pretrain SciBERT on scientific text. Cohan et al. (2020) learn SPECTER embeddings from the citation graph so that related papers are nearby. Both matter as background: scientific wording differs from Wikipedia. This project does not fine-tune SciBERT or SPECTER. Say that clearly so the examiner does not think you claimed a new encoder.

### Group B — Hallucination and factual evaluation

Ji et al. (2023) survey hallucinations in natural language generation: output that is fluent but unsupported. Min et al. (2023) break a long answer into atomic facts and score what fraction are supported by a knowledge source (FActScore). Es et al. (2024) define RAGAS so a retrieval-augmented system can be scored without a single gold paragraph: faithfulness of the answer to the context, answer relevancy, and context precision and recall. Gao et al. (2023) survey RAG designs and treat faithfulness as an open problem. These four papers justify your metrics. They are not systems you reimplement in full. You use the ideas: context-grounded scoring, plus a manual audit of citations.

### Group C — Agents that read papers

Dasigi et al. (2021) release Qasper: 5,049 questions over 1,585 NLP papers. Practitioners wrote questions from the title and abstract; other practitioners answered from the full text and marked evidence. Models that were strong on other QA tasks trailed humans by at least 27 F1 when they had to read the whole paper. Qasper is your dataset because the questions are information-seeking and the evidence spans are public.

Lala et al. (2023) present PaperQA, an agent that retrieves over full-text articles, scores passages, and answers with retrieval-augmented generation. They introduce LitQA for questions that need the full text. Skarlinski et al. (2024) extend this line: PaperQA2 searches papers, reranks evidence, and is reported to match or exceed subject-matter experts on literature search, summarization, and contradiction detection. Asai et al. (2025) present OpenScholar, which retrieves from about 45 million open-access papers, reranks, generates cited answers, and revises them with self-feedback. On their benchmark, a smaller open model with this pipeline is more correct than GPT-4o, and GPT-4o hallucinates citations 78–90% of the time. Experts preferred OpenScholar answers over expert-written answers more often than they preferred GPT-4o answers.

Yao et al. (2023) describe ReAct, which interleaves reasoning with tool use. Asai et al. (2024) describe Self-RAG, which learns when to retrieve and when to critique a draft using special tokens. Both support your design choice to add an explicit check after generation. You will not train reflection tokens. Your check is a second prompted decision: supported or not supported.

### Group D — What you will not claim to extend

PaperQA2 and OpenScholar are the closest systems. Your chapter must say how you differ:

- They search a large external literature. You index only the papers attached to your frozen Qasper subset.
- They optimize long scientific synthesis. You also emit a short literature-matrix row, because that is the academic task in this assignment.
- You compare three controlled variants of one generator. You do not compare against their published leaderboard numbers.

### 2.5 Research gap

Write this paragraph in your own words, close to this meaning:

Published generative research assistants show that retrieval and citation can make scientific answers more reliable than a general chat model. The same work depends on large paper indexes, agent tooling, or extra training. A naive retrieve-then-generate pipeline still has no separate test that each sentence is supported by the passage it cites, and it does not produce the structured review matrix used in a dissertation literature review. The gap addressed here is a reproducible, closed-corpus assistant that (i) retrieves with both keywords and embeddings, (ii) cites a passage for every factual sentence, (iii) drops unsupported sentences or abstains, and (iv) can list method, dataset, metric, and limitation from the same evidence.

### Literature matrix

Copy this table into the report, then rewrite the “Limitation” cell after you read the paper. If you cannot obtain the full text, use the publisher PDF (ACL Anthology, NeurIPS, Nature, arXiv) and say so.

| # | Paper | Year | Method | Data | Finding you cite | Limitation for your gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Lewis et al. | 2020 | RAG: generator plus dense Wikipedia memory | Knowledge-intensive NLP tasks | Retrieved text makes generation more specific and factual than a parametric model | Not scientific PDFs; no per-claim citation check |
| 2 | Reimers and Gurevych | 2019 | Siamese sentence embeddings | Semantic textual similarity | Usable dense retrieval without training a new model | General domain; 256-token limit on MiniLM |
| 3 | Beltagy et al. | 2019 | SciBERT pretraining | Scientific text | Domain language differs from general BERT | Encoder only; does not answer questions |
| 4 | Karpukhin et al. | 2020 | Dense passage retrieval | Open-domain QA | Dual-encoder retrieval can beat BM25 when trained | You have no new labeled retrieval set |
| 5 | Cohan et al. | 2020 | SPECTER citation embeddings | Scientific papers | Related-paper retrieval from citations | Does not answer from a section span |
| 6 | Dasigi et al. | 2021 | Qasper annotation protocol | 1,585 NLP papers, 5,049 questions | Evidence-backed paper QA; models lag humans by ≥27 F1 | A dataset, not an assistant |
| 7 | Ji et al. | 2023 | Hallucination survey | NLG literature | Fluent text can be unsupported | No pipeline for papers |
| 8 | Yao et al. | 2023 | ReAct tool use | Multi-step tasks | Reasoning plus tools beats act-only or reason-only | Not a citation checker |
| 9 | Min et al. | 2023 | FActScore atomic facts | Long-form biography answers | Factual precision can be scored fact by fact | Needs a support source you must supply |
| 10 | Lala et al. | 2023 | PaperQA agent | Science QA, LitQA | Full-text RAG agent for papers | No coursework literature matrix |
| 11 | Gao et al. | 2023 | RAG survey | Published RAG systems | Design space of chunking, retrieval, generation | Faithfulness left open |
| 12 | Es et al. | 2024 | RAGAS metrics | RAG pipelines | Faithfulness and relevancy without one gold paragraph | Metric library, not the assistant |
| 13 | Asai et al. | 2024 | Self-RAG | QA and long-form generation | Critique tokens improve factuality | Requires training special tokens |
| 14 | Skarlinski et al. | 2024 | PaperQA2 agent | LitQA2 and expert comparison | Agent matches or exceeds experts on literature tasks | Broad search, not a frozen student corpus |
| 15 | Asai et al. | 2025 | OpenScholar | ScholarQABench; 45M papers | Cited synthesis; GPT-4o citation hallucination 78–90% | Index and training exceed this project |

Minimum for the assignment is 10–15 papers. Fifteen are listed. Add a tool citation if you use it in a way that affects the method (for example the Sentence-Transformers model card, or the Gemini API documentation) under references, not inside this matrix.

---

## Chapter 3 — Research methodology

### 3.1 Research design

The study is experimental and comparative. The independent variable is the pipeline: no retrieval (B0), naive dense RAG (B1), or hybrid retrieval with claim verification and abstention (proposed). The generator, temperature, questions, and corpus stay fixed. The dependent variables are the metrics in section 3.5.

This is not a user study of a deployed product. A short human rating (section 3.6) checks whether the automatic citation score agrees with readers.

### 3.2 Dataset

Use Qasper (Dasigi et al., 2021), available as the Hugging Face dataset `allenai/qasper`. Each item has a paper and questions with answers and evidence.

Protocol:

1. Download the dataset once and record the date and the split you used.
2. From the validation or test split, sample 80 questions at random with a fixed seed (record the seed). Prefer questions that have a short answer and at least one evidence span.
3. If the split marks some questions as unanswerable from the paper, include about 15 of those inside the 80 so abstention can be measured. If it does not, build 15 “unanswerable” questions yourself by asking for a result the paper does not report, and label them by hand. Keep them separate in the table.
4. Draw 15 further questions as a development set. Use them only to choose the abstention threshold. Never report development numbers as the test result.
5. Save `questions.jsonl` with question id, paper id, question, gold answer, and evidence. Commit that file before you change prompts.

The corpus for retrieval is the full text of the papers that appear in this subset, not the whole internet. State the paper count after sampling.

Preprocessing:

- Extract title, abstract, and section text.
- Normalize whitespace. Keep section names.
- Chunk to about 200 tokens with an overlap of 40 tokens. MiniLM truncates beyond 256 word-pieces, so longer chunks silently lose their ending. That sentence belongs in the report.
- Store chunk id, paper id, section, and text.

You do not need a separate data-cleaning chapter for missing values. Papers are documents. Describe extraction failures (a section that did not parse) and how many chunks you dropped.

### 3.3 Systems

**B0 — LLM only.** Prompt contains the question and an instruction to answer from knowledge. No paper text. Ask it to cite anyway, so you can show unsupported citations.

**B1 — Naive RAG.** Embed chunks with `all-MiniLM-L6-v2`. Retrieve the top 5 by cosine similarity. Put them in the prompt. Generate an answer. No verifier and no abstention.

**Proposed.** BM25 and MiniLM each return 20 chunks. Fuse the ranks (reciprocal rank fusion). Keep the best 5 after fusion. Generate an answer that cites chunk ids. For each factual sentence, a second call sees only that sentence and its cited chunk and returns SUPPORTED or NOT_SUPPORTED. Drop NOT_SUPPORTED sentences. If the best fused score is below the threshold chosen on the development set, return “Insufficient evidence in the indexed papers.”

Ablations, same test questions:

- Proposed without the support check.
- Dense retrieval only (no BM25).
- BM25 only (no dense retrieval).

### 3.4 Generator settings

Use one Gemini model for every condition. Record the exact model id from the API. Set temperature to 0 for generation and for the support check. If the free tier rate-limits you, log retries and the final latency. Do not switch models halfway through the test set. If you must switch, rerun every system.

### 3.5 Metrics

| Metric | Definition for this project | Compared across |
| --- | --- | --- |
| Faithfulness | RAGAS score: claims in the answer follow from the retrieved context | B1 and proposed. B0 has no context; report it as not applicable, not as zero, unless you define a substitute |
| Answer relevancy | RAGAS score: answer addresses the question | B0, B1, proposed |
| Context precision | Retrieved chunks match Qasper evidence spans, or RAGAS context precision | B1 and proposed |
| Citation accuracy | Fraction of cited sentences whose chunk truly supports the sentence | All systems that cite. Human audit of 30 sentences as the primary number; model verifier as secondary |
| Abstention rate | Fraction of questions answered with the refusal string | Proposed, split by answerable vs unanswerable |
| Latency | Mean seconds per question, including retrieval and API | All three |

For B0, faithfulness against a retrieved context is undefined. Say that. Score B0 citation accuracy by checking whether the cited paper id exists in the corpus and whether any chunk supports the sentence. Expect this number to be low. That contrast is the point of the baseline.

### 3.6 Human check

Sample 20 proposed answers (include at least 3 refusals). Two readers who are not you score:

- Correctness, 1 to 5, against the gold Qasper answer.
- Citation usefulness, 1 to 5: could they find the support from the citation?

Report the mean of the two readers. If you know how, also report Cohen’s kappa for a binary “supported / not supported” on the 30-sentence audit. If the two readers disagree a lot, say so. Disagreement is a result.

### 3.7 Threats to validity

- Qasper is NLP papers. Results do not automatically transfer to medicine or law.
- The support check uses the same model family as the generator, so it can share blind spots. The human audit is there to measure that.
- Eighty questions is a sample, not the full Qasper test set. Report the sample size in every table caption.
- API answers can change if the provider updates the model. Pin the model id and the date of the run.
- RAGAS uses a judge model. Treat it as an instrument, and do not let it replace the citation audit.

---

## Chapter 4 — System analysis and design

### 4.1 Architecture

Modules, in order:

1. **Corpus loader** reads Qasper papers for the frozen ids.
2. **Chunker** writes passages with metadata.
3. **Index** builds a BM25 index and a FAISS index of MiniLM vectors.
4. **Retriever** runs both, fuses ranks, and returns five chunks.
5. **Generator** calls Gemini with the question and the five chunks.
6. **Verifier** calls Gemini once per factual sentence.
7. **Abstain gate** applies the development threshold before the answer is shown.
8. **Matrix writer** is a separate prompt over the same five chunks. It does not affect the QA metrics. Keep it out of the latency number for QA, or report it as its own column.
9. **Logger** stores the question id, retrieved ids, raw answer, verified answer, and timings.

Draw three figures:

- A block diagram of the nine modules.
- A sequence diagram for one question: user, retriever, generator, verifier, response.
- A flowchart with the abstain branch and the drop-sentence branch.

UML for this project means those three diagrams plus a short class or module list. You do not need a 20-class enterprise diagram.

### 4.2 Algorithm A — hybrid retrieve

Input: question q, chunk set C, cutoff k = 5.

1. Score every chunk with BM25. Keep the top 20 ids, best first.
2. Embed q with MiniLM. Score chunks by cosine similarity. Keep the top 20.
3. Reciprocal rank fusion: for each chunk, add 1 / (60 + rank) from each list it appears in. Rank starts at 1.
4. Sort by fused score. Return the first 5, with their scores and metadata.
5. If the best score is below threshold t, mark the question for abstention. Still return the chunks so the log shows what was found.

Choose t on the 15 development questions: pick the value that refuses most unanswerable development items without refusing most answerable ones. Describe the rule in one paragraph and give the final t.

### 4.3 Algorithm B — verify then answer

Input: question q, chunks c1..c5, or the abstain flag.

1. If the abstain flag is set, return “Insufficient evidence in the indexed papers.” and the chunk ids. Stop.
2. Prompt the generator: answer only from the chunks; every factual sentence ends with a chunk id in brackets; if the chunks do not contain the answer, say so.
3. Split the answer into sentences.
4. For each sentence that contains a citation, call the verifier with that sentence and the cited chunk only. Temperature 0. Label SUPPORTED or NOT_SUPPORTED.
5. Delete NOT_SUPPORTED sentences.
6. If no sentence remains, return the abstain message.
7. Log raw answer, labels, and final answer.

### 4.4 Literature matrix prompt

Separate from Algorithm B. Input is a paper’s retrieved chunks or the abstract plus introduction plus conclusion. Output is one row: paper title, year, method, dataset, reported metric, limitation stated by the authors. Instruct the model to write “not stated” rather than guess. This row is a demo of objective support for chapter 2. Do not mix its quality into the Qasper QA table unless you label a separate table.

### 4.5 Why these choices

- Hybrid retrieval: BM25 matches rare names (a model name, a dataset). Dense retrieval matches paraphrases. Either alone is an ablation, not a hidden extra.
- Short chunks: they fit MiniLM and give a citation the reader can check.
- A second call for support: it is inspectable. A single prompt that says “be accurate” is not a method.
- Abstention: Qasper-style reading includes the possibility that the paper does not say. A research assistant that always answers has not finished the task.

---

## Chapter 5 — Implementation

Suggested stack, all ordinary and explainable in a viva:

| Piece | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11 | Same ecosystem as the datasets and RAGAS |
| Papers | `datasets` library, `allenai/qasper` | Public, cited, reproducible |
| PDF fallback | `pypdf` only if you add your own PDFs for the demo | Qasper already has text |
| Keywords | `rank-bm25` | Simple baseline the examiner knows |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` | Runs on CPU |
| Vector index | `faiss-cpu` | Exact search is fine at this corpus size |
| Generator | `google-genai`, one Gemini model | You can describe one API |
| Metrics | `ragas` | Matches Es et al., 2024 |
| Demo UI | Streamlit | One screen: question, answer, cited chunks, matrix row |
| Tests | `pytest` for chunking, fusion, and the abstain branch | Shows the pipeline is not only a prompt |

Environment:

- Put the API key in `.env`. Do not paste it into the report or the slides.
- Pin versions in `requirements.txt` and copy that list into an appendix.

Demo script for the viva, two minutes:

1. Ask a question whose answer is in a methods section. Show the cited chunk highlighted.
2. Ask something the paper does not contain. Show the refusal.
3. Open the log row so the examiner sees raw answer versus sentences that were dropped.

Screenshots for the report: the three-system result table, one supported answer, one refusal, one matrix row.

---

## Chapter 6 — Results and discussion

Create these tables after the runs. Leave the cells blank until then. Inventing a faithfulness score is a reason to fail the viva.

### Table 6.1 Main comparison

Caption must include: model id, date, question count, seed.

| System | Answer relevancy | Faithfulness | Context precision | Citation accuracy (human, n=30) | Abstention rate | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B0 LLM-only | | n/a | n/a | | | |
| B1 Naive RAG | | | | | 0 by design | |
| Proposed | | | | | | |

### Table 6.2 Ablation

| Variant | Faithfulness | Citation accuracy | Abstention rate |
| --- | --- | --- | --- |
| Proposed | | | |
| No support check | | | |
| Dense only | | | |
| BM25 only | | | |

### Table 6.3 Human rating

| Question type | n | Mean correctness (1–5) | Mean citation usefulness (1–5) |
| --- | --- | --- | --- |
| Answerable | | | |
| Refusal cases | | | |

### How to discuss the numbers

Write one paragraph per table.

- If proposed faithfulness and citation accuracy are higher than B1, and B0 citation accuracy is lower, objective O4 is supported. Say “supported on this 80-question subset,” not “the system is state of the art.”
- If faithfulness rises but relevancy falls, the verifier is deleting useful sentences. Show two examples. That is a limitation, and it belongs in chapter 7.
- If dense-only beats hybrid, report it. A negative ablation is still a result. Explain with two questions: one where a rare token saved BM25, or one where BM25 added noise.
- If the model verifier and the human audit disagree, trust the human number in the abstract and explain the disagreement.
- Pick five errors: wrong section retrieved, citation to a neighbor chunk, refusal on a question that was answerable, a matrix cell that said “not stated” when the abstract had the metric, and a sentence that was fluent and still unsupported.

Compare with published work only qualitatively. You may write: “OpenScholar reports very high citation hallucination for a general model without its retrieval loop. B0 in this study is the same phenomenon on a smaller corpus.” You may not write: “Our system outperforms OpenScholar” unless you ran OpenScholar on your questions, which you should not promise.

### Graphs

- Bar chart of citation accuracy for B0, B1, and proposed.
- Bar chart of the ablation.
- One histogram of latency if the API variance is large.

Label axes with the metric and the unit. Caption: source is your run, with the date.

---

## Chapter 7 — Conclusion and future scope

Draft the logic now and fill the blanks after chapter 6.

This dissertation built a generative research assistant that answers from a closed set of Qasper papers. Hybrid retrieval, cited generation, a support check, and abstention were compared with a no-retrieval generator and with naive dense RAG. On the frozen subset, citation accuracy was ___ for the proposed system, ___ for naive RAG, and ___ for the generator alone. Faithfulness was ___ against ___ . Objective O1 is met to the extent context precision reached ___ . Objective O2 is met when every remaining factual sentence carries a chunk id. Objective O3 is met when unsupported sentences are removed and low-score questions are refused; the abstention rate was ___ on unanswerable items and ___ on answerable items. Objective O4 is met only if the proposed row is better on faithfulness and citation accuracy without a collapse in relevancy.

Limitations to keep even if the numbers look good:

- NLP papers only.
- Judge-model bias in the support check and in RAGAS.
- Sample of 80 questions.
- No new retriever training.
- Matrix extraction was demonstrated, not given a full gold-standard score, unless you add one.

Future work, two items are enough:

- Rerank the five chunks with a cross-encoder if a GPU is available.
- Repeat the same protocol on a second domain with openly licensed papers, without changing the metrics.

---

## Chapter 8 — References (IEEE)

Check page ranges on the PDF before final submission. Entries below match the papers named in this guide.

[1] P. Lewis *et al.*, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 9459–9474.

[2] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.

[3] I. Beltagy, K. Lo, and A. Cohan, “SciBERT: A pretrained language model for scientific text,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 3615–3620.

[4] V. Karpukhin *et al.*, “Dense passage retrieval for open-domain question answering,” in *Proc. EMNLP*, 2020, pp. 6769–6781.

[5] A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. S. Weld, “SPECTER: Document-level representation learning using citation-informed transformers,” in *Proc. ACL*, 2020, pp. 2270–2282.

[6] P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner, “A dataset of information-seeking questions and answers anchored in research papers,” in *Proc. NAACL-HLT*, 2021, pp. 4599–4610.

[7] Z. Ji *et al.*, “Survey of hallucination in natural language generation,” *ACM Comput. Surv.*, vol. 55, no. 12, Art. 248, Mar. 2023.

[8] S. Yao *et al.*, “ReAct: Synergizing reasoning and acting in language models,” in *Proc. ICLR*, 2023.

[9] S. Min *et al.*, “FActScore: Fine-grained atomic evaluation of factual precision in long form text generation,” in *Proc. EMNLP*, 2023, pp. 12076–12100.

[10] J. Lála, O. O’Donoghue, A. Shtedritski, S. Cox, S. G. Rodriques, and A. D. White, “PaperQA: Retrieval-augmented generative agent for scientific research,” arXiv:2312.07559, 2023.

[11] Y. Gao *et al.*, “Retrieval-augmented generation for large language models: A survey,” arXiv:2312.10997, 2023.

[12] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAS: Automated evaluation of retrieval augmented generation,” in *Proc. 18th Conf. European Chapter of the Association for Computational Linguistics: System Demonstrations*, 2024.

[13] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning to retrieve, generate, and critique through self-reflection,” in *Proc. ICLR*, 2024.

[14] M. D. Skarlinski *et al.*, “Language agents achieve superhuman synthesis of scientific knowledge,” arXiv:2409.13740, 2024.

[15] A. Asai *et al.*, “Synthesizing scientific literature with retrieval-augmented language models,” *Nature*, 2025, doi: 10.1038/s41586-025-10072-4.

---

## How the assignment will be marked

MST III section 8 lists nine parameters. This is where each one is answered.

| Parameter | Where you satisfy it |
| --- | --- |
| Problem and objectives | Chapter 1, objectives O1–O4 |
| Literature review of 10–15 papers | Chapter 2 and the 15-row matrix |
| Research gap | Section 2.5 |
| Methodology and justification | Chapter 3, especially why hybrid retrieval and a separate support check |
| Dataset and preprocessing | Qasper subset, chunk size justified by the 256-token encoder |
| Implementation | Chapter 5 and the running demo |
| Metrics | Section 3.5 and Tables 6.1–6.3 |
| Comparison with other methods | B0, B1, ablations, and a qualitative note on PaperQA2 and OpenScholar |
| Report and presentation | Eight chapters, three figures, two viva cases (one citation, one refusal) |

---

## Ten-week schedule

| Week | Done when |
| --- | --- |
| 1 | Title and objectives frozen. All 15 PDFs saved. |
| 2 | Matrix filled in your own words. Gap paragraph drafted. |
| 3 | Chapter 2 full draft. |
| 4 | `questions.jsonl` frozen. Chapters 3 and 4 drafted. Diagrams drawn. |
| 5 | Index builds. B0 and B1 run on 5 questions end to end. |
| 6 | Proposed pipeline, including verifier and abstain, runs on the development set. Threshold t chosen. |
| 7 | Full test run. RAGAS table. 30-sentence human audit. |
| 8 | Chapter 6 with error cases. No empty metric cells. |
| 9 | Chapters 1, 5, and 7 revised to match the tables. References checked on the publisher sites. |
| 10 | Streamlit demo rehearsed. Slides use the same numbers as chapter 6. |

---

## Viva questions to prepare

1. Why is a generative model not enough for a research assistant?
2. Why Qasper instead of a set of PDFs you chose after seeing the answers?
3. Why chunks of about 200 tokens?
4. What does reciprocal rank fusion do that a simple average of scores does not?
5. Why is the support check a separate call?
6. When should the system refuse?
7. Why is B0 faithfulness marked “not applicable”?
8. What would a higher faithfulness but a lower relevancy mean?
9. Why can’t you claim you beat OpenScholar?
10. What fails if the verifier and the generator share the same blind spot?

---

## What this guide does not do

It does not invent experimental results. It does not replace reading the fifteen papers. The CodeSentinel application in this repository reviews source code with a multi-agent workflow. That is a different system from topic 53. Use this guide for the generative research assistant report. The next build, when you want it, is the Qasper pipeline in a new folder: loader, chunker, hybrid index, Gemini answer, verifier, and the three-system evaluation script.
