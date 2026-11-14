import json

from research_assistant.corpus import load_sample_papers
from research_assistant.experiment import run_experiment
from research_assistant.llm import ExtractiveModel, answer_prompt
from research_assistant.metrics import evidence_covered, score_run, token_f1
from research_assistant.pipeline import answer, answer_ablation
from research_assistant.qasper_data import (
    normalize_qasper_answer,
    paper_from_official,
    questions_from_official_paper,
    select_subset,
)
from research_assistant.retrieve import HybridIndex
from research_assistant.types import ABSTAIN_MESSAGE
from tests.test_research_assistant import (
    FUSION_QUESTION,
    ScriptedModel,
    UNKNOWN_QUESTION,
    sample_index,
)


def test_official_qasper_shape_keeps_null_yes_no_and_evidence():
    paper = {
        "title": "Schema Fixture",
        "abstract": "A short abstract about retrieval.",
        "full_text": [
            {
                "section_name": "Method",
                "paragraphs": ["The retriever uses BM25.", "A second paragraph."],
            }
        ],
        "figures_and_tables": [{"file": "t.png", "caption": "Table 1: scores."}],
        "qas": [
            {
                "question": "Which retriever is used?",
                "question_id": "q-schema",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": ["BM25"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["The retriever uses BM25."],
                            "highlighted_evidence": ["The retriever uses BM25."],
                        }
                    }
                ],
            }
        ],
    }

    parsed = paper_from_official("paper-schema", paper)
    questions = questions_from_official_paper("paper-schema", paper)

    assert [section.name for section in parsed.sections] == [
        "Abstract",
        "Method",
        "Figures and tables",
    ]
    assert questions[0]["answer"] == "BM25"
    assert questions[0]["yes_no"] is None
    assert questions[0]["unanswerable"] is False
    assert questions[0]["evidence"] == ["The retriever uses BM25."]


def test_yes_no_false_is_a_real_no_not_a_missing_value():
    normalized = normalize_qasper_answer(
        {
            "unanswerable": False,
            "extractive_spans": [],
            "yes_no": False,
            "free_form_answer": "",
            "evidence": ["The paper says no."],
            "highlighted_evidence": [],
        }
    )

    assert normalized["answer"] == "no"
    assert normalized["yes_no"] is False


def test_subset_is_stratified_disjoint_and_repeatable():
    questions = []
    for index in range(20):
        questions.append(
            {
                "paper_id": f"p{index % 4}",
                "question_id": f"a{index:02d}",
                "question": "q",
                "unanswerable": False,
            }
        )
    for index in range(5):
        questions.append(
            {
                "paper_id": "p0",
                "question_id": f"u{index:02d}",
                "question": "q",
                "unanswerable": True,
            }
        )

    first_eval, first_cal = select_subset(
        [dict(row) for row in questions],
        seed=20261107,
        evaluation_size=10,
        calibration_size=4,
    )
    second_eval, second_cal = select_subset(
        [dict(row) for row in questions],
        seed=20261107,
        evaluation_size=10,
        calibration_size=4,
    )

    assert [row["question_id"] for row in first_eval] == [
        row["question_id"] for row in second_eval
    ]
    assert [row["question_id"] for row in first_cal] == [
        row["question_id"] for row in second_cal
    ]
    assert not ({row["question_id"] for row in first_eval} & {row["question_id"] for row in first_cal})
    assert sum(row["unanswerable"] for row in first_eval) == round(10 * 5 / 25)
    assert len(first_eval) == 10
    assert len(first_cal) == 4


def test_token_f1_ignores_citation_markers_and_scores_refusal_as_zero():
    assert token_f1("BM25 is used [paper:1].", "BM25 is used") > 0.9
    rows = [
        {
            "error": None,
            "unanswerable": False,
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "annotators": [{"answer": "BM25", "unanswerable": False}],
            "evidence": ["The retriever uses BM25."],
            "retrieved": [],
            "verifications": [],
            "latency_seconds": 0.1,
        }
    ]

    scored = score_run(rows)

    assert scored["token_f1_answerable"] == 0.0
    assert scored["api_failure_rate"] == 0.0


def test_evidence_cover_accepts_a_contained_paragraph():
    evidence = "The retriever uses BM25 and then reranks the passages."
    chunk = "Introduction. The retriever uses BM25 and then reranks the passages. Next sentence."

    assert evidence_covered(evidence, chunk) is True
    assert evidence_covered(evidence, "This chunk is about tomatoes.") is False


def test_b1_with_no_hits_still_records_dense_retrieval():
    model = ScriptedModel("No passage was supplied.")

    result = answer(UNKNOWN_QUESTION, sample_index(), model, "b1")

    assert result.retrieval_mode == "dense"
    assert result.retrieved == ()
    assert model.seen_chunks == []
    assert result.abstained is False
    prompt = answer_prompt(UNKNOWN_QUESTION, [])
    assert "Retrieval ran and returned no passages" in prompt
    assert "Answer the question from memory" not in prompt


def test_ablation_without_verifier_keeps_an_unsupported_sentence():
    raw = (
        "Reciprocal rank fusion combines a keyword list and a dense list [rag-notes:1]. "
        "Tomatoes are unrelated here [missing:9]."
    )
    proposed = answer(FUSION_QUESTION, sample_index(), ScriptedModel(raw, ["SUPPORTED"]), "proposed")
    kept = answer_ablation(
        FUSION_QUESTION,
        sample_index(),
        ScriptedModel(raw),
        "proposed_no_verifier",
    )

    assert "Tomatoes" not in proposed.answer
    assert proposed.verifications[0]["label"] == "SUPPORTED"
    assert "Tomatoes" in kept.answer
    assert kept.citation_accuracy is None
    assert kept.verifications == ()


def test_experiment_writes_rows_and_skips_completed_questions(tmp_path):
    papers = []
    for paper in load_sample_papers():
        papers.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "sections": [
                    {"name": section.name, "text": section.text} for section in paper.sections
                ],
            }
        )
    papers_path = tmp_path / "papers.json"
    papers_path.write_text(json.dumps(papers), encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    questions = [
        {
            "question_id": "q1",
            "paper_id": "rag-notes",
            "question": FUSION_QUESTION,
            "answer": "a keyword list and a dense list",
            "unanswerable": False,
            "yes_no": None,
            "evidence": [
                "Reciprocal rank fusion combines a keyword list from BM25 and a dense list from embeddings."
            ],
            "annotators": [
                {
                    "answer": "a keyword list from BM25 and a dense list",
                    "unanswerable": False,
                }
            ],
        },
        {
            "question_id": "q2",
            "paper_id": "leaf-spots",
            "question": UNKNOWN_QUESTION,
            "answer": "",
            "unanswerable": True,
            "yes_no": None,
            "evidence": [],
            "annotators": [{"answer": "", "unanswerable": True}],
        },
    ]
    questions_path.write_text(
        "\n".join(json.dumps(row) for row in questions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    summary = run_experiment(
        questions_path,
        papers_path,
        output,
        ExtractiveModel(),
        systems=("b0", "proposed"),
    )
    run_experiment(
        questions_path,
        papers_path,
        output,
        ExtractiveModel(),
        systems=("b0", "proposed"),
    )

    proposed_lines = (output / "proposed_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(proposed_lines) == 2
    assert summary["proposed"]["api_failure_rate"] == 0.0
    assert summary["proposed"]["unanswerable_questions"] == 1
    assert summary["proposed"]["appropriate_abstention_on_unanswerable"] == 1.0
    assert summary["b0"]["token_f1_answerable"] is not None
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    assert config["embedder"] == "HashingEmbedder"
    assert config["threshold"] is None
    index = HybridIndex.build(load_sample_papers())
    assert index.chunks
