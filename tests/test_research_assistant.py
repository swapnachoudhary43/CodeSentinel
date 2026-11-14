import pytest

from research_assistant.corpus import (
    chunk_paper,
    load_sample_papers,
    paper_from_qasper_row,
    questions_from_qasper_row,
)
from research_assistant.cli import main
from research_assistant.llm import ExtractiveModel, answer_prompt, parse_support, support_prompt
from research_assistant.pipeline import answer, compare, literature_row, summarize
from research_assistant.retrieve import HybridIndex, reciprocal_rank_fusion
from research_assistant.types import ABSTAIN_MESSAGE, AnswerResult, Paper, Section


FUSION_QUESTION = "What does reciprocal rank fusion combine?"
LEAF_QUESTION = "What metric does the leaf disease paper report?"
UNKNOWN_QUESTION = "What is the capital of France?"


class ScriptedModel:
    name = "scripted"

    def __init__(self, answer_text: str, judgments: list[str] | None = None):
        self.answer_text = answer_text
        self.judgments = list(judgments or [])
        self.seen_chunks = None
        self.judge_calls = 0

    def generate_answer(self, question, chunks):
        del question
        self.seen_chunks = chunks
        return self.answer_text

    def judge_support(self, sentence, chunk):
        del sentence, chunk
        self.judge_calls += 1
        if not self.judgments:
            return "NOT_SUPPORTED"
        return self.judgments.pop(0)

    def literature_row(self, paper, chunks):
        del chunks
        return f"Title: {paper.title}"


def sample_index(**kwargs) -> HybridIndex:
    return HybridIndex.build(load_sample_papers(), **kwargs)


def test_sections_stay_in_separate_chunks():
    paper = Paper(
        "demo",
        "Demo",
        2024,
        (
            Section("Abstract", "alpha beta gamma"),
            Section("Method", "delta epsilon zeta"),
        ),
    )

    chunks = chunk_paper(paper, chunk_words=50, overlap=10)

    assert [chunk.section for chunk in chunks] == ["Abstract", "Method"]
    assert chunks[0].chunk_id == "demo:0"
    assert "delta" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_overlap_windows_share_edge_words():
    words = " ".join(str(number) for number in range(100))
    paper = Paper("seq", "Seq", None, (Section("Body", words),))

    chunks = chunk_paper(paper, chunk_words=40, overlap=10)

    assert len(chunks) == 3
    assert chunks[0].text.split()[:3] == ["0", "1", "2"]
    assert chunks[1].text.split()[:3] == ["30", "31", "32"]
    assert chunks[-1].text.split()[-1] == "99"
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_invalid_overlap_is_rejected():
    paper = load_sample_papers()[0]
    with pytest.raises(ValueError):
        chunk_paper(paper, chunk_words=20, overlap=20)


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])

    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 61)


def test_bm25_ranks_the_fusion_note_first():
    index = sample_index()

    hits = index.search(FUSION_QUESTION, mode="bm25", k=4)

    assert hits
    assert hits[0].chunk.paper_id == "rag-notes"
    assert hits[0].chunk.section == "Method"
    assert hits[0].score > 0


def test_dense_search_ignores_an_unindexed_question():
    index = sample_index()

    hits = index.search(UNKNOWN_QUESTION, mode="dense", k=4)

    assert hits == []


def test_hybrid_search_cites_the_fusion_method():
    index = sample_index()

    hits = index.search(FUSION_QUESTION, mode="hybrid", k=2)

    assert hits[0].chunk.paper_id == "rag-notes"
    assert "reciprocal rank fusion" in hits[0].chunk.text.lower()


def test_b0_does_not_retrieve_or_verify():
    index = sample_index()
    model = ScriptedModel("Memory answer without a passage [ungrounded:0].")

    result = answer(FUSION_QUESTION, index, model, "b0")

    assert model.seen_chunks is None
    assert model.judge_calls == 0
    assert result.retrieval_mode == "none"
    assert result.retrieved == ()
    assert result.citation_accuracy is None
    assert result.abstained is False


def test_b1_keeps_the_raw_answer_without_a_support_check():
    index = sample_index()
    model = ScriptedModel("Tomatoes are green [leaf-spots:0].")

    result = answer(LEAF_QUESTION, index, model, "b1")

    assert model.judge_calls == 0
    assert result.retrieval_mode == "dense"
    assert result.answer == "Tomatoes are green [leaf-spots:0]."
    assert result.dropped == ()
    assert result.citation_accuracy is None
    assert result.retrieved


def test_proposed_drops_an_unsupported_sentence_and_scores_citations():
    index = sample_index()
    raw = (
        "Reciprocal rank fusion combines a keyword list and a dense list [rag-notes:1]. "
        "The leaf model reports accuracy [leaf-spots:1]."
    )
    model = ScriptedModel(raw, judgments=["SUPPORTED"])

    result = answer(FUSION_QUESTION, index, model, "proposed")

    assert model.judge_calls == 1
    assert result.abstained is False
    assert result.citations == ("rag-notes:1",)
    assert "leaf model" in result.dropped[0]
    assert result.citation_accuracy == pytest.approx(0.5)


def test_proposed_drops_a_sentence_with_no_citation_and_abstains():
    index = sample_index()
    model = ScriptedModel("Fusion is useful if you trust the model.")

    result = answer(FUSION_QUESTION, index, model, "proposed")

    assert result.abstained is True
    assert result.answer == ABSTAIN_MESSAGE
    assert result.citation_accuracy is None
    assert result.dropped


def test_low_fusion_score_abstains_before_generation():
    index = sample_index()
    model = ScriptedModel("should not be called")

    result = answer(FUSION_QUESTION, index, model, "proposed", threshold=1.0)

    assert result.abstained is True
    assert result.answer == ABSTAIN_MESSAGE
    assert model.seen_chunks is None
    assert result.retrieved


def test_unknown_question_abstains_without_passages():
    index = sample_index()

    result = answer(UNKNOWN_QUESTION, index, ExtractiveModel(), "proposed")

    assert result.abstained is True
    assert result.retrieved == ()
    assert result.answer == ABSTAIN_MESSAGE


def test_extractive_proposed_quotes_the_fusion_passage():
    index = sample_index()

    result = answer(FUSION_QUESTION, index, ExtractiveModel(), "proposed")

    assert result.abstained is False
    assert result.citations[0].startswith("rag-notes")
    assert "reciprocal rank fusion" in result.answer.lower()
    assert result.citation_accuracy == pytest.approx(1.0)
    assert result.generator == "extractive"


def test_extractive_leaf_question_stays_on_the_leaf_note():
    index = sample_index()

    result = answer(LEAF_QUESTION, index, ExtractiveModel(), "proposed")

    assert result.abstained is False
    assert result.citations == ("leaf-spots:1",)
    assert result.retrieved[0].chunk.section == "Method"
    assert "accuracy" in result.answer.lower()


def test_compare_runs_all_three_systems():
    index = sample_index()

    results = compare(FUSION_QUESTION, index, ExtractiveModel())

    assert [item.system for item in results] == ["b0", "b1", "proposed"]
    assert results[0].retrieval_mode == "none"
    assert results[1].retrieval_mode == "dense"
    assert results[2].retrieval_mode == "hybrid"
    assert results[2].citation_accuracy == pytest.approx(1.0)


def test_literature_row_names_the_method():
    papers = load_sample_papers()
    index = sample_index()

    row = literature_row(papers[0], index, ExtractiveModel())

    assert "Hybrid Retrieval Notes" in row
    assert "reciprocal rank fusion" in row


def test_summarize_abstention_and_accuracy():
    kept = AnswerResult(
        system="proposed",
        question="q1",
        answer="kept",
        abstained=False,
        raw_answer="kept",
        citations=("rag-notes:1",),
        retrieved=(),
        dropped=(),
        citation_accuracy=1.0,
        generator="scripted",
        retrieval_mode="hybrid",
    )
    refused = AnswerResult(
        system="proposed",
        question="q2",
        answer=ABSTAIN_MESSAGE,
        abstained=True,
        raw_answer="",
        citations=(),
        retrieved=(),
        dropped=(),
        citation_accuracy=None,
        generator="scripted",
        retrieval_mode="hybrid",
    )

    stats = summarize([kept, refused])

    assert stats["questions"] == 2
    assert stats["abstention_rate"] == pytest.approx(0.5)
    assert stats["mean_citation_accuracy"] == pytest.approx(1.0)


def test_parse_support_fails_closed():
    assert parse_support("SUPPORTED") == "SUPPORTED"
    assert parse_support("SUPPORTED.") == "SUPPORTED"
    assert parse_support("NOT_SUPPORTED") == "NOT_SUPPORTED"
    assert parse_support("The claim looks SUPPORTED") == "NOT_SUPPORTED"
    assert parse_support("") == "NOT_SUPPORTED"


def test_prompts_require_citations_and_the_abstain_string():
    index = sample_index()
    chunk = index.search(FUSION_QUESTION, mode="bm25", k=1)[0].chunk

    prompt = answer_prompt(FUSION_QUESTION, [chunk])
    judge = support_prompt("Fusion combines two lists [rag-notes:1].", chunk)

    assert chunk.chunk_id in prompt
    assert ABSTAIN_MESSAGE in prompt
    assert "SUPPORTED" in judge
    assert "NOT_SUPPORTED" in judge
    assert "no retrieved passages" in answer_prompt(FUSION_QUESTION, None)


def test_qasper_row_maps_sections_and_answers():
    row = {
        "id": "paper-7",
        "title": "A Sample NLP Paper",
        "abstract": "We study retrieval.",
        "full_text": {
            "section_name": ["Method", "Results"],
            "paragraphs": [
                ["The retriever uses BM25."],
                ["Faithfulness is 0.8 on the dev questions.", "A second result."],
            ],
        },
        "qas": {
            "question": ["Which retriever is used?"],
            "question_id": ["q1"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["BM25"],
                            "free_form_answer": "",
                            "yes_no": None,
                            "evidence": ["The retriever uses BM25."],
                        }
                    ]
                }
            ],
        },
    }

    paper = paper_from_qasper_row(row)
    questions = questions_from_qasper_row(row)

    assert [section.name for section in paper.sections] == ["Abstract", "Method", "Results"]
    assert "second result" in paper.sections[2].text
    assert questions[0]["answer"] == "BM25"
    assert questions[0]["unanswerable"] is False
    assert questions[0]["evidence"] == ["The retriever uses BM25."]


def test_unknown_system_is_rejected():
    index = sample_index()
    with pytest.raises(ValueError):
        answer(FUSION_QUESTION, index, ExtractiveModel(), "gpt")


def test_cli_unknown_question_abstains(capsys):
    code = main(["--question", UNKNOWN_QUESTION, "--system", "proposed"])
    output = capsys.readouterr().out

    assert code == 0
    assert "fixture notes" in output
    assert "extractive" in output
    assert "not a Gemini answer" in output
    assert ABSTAIN_MESSAGE in output


def test_cli_all_systems_lists_three_retrieval_modes(capsys):
    code = main(["--question", FUSION_QUESTION, "--system", "all", "--matrix"])
    output = capsys.readouterr().out

    assert code == 0
    assert "retrieval: none" in output
    assert "retrieval: dense" in output
    assert "retrieval: hybrid" in output
    assert "Method: reciprocal rank fusion" in output


def test_cli_gemini_without_a_key_fails(monkeypatch, capsys):
    monkeypatch.setattr("research_assistant.cli.get_gemini_api_key", lambda: "")

    code = main(["--question", FUSION_QUESTION, "--gemini"])
    output = capsys.readouterr().out

    assert code == 2
    assert "GEMINI_API_KEY" in output
