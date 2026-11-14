import re

from research_assistant.types import (
    ABSTAIN_MESSAGE,
    ABLATIONS,
    SYSTEMS,
    AnswerResult,
    Hit,
    Paper,
)
from research_assistant.retrieve import HybridIndex


_CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def answer(
    question: str,
    index: HybridIndex,
    model,
    system: str,
    threshold: float | None = None,
    k: int = 5,
) -> AnswerResult:
    """Run one of the three report systems.

    b0 is the generator with no passages.
    b1 is dense top-k pasted into the prompt, with no support check.
    proposed is hybrid retrieval, a citation on every kept sentence, a
    support check, and abstention when retrieval is empty or below threshold.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {', '.join(SYSTEMS)}.")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    if system == "b0":
        hits: list[Hit] = []
        mode = "none"
    elif system == "b1":
        hits = index.search(question, mode="dense", k=k)
        mode = "dense"
    else:
        hits = index.search(question, mode="hybrid", k=k)
        mode = "hybrid"
        if not hits or (threshold is not None and hits[0].score < threshold):
            return _abstain(question, system, hits, model.name, mode)

    supplied = None if system == "b0" else [hit.chunk for hit in hits]
    raw = model.generate_answer(question, supplied)
    if system != "proposed":
        return _unverified(question, system, raw, hits, model.name, mode)
    return _verified(question, system, raw, hits, model, mode)


def answer_ablation(
    question: str,
    index: HybridIndex,
    model,
    system: str,
    threshold: float | None = None,
    k: int = 5,
) -> AnswerResult:
    """Component removals. Retrieval abstention matches proposed.

    proposed_no_verifier keeps hybrid retrieval and skips the support check.
    proposed_bm25_only and proposed_dense_only keep the support check.
    """
    spec = {
        "proposed_no_verifier": ("hybrid", False),
        "proposed_bm25_only": ("bm25", True),
        "proposed_dense_only": ("dense", True),
    }
    if system not in spec:
        raise ValueError(f"ablation must be one of {', '.join(ABLATIONS)}.")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    mode, verify = spec[system]
    hits = index.search(question, mode=mode, k=k)
    if not hits or (threshold is not None and hits[0].score < threshold):
        return _abstain(question, system, hits, model.name, mode)
    raw = model.generate_answer(question, [hit.chunk for hit in hits])
    if not verify:
        return _unverified(question, system, raw, hits, model.name, mode)
    return _verified(question, system, raw, hits, model, mode)


def compare(question: str, index: HybridIndex, model, threshold: float | None = None) -> list[AnswerResult]:
    return [
        answer(question, index, model, system, threshold=threshold)
        for system in SYSTEMS
    ]


def literature_row(paper: Paper, index: HybridIndex, model) -> str:
    chunks = [chunk for chunk in index.chunks if chunk.paper_id == paper.paper_id][:5]
    if not chunks:
        raise ValueError(f"{paper.paper_id} is not in the index.")
    return model.literature_row(paper, chunks)


def summarize(results: list[AnswerResult]) -> dict[str, float | int | None]:
    if not results:
        raise ValueError("No results to summarize.")
    scored = [item.citation_accuracy for item in results if item.citation_accuracy is not None]
    abstained = sum(1 for item in results if item.abstained)
    return {
        "questions": len(results),
        "abstention_rate": abstained / len(results),
        "mean_citation_accuracy": (sum(scored) / len(scored)) if scored else None,
    }


def _unverified(question, system, raw, hits, generator, mode) -> AnswerResult:
    text = raw.strip()
    return AnswerResult(
        system=system,
        question=question,
        answer=text,
        abstained=False,
        raw_answer=text,
        citations=tuple(_citation_ids(text)),
        retrieved=tuple(hits),
        dropped=(),
        citation_accuracy=None,
        generator=generator,
        retrieval_mode=mode,
    )


def _verified(question, system, raw, hits, model, mode) -> AnswerResult:
    if raw.strip() == ABSTAIN_MESSAGE:
        return _abstain(question, system, hits, model.name, mode, raw=raw.strip())
    kept, dropped, supported, judged, decisions = _verify(raw, hits, model)
    accuracy = _accuracy(supported, judged)
    if not kept:
        return _abstain(
            question,
            system,
            hits,
            model.name,
            mode,
            raw=raw.strip(),
            dropped=dropped,
            citation_accuracy=accuracy,
            verifications=decisions,
        )
    final = " ".join(kept)
    return AnswerResult(
        system=system,
        question=question,
        answer=final,
        abstained=False,
        raw_answer=raw.strip(),
        citations=tuple(_citation_ids(final)),
        retrieved=tuple(hits),
        dropped=tuple(dropped),
        citation_accuracy=accuracy,
        generator=model.name,
        retrieval_mode=mode,
        verifications=tuple(decisions),
    )


def _verify(raw: str, hits: list[Hit], model):
    by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits}
    kept: list[str] = []
    dropped: list[str] = []
    decisions: list[dict] = []
    supported = 0
    judged = 0
    for sentence in _sentences(raw):
        cited = _citation_ids(sentence)
        if not cited:
            dropped.append(sentence)
            decisions.append(
                {"sentence": sentence, "citation": None, "label": "UNCITED", "kept": False}
            )
            continue
        judged += 1
        chunk = by_id.get(cited[0])
        if chunk is None:
            dropped.append(sentence)
            decisions.append(
                {
                    "sentence": sentence,
                    "citation": cited[0],
                    "label": "MISSING_CHUNK",
                    "kept": False,
                }
            )
            continue
        label = model.judge_support(sentence, chunk)
        keep = label == "SUPPORTED"
        if keep:
            supported += 1
            kept.append(sentence)
        else:
            dropped.append(sentence)
        decisions.append(
            {
                "sentence": sentence,
                "citation": cited[0],
                "label": label,
                "kept": keep,
            }
        )
    return kept, dropped, supported, judged, decisions


def _abstain(
    question: str,
    system: str,
    hits: list[Hit],
    generator: str,
    mode: str,
    raw: str = "",
    dropped: list[str] | None = None,
    citation_accuracy: float | None = None,
    verifications: list | None = None,
) -> AnswerResult:
    return AnswerResult(
        system=system,
        question=question,
        answer=ABSTAIN_MESSAGE,
        abstained=True,
        raw_answer=raw,
        citations=(),
        retrieved=tuple(hits),
        dropped=tuple(dropped or ()),
        citation_accuracy=citation_accuracy,
        generator=generator,
        retrieval_mode=mode,
        verifications=tuple(verifications or ()),
    )


def _accuracy(supported: int, judged: int) -> float | None:
    if judged == 0:
        return None
    return supported / judged


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text.strip()) if part.strip()]


def _citation_ids(text: str) -> list[str]:
    found = []
    for chunk_id in _CITATION.findall(text):
        if chunk_id not in found:
            found.append(chunk_id)
    return found
