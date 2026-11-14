"""Operational metrics for one frozen question set.

Token F1 follows the SQuAD token-overlap definition after lowercasing and
punctuation removal. Citation markers are removed from system answers before
the comparison so a chunk id is not scored as an extra token.

These scores are not RAGAS scores and they are not human judgments.
A model-verifier label is reported under its own name and is not treated
as ground truth.
"""

from __future__ import annotations

import re
from collections import Counter

from research_assistant.types import ABSTAIN_MESSAGE


_CITATION = re.compile(r"\[[^\]]+\]")
_NON_WORD = re.compile(r"[^a-z0-9\s]")


def tokens(text: str) -> list[str]:
    cleaned = _CITATION.sub(" ", text or "")
    cleaned = _NON_WORD.sub(" ", cleaned.lower())
    return [token for token in cleaned.split() if token]


def token_f1(prediction: str, reference: str) -> float:
    predicted = tokens(prediction)
    gold = tokens(reference)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def max_annotator_f1(prediction: str, annotators: list[dict]) -> float | None:
    answerable = [
        item["answer"]
        for item in annotators
        if not item.get("unanswerable") and str(item.get("answer") or "").strip()
    ]
    if not answerable:
        return None
    return max(token_f1(prediction, answer) for answer in answerable)


def normalize_space(text: str) -> str:
    return " ".join((text or "").lower().split())


def evidence_covered(evidence: str, chunk_text: str) -> bool:
    needle = normalize_space(evidence)
    haystack = normalize_space(chunk_text)
    if not needle or not haystack:
        return False
    if needle in haystack or haystack in needle:
        return True
    evidence_tokens = needle.split()
    if len(evidence_tokens) < 4:
        return False
    chunk_tokens = set(haystack.split())
    found = sum(1 for token in evidence_tokens if token in chunk_tokens)
    return found / len(evidence_tokens) >= 0.7


def retrieval_scores(evidence: list[str], chunks: list[str]) -> dict[str, float | None]:
    usable = [item for item in evidence if normalize_space(item)]
    if not usable:
        return {"context_recall": None, "context_precision": None}
    covered = sum(
        1 for item in usable if any(evidence_covered(item, chunk) for chunk in chunks)
    )
    useful_chunks = sum(
        1 for chunk in chunks if any(evidence_covered(item, chunk) for item in usable)
    )
    precision = (useful_chunks / len(chunks)) if chunks else 0.0
    return {
        "context_recall": covered / len(usable),
        "context_precision": precision,
    }


def refused(row: dict) -> bool:
    answer = str(row.get("answer") or "").strip()
    return bool(row.get("abstained")) or answer == ABSTAIN_MESSAGE


def score_run(rows: list[dict]) -> dict:
    """Score one system's jsonl rows. Failed rows stay in the failure rate."""
    if not rows:
        raise ValueError("No rows to score.")
    failures = [row for row in rows if row.get("error")]
    done = [row for row in rows if not row.get("error")]
    answerable = [row for row in done if not row.get("unanswerable")]
    unanswerable = [row for row in done if row.get("unanswerable")]

    f1_values = []
    for row in answerable:
        if refused(row):
            f1_values.append(0.0)
            continue
        annotators = row.get("annotators") or [
            {"answer": row.get("reference_answer") or "", "unanswerable": False}
        ]
        value = max_annotator_f1(row.get("answer") or "", annotators)
        f1_values.append(0.0 if value is None else value)

    recalls = []
    precisions = []
    for row in done:
        if row.get("unanswerable"):
            continue
        evidence = row.get("evidence") or []
        chunks = [item.get("text") or "" for item in (row.get("retrieved") or [])]
        scores = retrieval_scores(evidence, chunks)
        if scores["context_recall"] is not None:
            recalls.append(scores["context_recall"])
        if scores["context_precision"] is not None:
            precisions.append(scores["context_precision"])

    judge_hits = 0
    judge_total = 0
    evidence_hits = 0
    evidence_total = 0
    for row in done:
        evidence = [
            item for item in (row.get("evidence") or []) if normalize_space(item)
        ]
        retrieved = {
            item.get("chunk_id"): item.get("text") or ""
            for item in (row.get("retrieved") or [])
        }
        for decision in row.get("verifications") or []:
            if decision.get("label") in {"UNCITED", None}:
                continue
            judge_total += 1
            if decision.get("label") == "SUPPORTED":
                judge_hits += 1
            citation = decision.get("citation")
            if citation and evidence and not row.get("unanswerable"):
                evidence_total += 1
                chunk_text = retrieved.get(citation, "")
                if any(evidence_covered(item, chunk_text) for item in evidence):
                    evidence_hits += 1

    latencies = [
        float(row["latency_seconds"])
        for row in done
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    attempts = len(rows)

    def mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    appropriate = None
    if unanswerable:
        appropriate = sum(refused(row) for row in unanswerable) / len(unanswerable)
    false_abstain = None
    if answerable:
        false_abstain = sum(refused(row) for row in answerable) / len(answerable)

    return {
        "rows": attempts,
        "scored_rows": len(done),
        "api_failure_rate": len(failures) / attempts,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "token_f1_answerable": mean(f1_values),
        "token_f1_definition": (
            "Mean maximum SQuAD token F1 against answerable annotators. "
            "An answerable question that the system refuses scores 0. "
            "Unanswerable questions are excluded from this mean."
        ),
        "pipeline_abstention_rate": (
            sum(bool(row.get("abstained")) for row in done) / len(done) if done else None
        ),
        "refusal_rate": (sum(refused(row) for row in done) / len(done)) if done else None,
        "appropriate_abstention_on_unanswerable": appropriate,
        "refusal_rate_on_answerable": false_abstain,
        "context_recall": mean(recalls),
        "context_precision": mean(precisions),
        "retrieval_definition": (
            "Context recall is the fraction of non-empty gold evidence strings "
            "covered by any retrieved chunk. Coverage means the normalized "
            "evidence is a substring of the chunk, the chunk is a substring of "
            "the evidence, or at least 70 percent of evidence tokens of length "
            "4 or more appear in the chunk. Context precision is the fraction "
            "of retrieved chunks that cover any gold evidence string. Both are "
            "unavailable when the question has no textual evidence. b0 has no "
            "retrieved chunks, so its precision is 0 when evidence exists."
        ),
        "model_judge_support_rate": (
            judge_hits / judge_total if judge_total else None
        ),
        "model_judge_unsupported_rate": (
            (judge_total - judge_hits) / judge_total if judge_total else None
        ),
        "model_judge_definition": (
            "Among cited sentences sent to the verifier, the fraction labeled "
            "SUPPORTED. This is a same-model judgment, not ground truth. "
            "MISSING_CHUNK counts as unsupported. Uncited sentences are not "
            "in this denominator."
        ),
        "evidence_citation_hit_rate": (
            evidence_hits / evidence_total if evidence_total else None
        ),
        "evidence_citation_definition": (
            "Among verifier decisions that name a retrieved chunk, the fraction "
            "whose chunk covers a gold evidence string. Unavailable when the "
            "run has no such decisions or the questions have no evidence."
        ),
        "mean_latency_seconds": mean(latencies),
        "judge_decisions": judge_total,
    }
