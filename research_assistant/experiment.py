"""Run B0, B1, proposed, and the three ablations on a frozen question file.

Completed question/system pairs are skipped on a later run. API errors are
written into the jsonl and are not dropped.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

from research_assistant.llm import answer_prompt, support_prompt
from research_assistant.metrics import score_run
from research_assistant.pipeline import answer, answer_ablation
from research_assistant.qasper_data import load_frozen_papers, load_question_jsonl
from research_assistant.retrieve import HashingEmbedder, HybridIndex
from research_assistant.types import ABLATIONS, SYSTEMS, AnswerResult


RUN_SYSTEMS = SYSTEMS + ABLATIONS


def prompt_version() -> str:
    import hashlib

    source = inspect.getsource(answer_prompt) + inspect.getsource(support_prompt)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def run_experiment(
    questions_path: Path,
    papers_path: Path,
    output_dir: Path,
    model,
    systems: tuple[str, ...] = RUN_SYSTEMS,
    threshold: float | None = None,
    k: int = 5,
    chunk_words: int = 200,
    overlap: int = 40,
) -> dict:
    questions = load_question_jsonl(questions_path)
    papers = load_frozen_papers(papers_path)
    index = HybridIndex.build(
        papers,
        embedder=HashingEmbedder(),
        chunk_words=chunk_words,
        overlap=overlap,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "questions_path": str(questions_path),
        "papers_path": str(papers_path),
        "generator": getattr(model, "name", type(model).__name__),
        "temperature": 0,
        "prompt_version": prompt_version(),
        "embedder": "HashingEmbedder",
        "embedder_note": (
            "Cosine similarity over hashed token counts. This is the dense "
            "channel already implemented in the repository. It is not a "
            "neural embedding model such as all-MiniLM-L6-v2."
        ),
        "chunk_words": chunk_words,
        "overlap": overlap,
        "k": k,
        "threshold": threshold,
        "threshold_note": (
            "Null means the score threshold was not fitted. Proposed still "
            "abstains when retrieval returns nothing or the verifier deletes "
            "every sentence. The calibration split was not used."
        ),
        "systems": list(systems),
        "indexed_papers": len(papers),
        "indexed_chunks": len(index.chunks),
        "question_count": len(questions),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    paths = {system: output_dir / f"{system}_results.jsonl" for system in systems}
    done = {system: _completed_ids(paths[system]) for system in systems}
    for question in questions:
        for system in systems:
            if question["question_id"] in done[system]:
                continue
            record = _run_one(question, index, model, system, threshold, k)
            _append(paths[system], record)
            done[system].add(question["question_id"])
            status = "error" if record.get("error") else ("abstain" if record.get("abstained") else "answered")
            print(
                f"{system} {question['question_id']} {status} "
                f"{record.get('latency_seconds', 0):.1f}s",
                flush=True,
            )
    summary = {}
    for system, path in paths.items():
        rows = load_question_jsonl(path)
        summary[system] = score_run(rows)
    (output_dir / "metric_results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_one(question, index, model, system, threshold, k) -> dict:
    started = time.perf_counter()
    try:
        if system in ABLATIONS:
            result = answer_ablation(
                question["question"],
                index,
                model,
                system,
                threshold=threshold,
                k=k,
            )
        else:
            result = answer(
                question["question"],
                index,
                model,
                system,
                threshold=threshold,
                k=k,
            )
        elapsed = time.perf_counter() - started
        return _record(question, result, elapsed, None)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "question_id": question["question_id"],
            "paper_id": question["paper_id"],
            "question": question["question"],
            "reference_answer": question.get("answer") or "",
            "unanswerable": bool(question.get("unanswerable")),
            "yes_no": question.get("yes_no"),
            "evidence": question.get("evidence") or [],
            "annotators": question.get("annotators") or [],
            "system": system,
            "generator": getattr(model, "name", ""),
            "answer": "",
            "raw_answer": "",
            "abstained": False,
            "citations": [],
            "dropped": [],
            "verifications": [],
            "retrieved": [],
            "latency_seconds": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _record(question: dict, result: AnswerResult, elapsed: float, error: str | None) -> dict:
    return {
        "question_id": question["question_id"],
        "paper_id": question["paper_id"],
        "question": question["question"],
        "reference_answer": question.get("answer") or "",
        "unanswerable": bool(question.get("unanswerable")),
        "yes_no": question.get("yes_no"),
        "evidence": question.get("evidence") or [],
        "annotators": question.get("annotators") or [],
        "system": result.system,
        "generator": result.generator,
        "retrieval_mode": result.retrieval_mode,
        "answer": result.answer,
        "raw_answer": result.raw_answer,
        "abstained": result.abstained,
        "citations": list(result.citations),
        "dropped": list(result.dropped),
        "model_judge_citation_accuracy": result.citation_accuracy,
        "verifications": [dict(item) for item in result.verifications],
        "retrieved": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "paper_id": hit.chunk.paper_id,
                "section": hit.chunk.section,
                "score": hit.score,
                "text": hit.chunk.text,
            }
            for hit in result.retrieved
        ],
        "latency_seconds": elapsed,
        "error": error,
    }


def main() -> int:
    import argparse

    from config import get_gemini_api_key

    from research_assistant.llm import GeminiModel

    parser = argparse.ArgumentParser(
        description="Score the frozen Qasper subset. Re-running skips completed rows."
    )
    parser.add_argument(
        "--questions",
        default="experiments/dataset/evaluation_questions.jsonl",
    )
    parser.add_argument("--papers", default="experiments/dataset/papers.json")
    parser.add_argument("--output", default="experiments/runs")
    parser.add_argument("--systems", nargs="*", default=list(RUN_SYSTEMS))
    args = parser.parse_args()
    api_key = get_gemini_api_key()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not configured.")
    model = GeminiModel(
        api_key,
        cache_path=str(Path(args.output) / "gemini_cache.json"),
        retries=6,
    )
    summary = run_experiment(
        Path(args.questions),
        Path(args.papers),
        Path(args.output),
        model,
        systems=tuple(args.systems),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _completed_ids(path: Path) -> set[str]:
    found = set()
    for row in _rows(path):
        if row.get("question_id") and not row.get("error"):
            found.add(row["question_id"])
    return found


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(path)
    replaced = False
    updated = []
    for row in rows:
        if row.get("question_id") == record["question_id"] and row.get("error"):
            updated.append(record)
            replaced = True
        else:
            updated.append(row)
    if not replaced:
        updated.append(record)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_question_jsonl(path)


if __name__ == "__main__":
    raise SystemExit(main())
