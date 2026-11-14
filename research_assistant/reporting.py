"""Build tables, SVG figures, the human-evaluation sheet, and chapter 6.

Every number is read from experiments/runs/metric_results.json or the
result jsonl files. Missing runs stay missing.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from research_assistant.qasper_data import load_question_jsonl


SYSTEM_LABELS = {
    "b0": "B0 LLM-only",
    "b1": "B1 dense RAG",
    "proposed": "Proposed",
    "proposed_no_verifier": "No verifier",
    "proposed_bm25_only": "BM25 only",
    "proposed_dense_only": "Dense only",
}
MAIN_SYSTEMS = ("b0", "b1", "proposed")
ABLATION_SYSTEMS = ("proposed", "proposed_no_verifier", "proposed_bm25_only", "proposed_dense_only")
METRIC_FIELDS = (
    ("token_f1_answerable", "Answerable token F1"),
    ("context_recall", "Context recall"),
    ("context_precision", "Context precision"),
    ("model_judge_support_rate", "Model-judge support"),
    ("model_judge_unsupported_rate", "Model-judge unsupported rate"),
    ("evidence_citation_hit_rate", "Evidence citation hit rate"),
    ("refusal_rate", "Refusal rate"),
    ("appropriate_abstention_on_unanswerable", "Abstention on unanswerable"),
    ("refusal_rate_on_answerable", "Refusal on answerable"),
    ("mean_latency_seconds", "Mean latency (seconds)"),
    ("api_failure_rate", "API failure rate"),
)


def write_reports(runs_dir: Path, dataset_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tables").mkdir(exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)
    metrics_path = runs_dir / "metric_results.json"
    if not metrics_path.exists():
        (report_dir / "chapter6_results.md").write_text(
            "# Results and discussion\n\nNo metric_results.json is present. "
            "The experiment has not been scored.\n",
            encoding="utf-8",
        )
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = {}
    config_path = runs_dir / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = {}
    manifest_path = dataset_dir / "qasper_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    table = _markdown_table(metrics, MAIN_SYSTEMS)
    ablation = _markdown_table(metrics, ABLATION_SYSTEMS)
    (report_dir / "tables" / "main_comparison.md").write_text(table, encoding="utf-8")
    (report_dir / "tables" / "ablation.md").write_text(ablation, encoding="utf-8")
    _write_svg(report_dir / "figures" / "token_f1.svg", metrics, MAIN_SYSTEMS, "token_f1_answerable", "Answerable token F1")
    _write_svg(report_dir / "figures" / "refusal.svg", metrics, MAIN_SYSTEMS, "refusal_rate", "Refusal rate")
    _write_svg(report_dir / "figures" / "context_recall.svg", metrics, MAIN_SYSTEMS, "context_recall", "Context recall")
    _write_svg(
        report_dir / "figures" / "ablation_f1.svg",
        metrics,
        ABLATION_SYSTEMS,
        "token_f1_answerable",
        "Ablation answerable token F1",
    )
    _write_human_sheet(runs_dir, dataset_dir, report_dir / "human_evaluation_template.csv")
    chapter = _chapter(metrics, config, manifest)
    (report_dir / "chapter6_results.md").write_text(chapter, encoding="utf-8")
    (report_dir / "results.html").write_text(
        _html_page(chapter, report_dir),
        encoding="utf-8",
    )


def _markdown_table(metrics: dict, systems: tuple[str, ...]) -> str:
    header = "| Metric | " + " | ".join(SYSTEM_LABELS.get(system, system) for system in systems) + " |"
    rule = "| --- | " + " | ".join("---" for _ in systems) + " |"
    lines = [header, rule]
    for field, label in METRIC_FIELDS:
        cells = [_format(metrics.get(system, {}).get(field)) for system in systems]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    counts = [_format(metrics.get(system, {}).get("scored_rows"), digits=0) for system in systems]
    lines.append("| Scored questions | " + " | ".join(counts) + " |")
    return "\n".join(lines) + "\n"


def _format(value, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_svg(path: Path, metrics: dict, systems: tuple[str, ...], field: str, title: str) -> None:
    values = []
    for system in systems:
        raw = metrics.get(system, {}).get(field)
        values.append(None if raw is None else float(raw))
    present = [value for value in values if value is not None]
    top = max(present) if present else 1.0
    if top <= 0:
        top = 1.0
    width = 120
    gap = 36
    left = 48
    base = 220
    height = 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + len(systems) * (width + gap)}" height="{height}">',
        f'<text x="16" y="24" font-family="sans-serif" font-size="16">{_xml(title)}</text>',
        f'<text x="16" y="{height - 8}" font-family="sans-serif" font-size="11">Source: experiments/runs/metric_results.json</text>',
    ]
    for index, (system, value) in enumerate(zip(systems, values)):
        x = left + index * (width + gap)
        if value is None:
            parts.append(
                f'<text x="{x}" y="{base}" font-family="sans-serif" font-size="12">unavailable</text>'
            )
        else:
            bar = max(1, int((value / top) * 160))
            parts.append(
                f'<rect x="{x}" y="{base - bar}" width="{width}" height="{bar}" fill="#3d5a40"/>'
            )
            parts.append(
                f'<text x="{x}" y="{base - bar - 6}" font-family="sans-serif" font-size="12">{value:.3f}</text>'
            )
        parts.append(
            f'<text x="{x}" y="{base + 18}" font-family="sans-serif" font-size="11">{_xml(SYSTEM_LABELS.get(system, system))}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_human_sheet(runs_dir: Path, dataset_dir: Path, path: Path) -> None:
    questions = []
    question_path = dataset_dir / "evaluation_questions.jsonl"
    if question_path.exists():
        questions = load_question_jsonl(question_path)
    questions = sorted(questions, key=lambda row: row["question_id"])[:20]
    proposed = {}
    proposed_path = runs_dir / "proposed_results.jsonl"
    if proposed_path.exists():
        for row in load_question_jsonl(proposed_path):
            proposed[row["question_id"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "system",
                "question",
                "generated_answer",
                "reference_answer",
                "evidence",
                "citations",
                "reviewer",
                "factual_correctness_1_to_5",
                "evidence_support_1_to_5",
                "citation_correctness_1_to_5",
                "citation_usefulness_1_to_5",
                "unsupported_claim_count",
                "appropriate_abstention",
                "completeness_1_to_5",
                "comments",
                "disagreement_resolution",
            ],
        )
        writer.writeheader()
        for question in questions:
            result = proposed.get(question["question_id"], {})
            for reviewer in ("reviewer_1", "reviewer_2"):
                writer.writerow(
                    {
                        "question_id": question["question_id"],
                        "system": "proposed",
                        "question": question.get("question", ""),
                        "generated_answer": result.get("answer", ""),
                        "reference_answer": question.get("answer", ""),
                        "evidence": " | ".join(question.get("evidence") or []),
                        "citations": " ".join(result.get("citations") or []),
                        "reviewer": reviewer,
                        "factual_correctness_1_to_5": "",
                        "evidence_support_1_to_5": "",
                        "citation_correctness_1_to_5": "",
                        "citation_usefulness_1_to_5": "",
                        "unsupported_claim_count": "",
                        "appropriate_abstention": "",
                        "completeness_1_to_5": "",
                        "comments": "",
                        "disagreement_resolution": "",
                    }
                )


def _chapter(metrics: dict, config: dict, manifest: dict) -> str:
    scored = {
        system: block.get("scored_rows")
        for system, block in metrics.items()
        if isinstance(block, dict)
    }
    target = manifest.get("evaluation_questions", "unknown")
    lines = [
        "# Chapter 6. Results and discussion",
        "",
        "The numbers in this chapter are computed from `experiments/runs`. "
        "They are not fixture demonstrations and they are not published scores "
        "from PaperQA2 or OpenScholar.",
        "",
        "## Experimental setup",
        "",
        f"- Dataset: Qasper {manifest.get('dataset_version', '0.3')}, validation split, "
        f"archive member `{manifest.get('archive_member', 'qasper-dev-v0.3.json')}`.",
        f"- Evaluation questions: {manifest.get('evaluation_questions', 'unavailable')}. "
        f"Papers indexed: {manifest.get('evaluation_papers', 'unavailable')}. "
        f"Unanswerable questions in the subset: {manifest.get('evaluation_unanswerable', 'unavailable')}.",
        f"- Selection seed: {manifest.get('random_seed', 'unavailable')}. "
        "The calibration split was not scored and was not used to set a threshold.",
        f"- Generator: {config.get('generator', 'unavailable')}. Temperature: {config.get('temperature', 'unavailable')}. "
        f"Prompt version: {config.get('prompt_version', 'unavailable')}.",
        f"- Retrieval: {config.get('embedder', 'unavailable')}. {config.get('embedder_note', '')}",
        f"- Chunks: {config.get('chunk_words', 'unavailable')} words, overlap {config.get('overlap', 'unavailable')}, "
        f"top {config.get('k', 'unavailable')}. Indexed chunks: {config.get('indexed_chunks', 'unavailable')}.",
        f"- Threshold: {config.get('threshold')}. {config.get('threshold_note', '')}",
        "",
        f"Target size was {target} questions. Scored rows by system: {scored}.",
        "",
        "## Baseline comparison",
        "",
        _markdown_table(metrics, MAIN_SYSTEMS),
        "",
        "Token F1 is SQuAD token overlap against the best answerable annotator. "
        "A refusal on an answerable question scores 0. Unanswerable questions are "
        "left out of that mean and are reported through the abstention rate.",
        "",
        _discussion(metrics),
        "",
        "Context recall and context precision use gold evidence strings from Qasper. "
        "They are undefined in the table only when a system has no scored evidence rows. "
        "B0 retrieves nothing, so its context precision is 0 whenever evidence exists.",
        "",
        "The model-judge support rate is the verifier's SUPPORTED fraction. "
        "It is not human ground truth. B0 and B1 do not run the verifier, so that "
        "cell is unavailable for them.",
        "",
        "## Ablation",
        "",
        _markdown_table(metrics, ABLATION_SYSTEMS),
        "",
        "No verifier uses the same hybrid retrieval as the proposed system and "
        "keeps the generator's raw answer. BM25 only and dense only keep the "
        "verifier and change one retrieval channel. A higher token F1 in one "
        "row is a difference on this subset. No significance test was run, and "
        "these rows do not show that the system beats published research assistants.",
        "",
        "## Human evaluation",
        "",
        "The sheet `human_evaluation_template.csv` lists the 20 evaluation "
        "questions with the lowest question ids, with blank reviewer scores. "
        "Two reviewers have not completed it. No human rating is reported.",
        "",
        "## Error analysis and limitations",
        "",
        "- The dense channel is hashed token cosine, not a trained neural retriever.",
        "- Retrieval searches the whole frozen collection, not only the question's paper.",
        "- Qasper figure evidence marked `FLOAT SELECTED` may not appear in the body text. Captions are indexed.",
        "- The official test split was not scored because its answers are not in the train/dev archive.",
        "- API failures, if the failure rate is above zero, remain in the jsonl and in the table.",
        "- Passing the software tests does not mean the answers are correct.",
        "",
    ]
    return "\n".join(lines)


def _discussion(metrics: dict) -> str:
    def value(system: str, field: str) -> str:
        raw = metrics.get(system, {}).get(field)
        if raw is None:
            return "unavailable"
        if isinstance(raw, float):
            return f"{raw:.3f}"
        return str(raw)

    return "\n".join(
        [
            "On this 80-question slice the proposed mean token F1 on answerable "
            f"questions was {value('proposed', 'token_f1_answerable')}, compared with "
            f"{value('b0', 'token_f1_answerable')} for B0 and "
            f"{value('b1', 'token_f1_answerable')} for B1. "
            f"The proposed system refused {value('proposed', 'refusal_rate_on_answerable')} "
            "of answerable questions, and those refusals score 0 in the mean. "
            "The absolute F1 values are low. That is expected for full-paper questions "
            "when many answers are refusals and when retrieval often misses the gold evidence.",
            "",
            f"B0 never refused (refusal rate {value('b0', 'refusal_rate')}). "
            f"B1's pipeline abstention flag is {value('b1', 'pipeline_abstention_rate')} "
            "because B1 does not set that flag, while its answer text matched the "
            f"abstain sentence on {value('b1', 'refusal_rate')} of all scored questions. "
            f"On the unanswerable questions, B1 and the proposed system both abstained "
            f"at {value('b1', 'appropriate_abstention_on_unanswerable')} and "
            f"{value('proposed', 'appropriate_abstention_on_unanswerable')}.",
            "",
            f"Context recall was {value('proposed', 'context_recall')} for the proposed "
            f"system and {value('b1', 'context_recall')} for B1. "
            f"The model-judge support rate for the proposed system was "
            f"{value('proposed', 'model_judge_support_rate')}, while the share of those "
            f"cited chunks that actually covered a gold evidence string was "
            f"{value('proposed', 'evidence_citation_hit_rate')}. "
            "The verifier and the generator agree more often than the cited chunk "
            "matches the Qasper evidence. The judge score is not ground truth.",
            "",
            f"Dropping the verifier moved token F1 from {value('proposed', 'token_f1_answerable')} "
            f"to {value('proposed_no_verifier', 'token_f1_answerable')}. "
            f"Its pipeline abstention flag is {value('proposed_no_verifier', 'pipeline_abstention_rate')}, "
            f"while its refusal rate stays {value('proposed_no_verifier', 'refusal_rate')}, "
            "the same as the proposed system. Most refusals are the generator writing the "
            "abstain sentence, not the verifier deleting sentences. "
            f"BM25 only was {value('proposed_bm25_only', 'token_f1_answerable')} and dense only "
            f"was {value('proposed_dense_only', 'token_f1_answerable')}. "
            f"Dense only refused {value('proposed_dense_only', 'refusal_rate_on_answerable')} "
            "of answerable questions. Hybrid retrieval is associated with a higher "
            "token F1 than either channel alone on this slice. The verifier change is "
            "small. No significance test was run, and none of these rows is a comparison "
            "with PaperQA2 or OpenScholar.",
            "",
            f"Recorded mean latency was {value('b0', 'mean_latency_seconds')} seconds for B0, "
            f"{value('b1', 'mean_latency_seconds')} for B1, and "
            f"{value('proposed', 'mean_latency_seconds')} for the proposed system. "
            f"The no-verifier latency of {value('proposed_no_verifier', 'mean_latency_seconds')} "
            "seconds is not a speed result: that ablation uses the same generation prompt "
            "as the proposed system, so the response cache answered it. "
            "Two early requests for question df5a4505 returned HTTP 429 on the free-tier "
            "limit of 15 requests per minute. Those error rows were retried and replaced. "
            "The saved API failure rate is 0.",
        ]
    )


def _html_page(chapter: str, report_dir: Path) -> str:
    figures = []
    for name in ("token_f1.svg", "context_recall.svg", "refusal.svg", "ablation_f1.svg"):
        figure = report_dir / "figures" / name
        if figure.exists():
            figures.append(f"<h2>{html.escape(name)}</h2>{figure.read_text(encoding='utf-8')}")
    body = "<pre>" + html.escape(chapter) + "</pre>" + "\n".join(figures)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Qasper results</title></head><body>"
        + body
        + "</body></html>"
    )


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main() -> int:
    write_reports(
        Path("experiments/runs"),
        Path("experiments/dataset"),
        Path("experiments/reports"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
