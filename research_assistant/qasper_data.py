"""Load Qasper v0.3 and freeze a reproducible question subset.

The Hugging Face `datasets` loader script is rejected by datasets 5.x.
This module reads the official train/dev tarball instead:

https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

The public test split is not in that archive, and its answers are not used.
"""

from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

from research_assistant.corpus import paper_from_record
from research_assistant.types import Paper, Section


OFFICIAL_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
DATASET_NAME = "qasper"
DATASET_VERSION = "0.3"
SOURCE_PAGE = "https://allenai.org/data/qasper"
LICENSE_NOTE = (
    "The Hugging Face dataset card for allenai/qasper and its loader script "
    "state CC BY 4.0. The README inside qasper-train-dev-v0.3.tgz does not "
    "repeat the license line."
)
DEV_MEMBER = "qasper-dev-v0.3.json"


def load_dev_papers(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as bundle:
        payload = bundle.extractfile(DEV_MEMBER)
        if payload is None:
            raise ValueError(f"{archive} does not contain {DEV_MEMBER}.")
        papers = json.load(payload)
    if not isinstance(papers, dict):
        raise ValueError("Qasper dev JSON must be a mapping of arXiv id to paper.")
    return papers


def questions_from_official_paper(paper_id: str, paper: dict) -> list[dict]:
    """Use every annotator. The first annotator is the stored reference row."""
    rows = []
    for question in paper.get("qas") or []:
        text = str(question.get("question") or "").strip()
        question_id = str(question.get("question_id") or "").strip()
        if not text or not question_id:
            continue
        annotators = []
        for annotation in question.get("answers") or []:
            annotators.append(normalize_qasper_answer(annotation.get("answer") or {}))
        if not annotators:
            continue
        first = annotators[0]
        rows.append(
            {
                "paper_id": paper_id,
                "question_id": question_id,
                "question": text,
                "answer": first["answer"],
                "unanswerable": first["unanswerable"],
                "yes_no": first["yes_no"],
                "evidence": first["evidence"],
                "highlighted_evidence": first["highlighted_evidence"],
                "annotators": annotators,
                "source_split": "validation",
            }
        )
    return rows


def normalize_qasper_answer(answer: dict) -> dict:
    """Follow the v0.3 README: unanswerable, else exactly one answer channel."""
    spans = [
        str(span).strip()
        for span in (answer.get("extractive_spans") or [])
        if str(span).strip()
    ]
    free_form = str(answer.get("free_form_answer") or "").strip()
    yes_no = answer.get("yes_no")
    if yes_no not in (True, False):
        yes_no = None
    evidence = [
        str(item).strip()
        for item in (answer.get("evidence") or [])
        if str(item).strip()
    ]
    highlighted = [
        str(item).strip()
        for item in (answer.get("highlighted_evidence") or [])
        if str(item).strip()
    ]
    unanswerable = bool(answer.get("unanswerable"))
    if unanswerable:
        text = ""
    elif yes_no is True:
        text = "yes"
    elif yes_no is False:
        text = "no"
    else:
        text = free_form or (spans[0] if spans else "")
    return {
        "unanswerable": unanswerable,
        "answer": text,
        "yes_no": yes_no,
        "evidence": evidence,
        "highlighted_evidence": highlighted,
        "extractive_spans": spans,
        "free_form_answer": free_form,
    }


def paper_from_official(paper_id: str, paper: dict) -> Paper:
    sections: list[Section] = []
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        sections.append(Section("Abstract", abstract))
    for section in paper.get("full_text") or []:
        name = str(section.get("section_name") or "Body").strip() or "Body"
        paragraphs = [
            str(paragraph).strip()
            for paragraph in (section.get("paragraphs") or [])
            if str(paragraph).strip()
        ]
        if paragraphs:
            sections.append(Section(name, "\n".join(paragraphs)))
    captions = []
    for figure in paper.get("figures_and_tables") or []:
        caption = str(figure.get("caption") or "").strip()
        if caption:
            captions.append(caption)
    if captions:
        sections.append(Section("Figures and tables", "\n".join(captions)))
    if not sections:
        raise ValueError(f"{paper_id} has no text.")
    return Paper(paper_id, str(paper.get("title") or paper_id), None, tuple(sections))


def select_subset(
    questions: list[dict],
    seed: int,
    evaluation_size: int = 80,
    calibration_size: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Stratify on the first annotator's unanswerable flag.

    Lists are sorted by paper id and question id, then shuffled with
    random.Random(seed). Calibration items are taken after the evaluation
    items, so the two sets are disjoint. Nothing in here reads model output.
    """
    if evaluation_size < 1 or calibration_size < 0:
        raise ValueError("evaluation_size must be positive and calibration_size non-negative.")
    answerable = _sorted_unique(
        [row for row in questions if not row["unanswerable"]]
    )
    unanswerable = _sorted_unique(
        [row for row in questions if row["unanswerable"]]
    )
    total = len(answerable) + len(unanswerable)
    if total < evaluation_size + calibration_size:
        raise ValueError(
            f"Need {evaluation_size + calibration_size} questions, found {total}."
        )
    generator = random.Random(seed)
    generator.shuffle(answerable)
    generator.shuffle(unanswerable)

    eval_unanswerable = _quota(evaluation_size, len(unanswerable), total)
    calib_unanswerable = _quota(calibration_size, len(unanswerable), total)
    if len(unanswerable) < eval_unanswerable + calib_unanswerable:
        raise ValueError("Not enough unanswerable questions for the requested quotas.")
    eval_answerable = evaluation_size - eval_unanswerable
    calib_answerable = calibration_size - calib_unanswerable
    if len(answerable) < eval_answerable + calib_answerable:
        raise ValueError("Not enough answerable questions for the requested quotas.")

    evaluation = (
        answerable[:eval_answerable] + unanswerable[:eval_unanswerable]
    )
    calibration = (
        answerable[eval_answerable : eval_answerable + calib_answerable]
        + unanswerable[eval_unanswerable : eval_unanswerable + calib_unanswerable]
    )
    for row in evaluation:
        row["subset"] = "evaluation"
    for row in calibration:
        row["subset"] = "calibration"
    return evaluation, calibration


def write_freeze(
    archive: Path,
    output_dir: Path,
    seed: int = 20261107,
    evaluation_size: int = 80,
    calibration_size: int = 15,
) -> dict:
    papers = load_dev_papers(archive)
    questions = []
    for paper_id, paper in papers.items():
        questions.extend(questions_from_official_paper(paper_id, paper))
    evaluation, calibration = select_subset(
        questions,
        seed=seed,
        evaluation_size=evaluation_size,
        calibration_size=calibration_size,
    )
    eval_ids = {row["paper_id"] for row in evaluation}
    paper_records = []
    for paper_id in sorted(eval_ids):
        paper_records.append(_paper_record(paper_from_official(paper_id, papers[paper_id])))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "evaluation_questions.jsonl", evaluation)
    _write_jsonl(output_dir / "calibration_questions.jsonl", calibration)
    (output_dir / "papers.json").write_text(
        json.dumps(paper_records, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "source_url": SOURCE_PAGE,
        "archive_url": OFFICIAL_URL,
        "archive_member": DEV_MEMBER,
        "license": LICENSE_NOTE,
        "source_split": "validation",
        "official_dev_papers": len(papers),
        "official_dev_questions_with_annotators": len(questions),
        "evaluation_questions": len(evaluation),
        "evaluation_papers": len(eval_ids),
        "evaluation_unanswerable": sum(row["unanswerable"] for row in evaluation),
        "calibration_questions": len(calibration),
        "calibration_unanswerable": sum(row["unanswerable"] for row in calibration),
        "calibration_use": (
            "Held out. Not scored. Not used to choose the abstention threshold. "
            "The reported run leaves the threshold unset."
        ),
        "selection": (
            "Sort by paper id and question id. Shuffle answerable and "
            "unanswerable lists with random.Random(seed). Take a stratified "
            "quota that rounds evaluation_size times the dev unanswerable rate. "
            "Calibration is the next disjoint slice of each list."
        ),
        "random_seed": seed,
        "reference_answer": (
            "First annotator, per the order in the v0.3 JSON. Token F1 also "
            "reports the maximum over all annotators."
        ),
        "preprocessing": (
            "Abstract, full-text paragraphs, and figure/table captions are kept. "
            "Figure images are not downloaded. Chunks are 200 whitespace-separated "
            "words with overlap 40, split per section by the existing chunker. "
            "The indexed collection is only the papers that appear in the "
            "evaluation questions."
        ),
        "test_split": (
            "The official test split is not in qasper-train-dev-v0.3.tgz and "
            "is not scored."
        ),
    }
    (output_dir / "qasper_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_question_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_frozen_papers(path: Path) -> list[Paper]:
    return [paper_from_record(record) for record in json.loads(path.read_text(encoding="utf-8"))]


def _paper_record(paper: Paper) -> dict:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "sections": [{"name": section.name, "text": section.text} for section in paper.sections],
    }


def _sorted_unique(rows: list[dict]) -> list[dict]:
    by_id = {}
    for row in rows:
        by_id[row["question_id"]] = row
    return sorted(by_id.values(), key=lambda row: (row["paper_id"], row["question_id"]))


def _quota(size: int, unanswerable: int, total: int) -> int:
    if size == 0 or total == 0:
        return 0
    return int(round(size * unanswerable / total))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
