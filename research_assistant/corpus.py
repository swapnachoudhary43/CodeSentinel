import json
from pathlib import Path

from research_assistant.types import Chunk, Paper, Section


def chunk_paper(paper: Paper, chunk_words: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split each section on whitespace. Sections are not mixed in one chunk.

    Words stand in for tokens so the chunker does not need a model tokenizer.
    all-MiniLM-L6-v2 truncates past 256 word-pieces, so the default window
    stays at 200 words with a 40-word overlap.
    """
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1.")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be zero or a positive value below chunk_words.")

    chunks: list[Chunk] = []
    for section in paper.sections:
        words = section.text.split()
        if not words:
            continue
        for piece in _windows(words, chunk_words, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}:{len(chunks)}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                    section=section.name,
                    text=" ".join(piece),
                )
            )
    return chunks


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    step = size - overlap
    pieces: list[list[str]] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        pieces.append(words[start:end])
        if end == len(words):
            break
        start += step
    return pieces


def paper_from_record(record: dict) -> Paper:
    paper_id = str(record.get("paper_id") or "").strip()
    title = str(record.get("title") or "").strip()
    if not paper_id or not title:
        raise ValueError("Each paper needs a paper_id and a title.")
    raw_sections = record.get("sections") or []
    if not raw_sections:
        raise ValueError(f"{paper_id} has no sections.")
    sections = []
    for section in raw_sections:
        name = str(section.get("name") or "Body").strip() or "Body"
        text = str(section.get("text") or "").strip()
        if text:
            sections.append(Section(name, text))
    if not sections:
        raise ValueError(f"{paper_id} has no section text.")
    year = record.get("year")
    return Paper(paper_id, title, int(year) if year is not None else None, tuple(sections))


def load_papers(path: Path) -> list[Paper]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The corpus file must be a JSON list of papers.")
    return [paper_from_record(record) for record in payload]


def load_sample_papers() -> list[Paper]:
    return load_papers(Path(__file__).with_name("sample_papers.json"))


def paper_from_qasper_row(row: dict) -> Paper:
    """Map one allenai/qasper row into a Paper.

    The expected shape matches the dataset card: full_text.section_name and
    full_text.paragraphs. Confirm a downloaded row before the graded run.
    """
    paper_id = str(row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    if not paper_id or not title:
        raise ValueError("Qasper row needs an id and a title.")

    sections: list[Section] = []
    abstract = str(row.get("abstract") or "").strip()
    if abstract:
        sections.append(Section("Abstract", abstract))

    full_text = row.get("full_text") or {}
    names = list(full_text.get("section_name") or [])
    paragraphs = list(full_text.get("paragraphs") or [])
    for name, paras in zip(names, paragraphs):
        if isinstance(paras, str):
            text = paras
        else:
            text = "\n".join(str(part) for part in paras if str(part).strip())
        text = text.strip()
        if text:
            sections.append(Section(str(name or "Body").strip() or "Body", text))

    if not sections:
        raise ValueError(f"{paper_id} has no text.")
    return Paper(paper_id, title, None, tuple(sections))


def questions_from_qasper_row(row: dict) -> list[dict]:
    qas = row.get("qas") or {}
    questions = list(qas.get("question") or [])
    question_ids = list(qas.get("question_id") or [])
    groups = list(qas.get("answers") or [])
    parsed = []
    for index, question in enumerate(questions):
        group = groups[index] if index < len(groups) else {}
        answer_list = group.get("answer") if isinstance(group, dict) else []
        gold = _gold_answer(answer_list or [])
        parsed.append(
            {
                "paper_id": row.get("id"),
                "question_id": question_ids[index] if index < len(question_ids) else f"{row.get('id')}:{index}",
                "question": question,
                **gold,
            }
        )
    return parsed


def _gold_answer(answer_list: list) -> dict:
    if not answer_list:
        return {"unanswerable": True, "answer": "", "evidence": []}
    first = answer_list[0] or {}
    spans = [str(span) for span in (first.get("extractive_spans") or []) if str(span).strip()]
    free_form = str(first.get("free_form_answer") or "").strip()
    text = free_form or (spans[0] if spans else "")
    yes_no = first.get("yes_no")
    if yes_no is True and not text:
        text = "yes"
    if yes_no is False and not text:
        text = "no"
    evidence = [str(item) for item in (first.get("evidence") or [])]
    return {
        "unanswerable": bool(first.get("unanswerable")),
        "answer": text,
        "evidence": evidence,
    }


def load_qasper_split(split: str = "validation", limit: int | None = None) -> list[dict]:
    """Download Qasper when the datasets package is installed.

    Tests do not call this. The graded run should freeze the chosen rows to
    a local JSON file before any prompt is edited.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the datasets package to download Qasper: pip install datasets"
        ) from exc

    dataset = load_dataset("allenai/qasper", split=split)
    rows = []
    for row in dataset:
        rows.append(
            {
                "paper": paper_from_qasper_row(row),
                "questions": questions_from_qasper_row(row),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
