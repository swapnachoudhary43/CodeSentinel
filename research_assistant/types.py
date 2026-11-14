from dataclasses import dataclass


ABSTAIN_MESSAGE = "Insufficient evidence in the indexed papers."
SYSTEMS = ("b0", "b1", "proposed")
ABLATIONS = (
    "proposed_no_verifier",
    "proposed_bm25_only",
    "proposed_dense_only",
)


@dataclass(frozen=True)
class Section:
    name: str
    text: str


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    year: int | None
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    year: int | None
    section: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class AnswerResult:
    system: str
    question: str
    answer: str
    abstained: bool
    raw_answer: str
    citations: tuple[str, ...]
    retrieved: tuple[Hit, ...]
    dropped: tuple[str, ...]
    citation_accuracy: float | None
    generator: str
    retrieval_mode: str
    verifications: tuple = ()
