import os
import re
from typing import Protocol

from research_assistant.retrieve import tokenize
from research_assistant.types import ABSTAIN_MESSAGE, Chunk, Paper


_CITATION = re.compile(r"\[[A-Za-z0-9_.:-]+\]")


class ResearchModel(Protocol):
    name: str

    def generate_answer(self, question: str, chunks: list[Chunk] | None) -> str:
        """Answer a question. chunks is None for the no-retrieval baseline."""

    def judge_support(self, sentence: str, chunk: Chunk) -> str:
        """Return SUPPORTED or NOT_SUPPORTED."""

    def literature_row(self, paper: Paper, chunks: list[Chunk]) -> str:
        """Return one literature-matrix row."""


def answer_prompt(question: str, chunks: list[Chunk] | None) -> str:
    if chunks is None:
        return (
            "You are a research assistant with no retrieved passages.\n"
            "Answer the question from memory in one or two sentences.\n"
            "If you name a source, put it in square brackets. You were not "
            "given any passage from the indexed corpus.\n\n"
            f"Question: {question}"
        )
    if len(chunks) == 0:
        return (
            "You are a research assistant. Retrieval ran and returned no passages.\n"
            "Do not answer from memory. Reply with exactly:\n"
            f"{ABSTAIN_MESSAGE}\n\n"
            f"Question: {question}"
        )

    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.chunk_id}] {chunk.title} / {chunk.section}\n{chunk.text}"
        )
    passages = "\n\n".join(blocks)
    return (
        "You are a citation-grounded research assistant.\n"
        "Use only the passages below. Every factual sentence must end with "
        "exactly one chunk id in square brackets, before the period.\n"
        "Example: Reciprocal rank fusion combines two ranked lists [paper:1].\n"
        "If the passages do not contain the answer, reply with exactly:\n"
        f"{ABSTAIN_MESSAGE}\n\n"
        f"Passages:\n{passages}\n\n"
        f"Question: {question}"
    )


def support_prompt(sentence: str, chunk: Chunk) -> str:
    claim = _CITATION.sub("", sentence).strip()
    return (
        "Decide whether the passage states the claim.\n"
        "Reply with exactly one token: SUPPORTED or NOT_SUPPORTED.\n"
        "SUPPORTED means the claim is stated in the passage, not merely related.\n\n"
        f"Passage [{chunk.chunk_id}]:\n{chunk.text}\n\n"
        f"Claim: {claim}"
    )


def matrix_prompt(paper: Paper, chunks: list[Chunk]) -> str:
    excerpts = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)
    year = paper.year if paper.year is not None else "not stated"
    return (
        "Extract one literature-matrix row from the excerpts.\n"
        "If a field is absent, write not stated. Do not guess.\n"
        f"Title: {paper.title}\n"
        f"Year: {year}\n"
        "Method:\n"
        "Dataset:\n"
        "Metric:\n"
        "Limitation:\n\n"
        f"Excerpts:\n{excerpts}"
    )


def _retry_wait(message: str, attempt: int) -> float:
    import re

    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return 65.0
    return float(min(20, 2**attempt))


def parse_support(text: str) -> str:
    """Fail closed. Only a first token of SUPPORTED counts as support."""
    stripped = text.strip().upper()
    if not stripped:
        return "NOT_SUPPORTED"
    first = re.split(r"\s+", stripped, maxsplit=1)[0]
    first = re.sub(r"[^A-Z_]", "", first)
    if first == "SUPPORTED":
        return "SUPPORTED"
    return "NOT_SUPPORTED"


class ExtractiveModel:
    """Quotes the top passage. Offline stand-in, not a Gemini result."""

    name = "extractive"

    def generate_answer(self, question: str, chunks: list[Chunk] | None) -> str:
        del question
        if not chunks:
            return "This reply was produced with no indexed passage [ungrounded:0]."
        chunk = chunks[0]
        first = re.split(r"(?<=[.!?])\s+", chunk.text.strip())[0]
        snippet = first.strip().rstrip(".!?")
        return f"{snippet} [{chunk.chunk_id}]."

    def judge_support(self, sentence: str, chunk: Chunk) -> str:
        claim = tokenize(_CITATION.sub("", sentence))
        if not claim:
            return "NOT_SUPPORTED"
        source = set(tokenize(chunk.text))
        overlap = sum(1 for token in claim if token in source) / len(claim)
        if overlap >= 0.5:
            return "SUPPORTED"
        return "NOT_SUPPORTED"

    def literature_row(self, paper: Paper, chunks: list[Chunk]) -> str:
        blob = " ".join(chunk.text for chunk in chunks).lower()
        if "reciprocal rank fusion" in blob:
            method = "reciprocal rank fusion of keyword and dense lists"
        elif "convolutional" in blob:
            method = "convolutional network for leaf images"
        else:
            method = "not stated"
        dataset = "field photographs of leaves" if "photograph" in blob else "not stated"
        metric = "accuracy" if "accuracy" in blob else "not stated"
        if "dark images" in blob:
            limitation = "fails on dark images"
        elif "development set" in blob:
            limitation = "threshold is chosen on a development set"
        else:
            limitation = "not stated"
        year = paper.year if paper.year is not None else "not stated"
        return (
            f"Title: {paper.title}\n"
            f"Year: {year}\n"
            f"Method: {method}\n"
            f"Dataset: {dataset}\n"
            f"Metric: {metric}\n"
            f"Limitation: {limitation}"
        )


class GeminiModel:
    """One Gemini model for every system. Temperature is 0."""

    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
        cache_path: str | None = None,
        retries: int = 1,
    ):
        self.api_key = api_key
        self.model_name = model_name or os.getenv(
            "RESEARCH_ASSISTANT_MODEL",
            "gemini-3.5-flash-lite",
        )
        self.name = self.model_name
        self.cache_path = cache_path
        self.retries = max(1, retries)
        self.api_calls = 0
        self.cache_hits = 0
        self.api_errors: list[str] = []
        self._client = None
        self._cache: dict[str, str] | None = None
        self._last_call = 0.0

    def generate_answer(self, question: str, chunks: list[Chunk] | None) -> str:
        return self._complete(answer_prompt(question, chunks))

    def judge_support(self, sentence: str, chunk: Chunk) -> str:
        return parse_support(self._complete(support_prompt(sentence, chunk)))

    def literature_row(self, paper: Paper, chunks: list[Chunk]) -> str:
        return self._complete(matrix_prompt(paper, chunks))

    def _complete(self, prompt: str) -> str:
        import hashlib
        import time

        from google import genai
        from google.genai import types

        key = hashlib.sha256(
            f"{self.model_name}\n{prompt}".encode("utf-8")
        ).hexdigest()
        cached = self._read_cache().get(key)
        if cached:
            self.cache_hits += 1
            return cached

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        last_error = "Gemini request failed."
        for attempt in range(1, self.retries + 1):
            self._pace()
            self.api_calls += 1
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0),
                )
            except Exception as exc:
                last_error = f"Gemini request failed: {type(exc).__name__}: {exc}"
                self.api_errors.append(last_error)
                if attempt == self.retries:
                    raise RuntimeError(last_error) from exc
                time.sleep(_retry_wait(str(exc), attempt))
                continue
            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                last_error = "Gemini returned an empty response."
                self.api_errors.append(last_error)
                if attempt == self.retries:
                    raise RuntimeError(last_error)
                time.sleep(min(20, 2 ** attempt))
                continue
            cleaned = str(text).strip()
            self._write_cache(key, cleaned)
            return cleaned
        raise RuntimeError(last_error)

    def _pace(self) -> None:
        import time

        elapsed = time.time() - self._last_call
        if self._last_call and elapsed < 4.5:
            time.sleep(4.5 - elapsed)
        self._last_call = time.time()

    def _read_cache(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        if not self.cache_path:
            return self._cache
        import json
        from pathlib import Path

        path = Path(self.cache_path)
        if path.exists():
            self._cache = json.loads(path.read_text(encoding="utf-8"))
        return self._cache

    def _write_cache(self, key: str, text: str) -> None:
        import json
        from pathlib import Path

        cache = self._read_cache()
        cache[key] = text
        if not self.cache_path:
            return
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
