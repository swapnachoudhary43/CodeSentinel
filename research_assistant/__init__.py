"""Citation-grounded generative research assistant.

Three systems share one generator and one closed corpus:
b0 answers with no passages, b1 uses dense retrieval only, and proposed
adds hybrid retrieval, citation checks, and abstention.
"""

from research_assistant.pipeline import answer, compare
from research_assistant.types import ABSTAIN_MESSAGE

__all__ = ["ABSTAIN_MESSAGE", "answer", "compare"]
