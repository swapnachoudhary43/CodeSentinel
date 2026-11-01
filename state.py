
from typing import Any, TypedDict


class CodeSentinelState(TypedDict, total=False):
    source_code: str
    language: str
    line_count: int

    status: str
    errors: list[str]
    findings: list[dict[str, Any]]

    summary: str
    ai_summary: str

    agent_status: dict[str, str]
    agent_results: dict[str, Any]
    audit_log: list[dict[str, Any]]

    require_human_review: bool
    review_required: bool
    review_decision: str
    review_feedback: str
    review_completed: bool

    # Bounded self-correction
    ai_retry_count: int
    verification_feedback: list[dict[str, Any]]
