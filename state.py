from typing import TypedDict


class CodeSentinelState(TypedDict, total=False):
    source_code: str
    language: str
    status: str
    errors: list[str]
    findings: list[dict]
    summary: str
    line_count: int
