import ast
from typing import Any


def verify_finding(
    source_code: str,
    finding: dict[str, Any],
) -> dict[str, Any]:
    """Verify findings only when supported by independent static evidence.

    Never executes submitted source code.
    """
    result = {
        **finding,
        "verification": "not_verified",
        "verification_evidence": (
            "No independent verification rule matched this finding."
        ),
    }

    if finding.get("source") != "gemini_ai":
        result["verification"] = "tool_reported"
        result["verification_evidence"] = (
            "This finding was reported by an analysis tool. "
            "Its presence does not by itself establish a runtime defect."
        )
        return result

    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError, TypeError):
        result["verification_evidence"] = (
            "Verification could not inspect the source because parsing failed."
        )
        return result

    message = str(finding.get("message", "")).lower()
    reasoning = str(finding.get("reasoning", "")).lower()
    combined_text = f"{message} {reasoning}"

    # Narrow check: establish that a direct division expression exists.
    # This does not prove that a zero divisor can reach that expression.
    if (
        "zerodivisionerror" in combined_text
        or "divisor" in combined_text
        or "division by zero" in combined_text
    ):
        division_found = any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            for node in ast.walk(tree)
        )

        if division_found:
            result["verification"] = "requires_human_review"
            result["verification_evidence"] = (
                "The source contains a division expression. Static inspection "
                "does not establish whether its divisor can be zero or whether "
                "zero is invalid under the function's intended contract."
            )

    return result


def verify_findings(
    source_code: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return independently assessed copies of findings."""
    return [
        verify_finding(source_code, finding)
        for finding in findings
    ]