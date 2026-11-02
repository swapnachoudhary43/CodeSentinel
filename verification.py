import ast
from typing import Any


def _compile_ast(source_code: str):
    try:
        return ast.parse(source_code)
    except (SyntaxError, TypeError, ValueError):
        return None


def _iter_function_ranges(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node


def verify_finding(
    source_code: str,
    finding: dict[str, Any],
) -> dict[str, Any]:
    """Verify findings only when supported by independent static evidence.

    Never executes submitted source code.
    """
    source = str(finding.get("source") or "")
    rule_id = str(finding.get("rule_id") or "").upper()
    tool_status = str(
        finding.get("tool_status") or finding.get("execution_status") or ""
    ).upper()

    result = {
        **finding,
        "verification": "NOT_CHECKED",
        "verification_evidence": (
            "No independent verification rule matched this finding."
        ),
    }

    if source != "gemini_ai":
        syntax_sources = {
            "python_compile",
            "javac",
            "node",
            "clang",
            "gcc",
            "compiler",
        }
        is_syntax = (
            "SYNTAX" in rule_id
            or source.endswith("_syntax")
            or source in syntax_sources
        )
        if is_syntax and tool_status != "UNAVAILABLE":
            result["verification"] = "VERIFIED"
            result["verification_evidence"] = (
                "A syntax checker produced this diagnostic. "
                "The checker does not execute the submitted program."
            )
            return result

        result["verification"] = "NOT_CHECKED"
        result["verification_evidence"] = (
            "This finding was reported by an analysis tool. "
            "Its presence does not by itself establish a runtime defect."
        )
        return result

    tree = _compile_ast(source_code)
    if tree is None:
        result["verification"] = "NOT_CHECKED"
        result["verification_evidence"] = (
            "Verification could not inspect the source because parsing failed."
        )
        return result

    message = str(finding.get("message", "")).lower()
    reasoning = str(finding.get("reasoning", "")).lower()
    combined_text = f"{message} {reasoning}"

    if any(token in combined_text for token in ["division by zero", "zerodivisionerror", "divide by zero", "divisor"]):
        division_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        ]
        literal_zero = any(
            isinstance(node.right, ast.Constant) and node.right.value == 0
            for node in division_nodes
        )
        if literal_zero:
            result["verification"] = "VERIFIED"
            result["verification_evidence"] = (
                "The parsed source contains a division by a literal zero."
            )
            return result
        if division_nodes:
            result["verification"] = "UNRESOLVED"
            result["verification_evidence"] = (
                "The source contains a division expression, but static analysis cannot prove "
                "that the divisor is ever zero or that the code intends to reject zero."
            )
            return result
        result["verification"] = "CONTRADICTED"
        result["verification_evidence"] = (
            "The finding claims a division risk, but the parsed source contains no division expression."
        )
        return result

    if any(token in combined_text for token in ["null", "none", "attributeerror", "keyerror"]):
        lines = source_code.splitlines()
        location = finding.get("line")
        if isinstance(location, int) and lines:
            target_line = min(max(location, 1), len(lines))
            line_text = lines[target_line - 1].strip()
            if line_text and any(token in line_text.lower() for token in [".", "[", "get(", "append(", "split("]):
                result["verification"] = "UNRESOLVED"
                result["verification_evidence"] = (
                    "The finding mentions a null or attribute risk, but the code structure alone "
                    "does not establish that the suspect object can be None at runtime."
                )
                return result
        result["verification"] = "CONTRADICTED"
        result["verification_evidence"] = (
            "The finding claims a null or attribute risk, but the parsed source does not show a matching access."
        )
        return result

    if any(token in combined_text for token in ["index out of range", "indexerror", "list index", "subscript"]):
        slice_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
        ]
        if slice_nodes:
            result["verification"] = "UNRESOLVED"
            result["verification_evidence"] = (
                "The source uses indexed access, but static analysis cannot confirm the index bounds."
            )
            return result
        result["verification"] = "CONTRADICTED"
        result["verification_evidence"] = (
            "The finding claims an indexing risk, but the parsed source contains no indexed access."
        )
        return result

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