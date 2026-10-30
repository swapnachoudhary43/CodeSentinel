from typing import Literal

from langgraph.graph import END, START, StateGraph

from state import CodeSentinelState
from tools import run_ruff

MAX_SOURCE_LENGTH = 50_000


def validate_input(state: CodeSentinelState) -> dict:
    """Validate submitted source and initialize the analysis state."""
    source_code = state.get("source_code")
    language = state.get("language", "python")

    if not isinstance(source_code, str) or not source_code.strip():
        return {
            "status": "invalid_input",
            "errors": ["Source code must be a non-empty string."],
            "findings": [],
        }

    if len(source_code) > MAX_SOURCE_LENGTH:
        return {
            "status": "invalid_input",
            "errors": [
                f"Source code exceeds the {MAX_SOURCE_LENGTH:,}-character limit."
            ],
            "findings": [],
        }

    if not isinstance(language, str) or language.strip().lower() != "python":
        return {
            "status": "invalid_input",
            "errors": ["This initial version supports Python source only."],
            "findings": [],
        }

    return {
        "source_code": source_code,
        "language": "python",
        "status": "ready",
        "errors": [],
        "findings": [],
        "line_count": len(source_code.splitlines()),
    }


def route_after_validation(
    state: CodeSentinelState,
) -> Literal["syntax_check", "finalize"]:
    """Route valid input to analysis and invalid input to finalization."""
    if state.get("status") == "ready":
        return "syntax_check"
    return "finalize"


def syntax_check(state: CodeSentinelState) -> dict:
    """Check Python syntax without executing the submitted source."""
    try:
        compile(state["source_code"], "<submitted_code>", "exec")
    except SyntaxError as exc:
        return {
            "status": "issues_found",
            "findings": [
                {
                    "source": "python_compile",
                    "rule_id": "PY-SYNTAX",
                    "severity": "error",
                    "line": exc.lineno,
                    "column": exc.offset,
                    "message": exc.msg,
                }
            ],
            "errors": [],
        }
    except (ValueError, TypeError) as exc:
        return {
            "status": "analysis_error",
            "errors": [f"Python compilation check failed: {exc}"],
            "findings": [],
        }

    return {"status": "syntax_passed", "findings": [], "errors": []}


def route_after_syntax(
    state: CodeSentinelState,
) -> Literal["static_analysis", "finalize"]:
    """Run Ruff only when the syntax check succeeds."""
    if state.get("status") == "syntax_passed":
        return "static_analysis"
    return "finalize"


def static_analysis(state: CodeSentinelState) -> dict:
    """Run Ruff and translate its result into graph state."""
    result = run_ruff(state["source_code"])

    if not result["ok"]:
        return {
            "status": "analysis_error",
            "errors": [result["error"]],
            "findings": [],
        }

    findings = result["findings"]

    return {
        "status": "issues_found" if findings else "analysis_passed",
        "findings": findings,
        "errors": [],
    }


def finalize(state: CodeSentinelState) -> dict:
    """Report what the workflow actually established."""
    status = state.get("status", "unknown")
    findings = state.get("findings", [])

    if status == "invalid_input":
        summary = "Analysis not run: input validation failed."
    elif status == "analysis_error":
        summary = (
            "Analysis incomplete: a tool or analysis step failed. "
            "Check the reported error."
        )
    elif status == "issues_found":
        syntax_findings = [
            finding
            for finding in findings
            if finding.get("source") == "python_compile"
        ]

        if syntax_findings:
            summary = (
                f"Python compilation check found {len(syntax_findings)} "
                "syntax issue(s). Ruff was not run. No source code was executed."
            )
        else:
            summary = (
                f"Ruff reported {len(findings)} finding(s). "
                "No source code was executed."
            )
    elif status == "analysis_passed":
        summary = (
            "Python syntax and Ruff checks passed. "
            "This does not establish logical correctness or security. "
            "No source code was executed."
        )
    else:
        summary = "Analysis incomplete: the workflow ended in an unexpected state."

    return {"summary": summary}


def build_graph():
    """Build and compile the CodeSentinel analysis graph."""
    builder = StateGraph(CodeSentinelState)

    builder.add_node("validate_input", validate_input)
    builder.add_node("syntax_check", syntax_check)
    builder.add_node("static_analysis", static_analysis)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "validate_input")

    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "syntax_check": "syntax_check",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "syntax_check",
        route_after_syntax,
        {
            "static_analysis": "static_analysis",
            "finalize": "finalize",
        },
    )

    builder.add_edge("static_analysis", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


code_sentinel_graph = build_graph()
