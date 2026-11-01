
import inspect
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai_agent import review_python_code
from state import CodeSentinelState
from tools import run_ruff
from verification import verify_findings

MAX_SOURCE_LENGTH = 50_000
MAX_AI_RETRIES = 1
SUPPORTED_LANGUAGES = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "java": "java",
    "c": "c",
    "c++": "c++",
    "cpp": "c++",
    "cxx": "c++",
}
LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
    "c": "C",
    "c++": "C++",
}


def normalize_language(language: Any) -> str | None:
    if not isinstance(language, str):
        return None

    normalized = language.strip().lower()
    if not normalized:
        return None

    value = SUPPORTED_LANGUAGES.get(normalized)
    if value is not None:
        return value

    return None


def language_label(language: str | None) -> str:
    if not language:
        return "Python"
    normalized = normalize_language(language)
    if normalized is None:
        return str(language)
    return LANGUAGE_LABELS.get(normalized, normalized)


def _run_standalone_syntax_check(
    language: str,
    source_code: str,
) -> tuple[bool, str | None]:
    if language == "python":
        try:
            compile(source_code, "<submitted_code>", "exec")
            return True, None
        except SyntaxError as exc:
            return False, f"{exc.msg} (line {exc.lineno}, column {exc.offset})"
        except (TypeError, ValueError) as exc:
            return False, str(exc)

    if language == "javascript":
        node_path = shutil.which("node")
        if not node_path:
            return True, None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".js",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(source_code)

            command = [node_path, "--check", temp_path]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return True, None
            detail = (result.stderr or result.stdout).strip() or "Syntax check failed."
            return False, detail
        except subprocess.TimeoutExpired:
            return False, "Syntax check exceeded the 10-second limit."
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    if language in {"c", "c++"}:
        compiler = "gcc" if language == "c" else "g++"
        compiler_path = shutil.which(compiler)
        if not compiler_path:
            return True, None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".c" if language == "c" else ".cpp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(source_code)

            command = [compiler_path, "-x", "c" if language == "c" else "c++", "-fsyntax-only", temp_path]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return True, None
            detail = (result.stderr or result.stdout).strip() or "Syntax check failed."
            return False, detail
        except subprocess.TimeoutExpired:
            return False, "Syntax check exceeded the 10-second limit."
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    if language == "java":
        compiler_path = shutil.which("javac")
        if not compiler_path:
            return True, None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".java",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(source_code)

            command = [compiler_path, temp_path]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return True, None
            detail = (result.stderr or result.stdout).strip() or "Syntax check failed."
            return False, detail
        except subprocess.TimeoutExpired:
            return False, "Syntax check exceeded the 10-second limit."
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    return True, None


def record_agent(
    state: CodeSentinelState,
    agent: str,
    status: str,
    result: Any,
) -> dict:
    agent_status = dict(state.get("agent_status", {}))
    agent_results = dict(state.get("agent_results", {}))
    audit_log = list(state.get("audit_log", []))

    agent_status[agent] = status
    agent_results[agent] = result
    audit_log.append({
        "agent": agent,
        "status": status,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "agent_status": agent_status,
        "agent_results": agent_results,
        "audit_log": audit_log,
    }


def validate_input(state: CodeSentinelState) -> dict:
    source_code = state.get("source_code")
    language = normalize_language(state.get("language", "python"))

    if not isinstance(source_code, str) or not source_code.strip():
        result = {
            "status": "invalid_input",
            "errors": ["Source code must be a non-empty string."],
            "findings": [],
        }
        return {
            **result,
            **record_agent(state, "input_validation", "failed", result),
        }

    if len(source_code) > MAX_SOURCE_LENGTH:
        result = {
            "status": "invalid_input",
            "errors": [
                f"Source code exceeds the {MAX_SOURCE_LENGTH:,}-character limit."
            ],
            "findings": [],
        }
        return {
            **result,
            **record_agent(state, "input_validation", "failed", result),
        }

    if not language:
        result = {
            "status": "invalid_input",
            "errors": [
                "Unsupported language. Supported options: Python, JavaScript, Java, C, and C++."
            ],
            "findings": [],
        }
        return {
            **result,
            **record_agent(state, "input_validation", "failed", result),
        }

    result = {
        "source_code": source_code,
        "language": language,
        "status": "ready",
        "errors": [],
        "findings": [],
        "ai_summary": "",
        "line_count": len(source_code.splitlines()),
        "ai_retry_count": 0,
        "verification_feedback": [],
    }

    return {
        **result,
        **record_agent(state, "input_validation", "completed", "Input validated."),
    }


def route_after_validation(
    state: CodeSentinelState,
) -> Literal["syntax_check", "finalize"]:
    return "syntax_check" if state.get("status") == "ready" else "finalize"


def syntax_check(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"
    source_code = state["source_code"]

    ok, detail = _run_standalone_syntax_check(language, source_code)
    if not ok:
        source_name = language_label(language)
        rule_id = "PY-SYNTAX" if language == "python" else f"{language.upper().replace('+', 'PLUS')}-SYNTAX"
        result = {
            "status": "issues_found",
            "findings": [{
                "source": f"{language}_syntax",
                "rule_id": rule_id,
                "severity": "error",
                "line": 1,
                "column": 1,
                "message": detail or f"{source_name} syntax check failed.",
            }],
            "errors": [],
        }
        return {
            **result,
            **record_agent(state, "syntax_checker", "completed", result),
        }

    result = {"status": "syntax_passed", "findings": [], "errors": []}
    return {
        **result,
        **record_agent(state, "syntax_checker", "completed", result),
    }


def route_after_syntax(
    state: CodeSentinelState,
) -> Literal["static_analysis", "finalize"]:
    return (
        "static_analysis"
        if state.get("status") == "syntax_passed"
        else "finalize"
    )


def static_analysis(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"

    if language != "python":
        output = {
            "status": "analysis_passed",
            "findings": [],
            "errors": [],
        }
        return {
            **output,
            **record_agent(
                state,
                "static_analysis_agent",
                "completed",
                {"finding_count": 0, "findings": [], "language": language},
            ),
        }

    result = run_ruff(state["source_code"])

    if not result["ok"]:
        output = {
            "status": "analysis_error",
            "errors": [result["error"]],
            "findings": [],
        }
        return {
            **output,
            **record_agent(state, "static_analysis_agent", "failed", output),
        }

    findings = result["findings"]
    output = {
        "status": "issues_found" if findings else "analysis_passed",
        "findings": findings,
        "errors": [],
    }

    return {
        **output,
        **record_agent(
            state,
            "static_analysis_agent",
            "completed",
            {"finding_count": len(findings), "findings": findings},
        ),
    }


def call_review_function(
    source_code: str,
    verification_feedback: list[dict[str, Any]] | None = None,
    language: str = "python",
) -> dict[str, Any]:
    try:
        parameters = inspect.signature(review_python_code).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "language" in parameters:
        return review_python_code(
            source_code,
            verification_feedback=verification_feedback,
            language=language,
        )

    return review_python_code(
        source_code,
        verification_feedback=verification_feedback,
    )


def ai_review(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"
    result = call_review_function(
        state["source_code"],
        verification_feedback=None,
        language=language,
    )

    existing_findings = list(state.get("findings", []))
    existing_errors = list(state.get("errors", []))

    if not result["ok"]:
        error = result.get("error") or "Gemini review failed."
        existing_errors.append(error)

        output = {
            "status": (
                "analysis_error"
                if state.get("status") == "analysis_error"
                and not existing_findings
                else "partial_analysis"
            ),
            "findings": existing_findings,
            "errors": existing_errors,
            "ai_summary": result.get(
                "summary", "Gemini review could not be completed."
            ),
        }

        return {
            **output,
            **record_agent(state, "ai_review_agent", "failed", error),
        }

    ai_findings = result.get("findings", [])
    ai_summary = result.get("summary", "")

    summary_inconsistency = (
        not ai_findings
        and bool(ai_summary.strip())
        and any(
            phrase in ai_summary.lower()
            for phrase in (
                "may raise",
                "could raise",
                "potential issue",
                "potential bug",
                "lacks handling",
                "does not handle",
                "not handled",
                "division by zero",
                "zerodivisionerror",
            )
        )
    )

    combined = existing_findings + ai_findings

    if summary_inconsistency:
        existing_errors.append(
            "Gemini summary/findings inconsistency: "
            "the summary describes a potential issue, "
            "but the structured findings list is empty."
        )

    output = {
        "status": (
            "partial_analysis"
            if existing_errors or summary_inconsistency
            else "issues_found"
            if combined
            else "analysis_passed"
        ),
        "findings": combined,
        "errors": existing_errors,
        "ai_summary": ai_summary,
    }

    return {
        **output,
        **record_agent(
            state,
            "ai_review_agent",
            "completed",
            {
                "finding_count": len(ai_findings),
                "summary": ai_summary,
                "summary_findings_inconsistent": summary_inconsistency,
            },
        ),
    }




def verify_analysis_findings(state: CodeSentinelState) -> dict:
    verified = verify_findings(
        state.get("source_code", ""),
        state.get("findings", []),
    )

    feedback = [
        {
            "rule_id": item.get("rule_id"),
            "message": item.get("message"),
            "verification": item.get("verification"),
            "verification_evidence": item.get("verification_evidence"),
        }
        for item in verified
        if item.get("source") == "gemini_ai"
        and item.get("verification") == "not_verified"
    ]

    human_review_findings = [
        item
        for item in verified
        if item.get("verification") == "requires_human_review"
    ]

    return {
        "findings": verified,
        "verification_feedback": feedback,
        **record_agent(
            state,
            "verification_agent",
            "completed",
            {
                "finding_count": len(verified),
                "unverified_ai_findings": len(feedback),
                "requires_human_review_count": len(human_review_findings),
            },
        ),
    }



def route_after_verification(
    state: CodeSentinelState,
) -> Literal["ai_reassessment", "prepare_review"]:
    retry_count = state.get("ai_retry_count", 0)
    feedback = state.get("verification_feedback", [])

    if feedback and retry_count < MAX_AI_RETRIES:
        return "ai_reassessment"

    return "prepare_review"


def ai_reassessment(state: CodeSentinelState) -> dict:
    """Ask Gemini once to reassess findings using verifier evidence."""
    next_retry = state.get("ai_retry_count", 0) + 1

    try:
        result = call_review_function(
            state["source_code"],
            verification_feedback=state.get("verification_feedback", []),
            language=normalize_language(state.get("language", "python")) or "python",
        )
    except Exception as exc:
        result = {
            "ok": False,
            "findings": [],
            "summary": "Gemini reassessment could not be completed.",
            "error": str(exc),
        }

    existing = list(state.get("findings", []))
    ruff_findings = [
        item for item in existing if item.get("source") != "gemini_ai"
    ]
    errors = list(state.get("errors", []))

    if not result.get("ok"):
        error = result.get("error") or "Gemini reassessment failed."
        errors.append(error)

        return {
            "ai_retry_count": next_retry,
            "errors": errors,
            "status": "partial_analysis",
            **record_agent(
                state,
                "ai_reassessment_agent",
                "failed",
                {
                    "retry": next_retry,
                    "error": error,
                    "previous_findings_preserved": True,
                },
            ),
        }

    revised_findings = result.get("findings", [])
    combined = ruff_findings + revised_findings

    return {
        "ai_retry_count": next_retry,
        "findings": combined,
        "ai_summary": result.get("summary", state.get("ai_summary", "")),
        "status": (
            "partial_analysis"
            if errors
            else "issues_found"
            if combined
            else "analysis_passed"
        ),
        **record_agent(
            state,
            "ai_reassessment_agent",
            "completed",
            {
                "retry": next_retry,
                "finding_count": len(revised_findings),
                "summary": result.get("summary", ""),
            },
        ),
    }


def prepare_review(state: CodeSentinelState) -> dict:
    review_required = bool(
        state.get("require_human_review") and state.get("findings")
    )

    return {
        "review_required": review_required,
        "review_completed": False,
        **record_agent(
            state,
            "human_review",
            "pending" if review_required else "skipped",
            (
                "Waiting for human review."
                if review_required
                else "Human review was not requested or no findings exist."
            ),
        ),
    }


def route_after_review_preparation(
    state: CodeSentinelState,
) -> Literal["human_review", "finalize"]:
    return "human_review" if state.get("review_required") else "finalize"


def human_review(state: CodeSentinelState) -> dict:
    response = interrupt({
        "type": "human_review_required",
        "message": "Review the analysis findings before completing this workflow.",
        "findings": state.get("findings", []),
        "ai_summary": state.get("ai_summary", ""),
        "line_count": state.get("line_count", 0),
    })

    if not isinstance(response, dict):
        raise TypeError("Human review response must be an object.")

    decision = response.get("decision")
    if decision not in ("approve", "reject"):
        raise ValueError("Review decision must be 'approve' or 'reject'.")

    feedback = response.get("feedback", "")
    if not isinstance(feedback, str):
        raise TypeError("Review feedback must be a string.")

    return {
        "review_required": False,
        "review_decision": decision,
        "review_feedback": feedback,
        "review_completed": True,
        **record_agent(
            state,
            "human_review",
            "approved" if decision == "approve" else "rejected",
            {"decision": decision, "feedback": feedback},
        ),
    }


def finalize(state: CodeSentinelState) -> dict:
    status = state.get("status", "unknown")
    findings = state.get("findings", [])
    errors = state.get("errors", [])
    ai_summary = state.get("ai_summary", "")
    language = normalize_language(state.get("language", "python")) or "python"
    language_name = language_label(language)

    if status == "invalid_input":
        summary = "Analysis not run: input validation failed."

    elif status == "analysis_error":
        summary = (
            "Analysis incomplete: no findings were produced by the available "
            "analysis steps. Check the reported errors."
        )

    elif status == "partial_analysis":
        summary = (
            f"Analysis partially completed. {len(findings)} finding(s) "
            "were collected from the checks that completed successfully."
        )
        if ai_summary:
            summary += f" Gemini summary: {ai_summary}"
        if errors:
            summary += " Failed step(s): " + " | ".join(errors)
        summary += " Findings may require independent verification."

    elif status == "issues_found":
        syntax_findings = [
            item for item in findings
            if item.get("source", "").endswith("_syntax")
            or item.get("source") == "python_compile"
        ]

        if syntax_findings:
            summary = (
                f"{language_name} syntax check found {len(syntax_findings)} "
                "syntax issue(s). Further analysis was not run."
            )
        else:
            summary = (
                f"Analysis reported {len(findings)} finding(s) across "
                "the available checks."
            )
            if ai_summary:
                summary += f" Gemini summary: {ai_summary}"
            summary += (
                " AI findings are potential issues, not independently verified."
            )

        summary += " No source code was executed."

    elif status == "analysis_passed":
        if language == "python":
            summary = (
                "Python syntax and Ruff checks passed."
                f" Gemini summary: {ai_summary}"
                " Passing these checks does not establish logical correctness "
                "or security. No source code was executed."
            )
        else:
            summary = (
                f"{language_name} syntax validation passed."
                f" Gemini summary: {ai_summary}"
                " Passing these checks does not establish logical correctness "
                "or security. No source code was executed."
            )

    else:
        summary = (
            "Analysis incomplete: the workflow ended in an unexpected state."
        )

    decision = state.get("review_decision")
    if state.get("review_completed"):
        summary += (
            f" Human review decision: {decision}."
            " This records the reviewer's decision; it does not independently "
            "prove or disprove the findings."
        )
        feedback = state.get("review_feedback", "")
        if feedback:
            summary += f" Reviewer feedback: {feedback}"

    summary += (
        f" Gemini reassessment attempts: {state.get('ai_retry_count', 0)}."
    )

    return {
        "summary": summary,
        **record_agent(
            state,
            "report_agent",
            "completed",
            {"finding_count": len(findings), "summary": summary},
        ),
    }


def build_graph(checkpointer: Any = None):
    builder = StateGraph(CodeSentinelState)

    builder.add_node("validate_input", validate_input)
    builder.add_node("syntax_check", syntax_check)
    builder.add_node("static_analysis", static_analysis)
    builder.add_node("ai_review", ai_review)
    builder.add_node("verify_findings", verify_analysis_findings)
    builder.add_node("ai_reassessment", ai_reassessment)
    builder.add_node("prepare_review", prepare_review)
    builder.add_node("human_review", human_review)
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

    builder.add_edge("static_analysis", "ai_review")
    builder.add_edge("ai_review", "verify_findings")

    builder.add_conditional_edges(
        "verify_findings",
        route_after_verification,
        {
            "ai_reassessment": "ai_reassessment",
            "prepare_review": "prepare_review",
        },
    )

    builder.add_edge("ai_reassessment", "verify_findings")

    builder.add_conditional_edges(
        "prepare_review",
        route_after_review_preparation,
        {
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )

    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


code_sentinel_graph = build_graph()
