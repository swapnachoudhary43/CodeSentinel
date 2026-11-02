
import ast
import hashlib
import inspect
import re
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai_agent import review_python_code
from state import CodeSentinelState
from tools import run_ruff, run_syntax_check
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
REVIEW_DECISIONS = {
    "approve",
    "approve_with_warnings",
    "request_changes",
    "reject",
    "defer",
}
INCOMPLETE_SUMMARY = "INSUFFICIENT EVIDENCE — ANALYSIS INCOMPLETE"


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
) -> tuple[bool, str | None, str]:
    if not isinstance(source_code, str) or not source_code.strip():
        return False, "Source code must be a non-empty string.", "FAILED"

    result = run_syntax_check(source_code, language)
    status = result.get("status", "FAILED")

    if result.get("ok") is False:
        return False, result.get("error") or "Syntax check failed.", status

    findings = result.get("findings") or []
    if not findings:
        return True, None, status

    first = findings[0]
    detail = str(first.get("message") or "Syntax check failed.")
    return False, detail, status


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


def _finding_identity(finding: dict) -> str:
    source = str(finding.get("source") or "unknown")
    rule_id = str(finding.get("rule_id") or "unknown")
    message = str(finding.get("message") or "")
    line = str(finding.get("line") or "")
    return f"{source}:{rule_id}:{line}:{message.strip()}"


def assign_finding_id(finding: dict) -> dict:
    """Give a finding a stable id derived from its identity."""
    if finding.get("finding_id"):
        return finding

    digest = hashlib.sha256(
        _finding_identity(finding).encode("utf-8")
    ).hexdigest()[:16]
    return {**finding, "finding_id": f"f-{digest}"}


def deduplicate_findings(existing: list[dict], updated: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()

    for finding in [*existing, *updated]:
        identity = _finding_identity(finding)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(assign_finding_id(finding))

    return unique


def _merge_agent_findings(
    state: CodeSentinelState,
    agent_name: str,
    findings: list[dict],
    *,
    extra: dict | None = None,
) -> dict:
    merged = deduplicate_findings(list(state.get("findings", [])), findings)
    payload = {"findings": merged}
    if extra:
        payload.update(extra)
    return {
        **payload,
        **record_agent(state, agent_name, "completed", {"finding_count": len(findings), **(extra or {})}),
    }


def _merge_specialized_results(state: CodeSentinelState, agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    combined = dict(state.get("specialized_agent_results", {}))
    combined[agent_name] = payload
    return {"specialized_agent_results": combined}


def _python_context(source_code: str) -> dict:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {"imports": [], "functions": [], "classes": [], "relevant_code_locations": []}

    imports: list[str] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                imports.append(module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args],
            })
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": [
                    child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
            })

    relevant = []
    for item in functions[:10]:
        relevant.append({"kind": "function", "name": item["name"], "line": item["line"]})
    for item in classes[:10]:
        relevant.append({"kind": "class", "name": item["name"], "line": item["line"]})

    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "relevant_code_locations": relevant,
    }


def input_context_agent(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"
    source_code = state.get("source_code", "")

    context = {
        "validation_status": "validated",
        "detected_language": language,
        "imports": [],
        "functions": [],
        "classes": [],
        "relevant_code_locations": [],
    }

    if language == "python":
        context.update(_python_context(source_code))
        context["parser_status"] = "PASSED"
    else:
        context["parser_status"] = "UNAVAILABLE"
        context["parser_note"] = (
            "Structural parsing is implemented for Python only. "
            "Empty structure lists do not mean the program has no functions."
        )

    result = {
        "validation_status": "validated",
        "detected_language": language,
        "imports": context.get("imports", []),
        "functions": context.get("functions", []),
        "classes": context.get("classes", []),
        "relevant_code_locations": context.get("relevant_code_locations", []),
    }

    payload = {
        "status": state.get("status", "ready"),
        "context": result,
        "errors": list(state.get("errors", [])),
        "findings": list(state.get("findings", [])),
    }
    payload.update(_merge_specialized_results(state, "input_context_agent", result))
    return {
        **payload,
        **record_agent(state, "input_context_agent", "completed", result),
    }


def _status_with_findings(state: CodeSentinelState, added_findings: list[dict]) -> str:
    """Keep an incomplete run incomplete, and do not call a missing tool a pass."""
    current = state.get("status", "analysis_passed")
    if current in {"partial_analysis", "analysis_error", "invalid_input"}:
        return current
    if added_findings or state.get("findings"):
        return "issues_found"
    if current in {"syntax_passed", "ready"}:
        return "analysis_passed"
    return current


def static_analysis_agent(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"
    source_code = state.get("source_code", "")

    if language != "python":
        tool_payload = {
            "tool_name": "language_linter",
            "execution_status": "UNAVAILABLE",
            "findings": [],
            "diagnostic_evidence": [],
            "tool_errors": [
                (
                    f"No static-analysis linter is configured for {language_label(language)}. "
                    "Syntax validation already ran separately and is not a linter result."
                )
            ],
        }
        return {
            "findings": list(state.get("findings", [])),
            "status": state.get("status", "analysis_passed"),
            **_merge_specialized_results(state, "static_analysis_agent", tool_payload),
            **record_agent(state, "static_analysis_agent", "unavailable", tool_payload),
        }

    result = run_ruff(source_code)
    if not result.get("ok"):
        error = result.get("error") or "Ruff was unavailable."
        errors = [*list(state.get("errors", [])), error]
        tool_payload = {
            "tool_name": "ruff",
            "execution_status": "UNAVAILABLE",
            "findings": [],
            "diagnostic_evidence": [],
            "tool_errors": [error],
        }
        return {
            "errors": errors,
            "findings": list(state.get("findings", [])),
            "status": "partial_analysis",
            **_merge_specialized_results(state, "static_analysis_agent", tool_payload),
            **record_agent(state, "static_analysis_agent", "unavailable", tool_payload),
        }

    findings: list[dict] = []
    for item in result.get("findings", []):
        copied = dict(item)
        copied.setdefault("source", "ruff")
        copied.setdefault("tool_name", "ruff")
        copied["execution_status"] = "FAILED"
        findings.append(copied)

    tool_status = "FAILED" if findings else "PASSED"
    tool_payload = {
        "tool_name": "ruff",
        "execution_status": tool_status,
        "findings": findings,
        "diagnostic_evidence": findings,
        "tool_errors": [],
    }
    payload = {
        **_merge_agent_findings(
            state,
            "static_analysis_agent",
            findings,
            extra={"tool_status": tool_status},
        ),
        "status": _status_with_findings(state, findings),
    }
    payload.update(_merge_specialized_results(state, "static_analysis_agent", tool_payload))
    return payload


def logic_bug_agent(state: CodeSentinelState) -> dict:
    language = normalize_language(state.get("language", "python")) or "python"
    source_code = state.get("source_code", "")
    findings: list[dict] = []
    result: dict[str, Any] = {
        "suspected_bug": None,
        "exact_location": None,
        "failure_scenario": "No logical defect was detected by the AST inspection pass.",
        "expected_vs_actual": None,
        "severity": "info",
        "confidence": "low",
        "suggested_correction": None,
        "execution_status": "PASSED",
    }

    if language != "python":
        unavailable = {
            "execution_status": "UNAVAILABLE",
            "reason": "Logic inspection is implemented for Python AST only.",
        }
        return {
            "findings": list(state.get("findings", [])),
            "status": state.get("status", "analysis_passed"),
            **_merge_specialized_results(state, "logic_bug_agent", unavailable),
            **record_agent(state, "logic_bug_agent", "unavailable", unavailable),
        }

    try:
        tree = ast.parse(source_code)
    except (SyntaxError, TypeError, ValueError):
        skipped = {
            "execution_status": "FAILED",
            "reason": "Syntax errors prevented AST inspection.",
        }
        payload = {
            **record_agent(state, "logic_bug_agent", "failed", skipped),
            "findings": list(state.get("findings", [])),
            "status": state.get("status", "analysis_passed"),
        }
        payload.update(_merge_specialized_results(state, "logic_bug_agent", skipped))
        return payload

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            result = {
                "suspected_bug": "Division by zero risk",
                "exact_location": {"line": node.lineno, "column": node.col_offset},
                "failure_scenario": "If the divisor is zero or becomes zero at runtime, the function raises ZeroDivisionError.",
                "expected_vs_actual": "Expected arithmetic to proceed safely; actual code can terminate with a runtime exception for zero divisors.",
                "severity": "warning",
                "confidence": "medium",
                "suggested_correction": "Guard the divisor before performing the division or validate input values before arithmetic.",
            }
            findings.append({
                "source": "logic_agent",
                "tool_name": "ast_inspection",
                "execution_status": "FAILED",
                "rule_id": "LOGIC-DIV0",
                "severity": "warning",
                "line": node.lineno,
                "message": "Division expression found; the divisor may be zero depending on input values.",
                "failure_scenario": result["failure_scenario"],
                "expected_vs_actual": result["expected_vs_actual"],
                "suggested_correction": result["suggested_correction"],
            })
            break

    if findings:
        result["execution_status"] = "FAILED"
    payload = {
        **_merge_agent_findings(state, "logic_bug_agent", findings),
        "status": _status_with_findings(state, findings),
    }
    payload.update(_merge_specialized_results(state, "logic_bug_agent", result))
    return payload


def security_analysis_agent(state: CodeSentinelState) -> dict:
    source_code = state.get("source_code", "")
    findings: list[dict] = []
    lower = source_code.lower()
    issues: list[dict[str, Any]] = []

    if ("cursor.execute" in lower or "execute(" in lower) and ("select" in lower or "insert" in lower or "update" in lower):
        if "f\"" in source_code or "f'" in source_code or "%s" in source_code:
            issue = {
                "vulnerability_category": "SQL injection",
                "evidence": "Dynamic SQL string construction includes user-controlled data.",
                "attack_or_failure_scenario": "A crafted input can alter the SQL statement and execute unintended database operations.",
                "severity": "high",
                "confidence": "medium",
                "recommended_mitigation": "Use parameterized queries and avoid interpolating raw input into SQL statements.",
            }
            issues.append(issue)
            findings.append({
                "source": "security_agent",
                "tool_name": "heuristic_security_scan",
                "execution_status": "PASSED",
                "rule_id": "SEC-SQLI",
                "severity": "high",
                "line": 1,
                "message": issue["evidence"],
                "attack_scenario": issue["attack_or_failure_scenario"],
                "recommended_mitigation": issue["recommended_mitigation"],
            })

    if "shell=true" in lower or "os.system(" in lower or ("subprocess.run(" in lower and "shell=true" in lower):
        issue = {
            "vulnerability_category": "Command injection",
            "evidence": "A shell command is constructed dynamically or executed through a shell.",
            "attack_or_failure_scenario": "Untrusted input can inject shell metacharacters and execute arbitrary commands.",
            "severity": "high",
            "confidence": "medium",
            "recommended_mitigation": "Avoid shell execution; use argument arrays and strict validation.",
        }
        issues.append(issue)
        findings.append({
            "source": "security_agent",
            "tool_name": "heuristic_security_scan",
            "execution_status": "PASSED",
            "rule_id": "SEC-CMDI",
            "severity": "high",
            "line": 1,
            "message": issue["evidence"],
            "attack_scenario": issue["attack_or_failure_scenario"],
            "recommended_mitigation": issue["recommended_mitigation"],
        })

    if re.search(r"(?:password|api[_-]?key|secret|token)\s*[:=]\s*[\"']", source_code, re.I):
        issue = {
            "vulnerability_category": "Exposed secret",
            "evidence": "A secret-like value appears to be hard-coded in source.",
            "attack_or_failure_scenario": "Anyone with repository access can recover and abuse the embedded secret.",
            "severity": "medium",
            "confidence": "medium",
            "recommended_mitigation": "Move the credential to an environment variable or secure secret store.",
        }
        issues.append(issue)
        findings.append({
            "source": "security_agent",
            "tool_name": "heuristic_security_scan",
            "execution_status": "PASSED",
            "rule_id": "SEC-SECRETS",
            "severity": "medium",
            "line": 1,
            "message": issue["evidence"],
            "attack_scenario": issue["attack_or_failure_scenario"],
            "recommended_mitigation": issue["recommended_mitigation"],
        })

    if "pickle.loads" in lower or "yaml.load" in lower or "marshal.loads" in lower:
        issue = {
            "vulnerability_category": "Unsafe deserialization",
            "evidence": "Untrusted data is deserialized through a permissive parser.",
            "attack_or_failure_scenario": "An attacker can send a crafted payload that executes code during deserialization.",
            "severity": "high",
            "confidence": "high",
            "recommended_mitigation": "Use safe, structured parsers and never deserialize untrusted data.",
        }
        issues.append(issue)
        findings.append({
            "source": "security_agent",
            "tool_name": "heuristic_security_scan",
            "execution_status": "PASSED",
            "rule_id": "SEC-DESER",
            "severity": "high",
            "line": 1,
            "message": issue["evidence"],
            "attack_scenario": issue["attack_or_failure_scenario"],
            "recommended_mitigation": issue["recommended_mitigation"],
        })

    payload = {
        **_merge_agent_findings(state, "security_analysis_agent", findings),
        "status": _status_with_findings(state, findings),
    }
    payload.update(_merge_specialized_results(state, "security_analysis_agent", issues))
    return payload


def edge_case_agent(state: CodeSentinelState) -> dict:
    findings = list(state.get("findings", []))
    suggestions: list[str] = []
    recommendations = {
        "focus_areas": [],
        "regression_test_suggestions": [],
    }

    for finding in findings:
        rule_id = str(finding.get("rule_id", "")).upper()
        if "DIV0" in rule_id or "LOGIC" in rule_id:
            suggestions.extend([
                "Empty input or zero divisor edge case",
                "Negative boundary values",
                "Large numeric value stress test",
            ])
        if "SQL" in rule_id or "CMD" in rule_id or "SECRETS" in rule_id:
            suggestions.extend([
                "Empty input and malformed payloads",
                "Duplicate or malformed data inputs",
                "Unexpected null or missing values",
            ])

    recommendations = {
        "focus_areas": sorted(set(suggestions)),
        "regression_test_suggestions": [
            {
                "scenario": item,
                "expected_behavior": (
                    "The code should reject or handle the case safely without crashing, "
                    "exposing secrets, or allowing unsafe execution."
                ),
            }
            for item in sorted(set(suggestions))
        ],
        "evidence_note": (
            "Suggestions are derived only from findings already produced."
            if suggestions
            else "No code-specific edge cases were derived from the findings."
        ),
    }

    payload = {
        **record_agent(state, "edge_case_agent", "completed", recommendations),
        "findings": findings,
        "status": state.get("status", "analysis_passed"),
        "edge_case_recommendations": recommendations,
    }
    payload.update(_merge_specialized_results(state, "edge_case_agent", recommendations))
    return payload


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

    ok, detail, tool_status = _run_standalone_syntax_check(language, source_code)
    if tool_status == "UNAVAILABLE":
        error = detail or f"{language_label(language)} syntax tool is unavailable."
        result = {
            "status": "partial_analysis",
            "findings": [],
            "errors": [error],
            "tool_status": "UNAVAILABLE",
        }
        return {
            **result,
            **record_agent(state, "syntax_checker", "unavailable", result),
        }

    if not ok:
        source_name = language_label(language)
        rule_id = "PY-SYNTAX" if language == "python" else f"{language.upper().replace('+', 'PLUS')}-SYNTAX"
        result = {
            "status": "issues_found",
            "findings": [assign_finding_id({
                "source": f"{language}_syntax",
                "rule_id": rule_id,
                "severity": "error",
                "line": 1,
                "column": 1,
                "message": detail or f"{source_name} syntax check failed.",
                "tool_status": tool_status or "FAILED",
                "execution_status": "FAILED",
            })],
            "errors": [],
        }
        return {
            **result,
            **record_agent(state, "syntax_checker", "completed", result),
        }

    result = {
        "status": "syntax_passed",
        "findings": [],
        "errors": [],
        "tool_status": "PASSED",
    }
    return {
        **result,
        **record_agent(state, "syntax_checker", "completed", result),
    }


def route_after_syntax(
    state: CodeSentinelState,
) -> Literal["input_context_agent", "finalize"]:
    return (
        "input_context_agent"
        if state.get("status") == "syntax_passed"
        else "finalize"
    )


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
    try:
        result = call_review_function(
            state["source_code"],
            verification_feedback=None,
            language=language,
        )
    except Exception as exc:
        result = {
            "ok": False,
            "findings": [],
            "summary": "Gemini review could not be completed.",
            "error": f"{type(exc).__name__}: {exc}",
        }

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

    combined = deduplicate_findings(existing_findings, ai_findings)

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
        and item.get("verification") in {"NOT_CHECKED", "UNRESOLVED"}
    ]

    human_review_findings = [
        item
        for item in verified
        if item.get("verification") == "UNRESOLVED"
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
) -> Literal["ai_reassessment", "adversarial_review"]:
    retry_count = state.get("ai_retry_count", 0)
    feedback = state.get("verification_feedback", [])

    if feedback and retry_count < MAX_AI_RETRIES:
        return "ai_reassessment"

    return "adversarial_review"


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
    combined = deduplicate_findings(ruff_findings, revised_findings)

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


def _loop_depth_issues(tree: ast.AST) -> list[int]:
    lines: list[int] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_For(self, node: ast.For) -> None:
            self._visit_loop(node)

        def visit_While(self, node: ast.While) -> None:
            self._visit_loop(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self._visit_loop(node)

        def _visit_loop(self, node: ast.AST) -> None:
            if self.depth >= 1:
                lines.append(getattr(node, "lineno", 1))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

    Visitor().visit(tree)
    return lines


def performance_analysis_agent(state: CodeSentinelState) -> dict:
    """Inspect Python structure for costly patterns. Does not execute code."""
    language = normalize_language(state.get("language", "python")) or "python"
    if language != "python":
        unavailable = {
            "execution_status": "UNAVAILABLE",
            "reason": "Performance inspection is implemented for Python AST only.",
        }
        return {
            "findings": list(state.get("findings", [])),
            "status": state.get("status", "analysis_passed"),
            **_merge_specialized_results(state, "performance_analysis_agent", unavailable),
            **record_agent(state, "performance_analysis_agent", "unavailable", unavailable),
        }

    try:
        tree = ast.parse(state.get("source_code", ""))
    except (SyntaxError, TypeError, ValueError):
        failed = {
            "execution_status": "FAILED",
            "reason": "Syntax errors prevented performance inspection.",
        }
        return {
            "findings": list(state.get("findings", [])),
            "status": state.get("status", "analysis_passed"),
            **_merge_specialized_results(state, "performance_analysis_agent", failed),
            **record_agent(state, "performance_analysis_agent", "failed", failed),
        }

    findings: list[dict] = []
    nested_lines = _loop_depth_issues(tree)
    if nested_lines:
        findings.append({
            "source": "performance_agent",
            "tool_name": "python_ast",
            "execution_status": "FAILED",
            "rule_id": "PERF-NESTED-LOOP",
            "severity": "info",
            "line": nested_lines[0],
            "message": (
                "Nested loops were found. This can become expensive for large inputs, "
                "but the cost is not proven by this static check."
            ),
        })

    tool_payload = {
        "execution_status": "FAILED" if findings else "PASSED",
        "finding_count": len(findings),
        "nested_loop_lines": nested_lines,
    }
    payload = {
        **_merge_agent_findings(state, "performance_analysis_agent", findings),
        "status": _status_with_findings(state, findings),
    }
    payload.update(_merge_specialized_results(state, "performance_analysis_agent", tool_payload))
    return payload


def adversarial_review_agent(state: CodeSentinelState) -> dict:
    """Challenge existing findings. This agent does not invent new defects."""
    findings = list(state.get("findings", []))
    challenges: list[dict[str, Any]] = []

    for finding in findings:
        verification = str(finding.get("verification") or "NOT_CHECKED")
        if verification not in {"CONTRADICTED", "UNRESOLVED", "NOT_CHECKED"}:
            continue
        if finding.get("source") not in {"gemini_ai", "logic_agent", "security_agent", "performance_agent"}:
            continue
        challenges.append({
            "finding_id": finding.get("finding_id"),
            "rule_id": finding.get("rule_id"),
            "outcome": verification,
            "reason": finding.get("verification_evidence") or (
                "The claim is not independently established and remains a potential issue."
            ),
        })

    result = {
        "execution_status": "PASSED",
        "challenge_count": len(challenges),
        "challenges": challenges,
        "note": "No additional defects were created by the adversarial pass.",
    }
    return {
        "findings": findings,
        "status": state.get("status", "analysis_passed"),
        **_merge_specialized_results(state, "adversarial_review_agent", result),
        **record_agent(state, "adversarial_review_agent", "completed", result),
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
        "awaiting_human_review" if review_required else "skipped",
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
    if decision not in REVIEW_DECISIONS:
        allowed = ", ".join(sorted(REVIEW_DECISIONS))
        raise ValueError(f"Review decision must be one of: {allowed}.")

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
            "completed",
            {
                "decision": decision,
                "feedback": feedback,
                "note": (
                    "The decision records the reviewer's judgment. "
                    "It does not prove the code is correct or secure."
                ),
            },
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
            f"{INCOMPLETE_SUMMARY}. No findings were produced by the available "
            "analysis steps. Check the reported errors."
        )

    elif status == "partial_analysis":
        summary = (
            f"{INCOMPLETE_SUMMARY}. {len(findings)} finding(s) "
            "were collected from the checks that completed."
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
            linter_note = ""
            static_result = state.get("specialized_agent_results", {}).get(
                "static_analysis_agent", {}
            )
            if isinstance(static_result, dict) and static_result.get("execution_status") == "UNAVAILABLE":
                linter_note = (
                    " No dedicated static-analysis linter ran for this language."
                )
            summary = (
                f"{language_name} syntax validation passed."
                f"{linter_note}"
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
    builder.add_node("input_context_agent", input_context_agent)
    builder.add_node("static_analysis_agent", static_analysis_agent)
    builder.add_node("logic_bug_agent", logic_bug_agent)
    builder.add_node("security_analysis_agent", security_analysis_agent)
    builder.add_node("performance_analysis_agent", performance_analysis_agent)
    builder.add_node("edge_case_agent", edge_case_agent)
    builder.add_node("ai_review", ai_review)
    builder.add_node("verify_findings", verify_analysis_findings)
    builder.add_node("ai_reassessment", ai_reassessment)
    builder.add_node("adversarial_review", adversarial_review_agent)
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
            "input_context_agent": "input_context_agent",
            "finalize": "finalize",
        },
    )

    builder.add_edge("input_context_agent", "static_analysis_agent")
    builder.add_edge("static_analysis_agent", "logic_bug_agent")
    builder.add_edge("logic_bug_agent", "security_analysis_agent")
    builder.add_edge("security_analysis_agent", "performance_analysis_agent")
    builder.add_edge("performance_analysis_agent", "edge_case_agent")
    builder.add_edge("edge_case_agent", "ai_review")
    builder.add_edge("ai_review", "verify_findings")

    builder.add_conditional_edges(
        "verify_findings",
        route_after_verification,
        {
            "ai_reassessment": "ai_reassessment",
            "adversarial_review": "adversarial_review",
        },
    )

    builder.add_edge("ai_reassessment", "verify_findings")
    builder.add_edge("adversarial_review", "prepare_review")

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
