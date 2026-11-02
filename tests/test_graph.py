
import pytest

import graph
from graph import code_sentinel_graph


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------

def make_ai_finding(
    rule_id="AI-LOGIC-001",
    message="Potential logic issue.",
    line=2,
    verification="potential_issue_not_independently_verified",
):
    return {
        "source": "gemini_ai",
        "rule_id": rule_id,
        "severity": "warning",
        "line": line,
        "message": message,
        "reasoning": "The code may not behave as intended.",
        "verification": verification,
    }


def make_ai_success(findings=None, summary="AI review completed."):
    return {
        "ok": True,
        "findings": findings or [],
        "summary": summary,
        "error": None,
    }


def make_ai_failure(error="Gemini request failed."):
    return {
        "ok": False,
        "findings": [],
        "summary": "",
        "error": error,
    }


def invoke_graph(source_code, **extra_state):
    state = {
        "source_code": source_code,
        "language": "python",
        **extra_state,
    }
    return code_sentinel_graph.invoke(state)


# ---------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------

def test_valid_python_passes_analysis(monkeypatch):
    def mock_ai(_source_code, verification_feedback=None):
        return make_ai_success()

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def greet():\n"
        "    return 'hello'\n"
    )

    assert result["status"] == "analysis_passed"
    assert result["findings"] == []
    assert result["line_count"] == 2
    assert "does not establish logical correctness" in (
        result["summary"].lower()
    )


def test_invalid_python_reports_syntax_error(monkeypatch):
    def mock_ai(_source_code, verification_feedback=None):
        pytest.fail("AI review must not run for invalid syntax.")

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def greet(\n"
        "    return 'hello'\n"
    )

    assert result["status"] == "issues_found"
    assert any(
        finding["rule_id"] == "PY-SYNTAX"
        for finding in result["findings"]
    )
    assert "Further analysis was not run" in result["summary"]


def test_empty_source_is_rejected():
    result = invoke_graph("")

    assert result["status"] == "invalid_input"
    assert result["errors"]


def test_javascript_language_is_accepted(monkeypatch):
    def mock_ai(_source_code, verification_feedback=None):
        return make_ai_success()

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "console.log('hello');",
        language="javascript",
    )

    assert result["status"] == "analysis_passed"
    assert result["language"] == "javascript"


def test_oversized_source_is_rejected():
    result = invoke_graph(
        "x = 1\n" * 25001
    )

    assert result["status"] == "invalid_input"
    assert result["errors"]


# ---------------------------------------------------------
# 2. Static analysis
# ---------------------------------------------------------

def test_ruff_reports_unused_local_variable():
    result = graph.run_ruff(
        "def calculate():\n"
        "    unused_value = 10\n"
        "    return 5\n"
    )

    assert result["ok"] is True
    assert any(
        finding["rule_id"] == "F841"
        for finding in result["findings"]
    )


def test_ruff_finding_contains_location_and_message():
    result = graph.run_ruff(
        "def calculate():\n"
        "    unused_value = 10\n"
        "    return 5\n"
    )

    assert result["ok"] is True
    assert result["findings"]

    finding = next(
        item
        for item in result["findings"]
        if item["rule_id"] == "F841"
    )

    assert finding["line"] is not None
    assert finding["message"]


# ---------------------------------------------------------
# 3. AI and static-analysis integration
# ---------------------------------------------------------

def test_ai_findings_are_combined_with_ruff_findings(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success(
            findings=[
                make_ai_finding()
            ],
            summary="One potential logic issue identified.",
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def greet(name):\n"
        "    unused = 1\n"
        "    return 'hello'\n"
    )

    sources = {
        finding["source"]
        for finding in result["findings"]
    }

    assert "ruff" in sources
    assert "gemini_ai" in sources
    assert result["status"] in (
        "issues_found",
        "partial_analysis",
    )


def test_gemini_failure_preserves_ruff_findings(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_failure()

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def calculate():\n"
        "    unused_value = 10\n"
        "    return 5\n"
    )

    assert any(
        finding["source"] == "ruff"
        for finding in result["findings"]
    )

    assert result["status"] == "partial_analysis"


# ---------------------------------------------------------
# 4. Finding verification
# ---------------------------------------------------------

def test_graph_verifies_gemini_division_finding(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success(
            findings=[
                make_ai_finding(
                    rule_id="AI-ZERO-DIVISION",
                    message="Potential ZeroDivisionError",
                    line=2,
                )
            ]
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def divide(a, b):\n"
        "    return a / b\n"
    )

    finding = next(
        item
        for item in result["findings"]
        if item["source"] == "gemini_ai"
    )

    assert finding["verification"] == "UNRESOLVED"


def test_graph_keeps_unrelated_ai_finding_unverified(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success(
            findings=[
                make_ai_finding(
                    rule_id="AI-STYLE-001",
                    message="Potential business logic issue.",
                )
            ]
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def calculate(value):\n"
        "    return value + 10\n"
    )

    finding = next(
        item
        for item in result["findings"]
        if item["source"] == "gemini_ai"
    )

    assert finding["verification"] == "NOT_CHECKED"


# ---------------------------------------------------------
# 5. AI reassessment / self-correction
# ---------------------------------------------------------

def test_ai_reassessment_receives_verification_feedback(monkeypatch):
    calls = []

    def mock_ai(
        source_code,
        verification_feedback=None,
    ):
        calls.append(verification_feedback)

        if verification_feedback is None:
            return make_ai_success(
                findings=[
                    make_ai_finding()
                ]
            )

        return make_ai_success(
            findings=[
                make_ai_finding(
                    rule_id="AI-LOGIC-001",
                    verification="not_verified",
                )
            ],
            summary="Reassessment completed.",
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def calculate(value):\n"
        "    return value + 10\n"
    )

    assert len(calls) == 2
    assert calls[0] is None
    assert isinstance(calls[1], list)
    assert result["findings"]


def test_ai_reassessment_failure_preserves_previous_findings(
    monkeypatch,
):
    calls = []

    def mock_ai(
        source_code,
        verification_feedback=None,
    ):
        calls.append(verification_feedback)

        if verification_feedback is None:
            return make_ai_success(
                findings=[
                    make_ai_finding()
                ]
            )

        return make_ai_failure(
            "Reassessment request failed."
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def calculate(value):\n"
        "    return value + 10\n"
    )

    assert len(calls) == 2
    assert any(
        finding["source"] == "gemini_ai"
        for finding in result["findings"]
    )


def test_ai_reassessment_runs_at_most_once(monkeypatch):
    calls = []

    def mock_ai(
        source_code,
        verification_feedback=None,
    ):
        calls.append(verification_feedback)

        return make_ai_success(
            findings=[
                make_ai_finding()
            ]
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    invoke_graph(
        "def calculate(value):\n"
        "    return value + 10\n"
    )

    assert len(calls) <= 2


# ---------------------------------------------------------
# 6. Human-in-the-loop
# ---------------------------------------------------------

def test_human_review_can_be_disabled(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success(
            findings=[
                make_ai_finding(
                    rule_id="AI-ZERO-DIVISION",
                    message="Potential ZeroDivisionError",
                )
            ]
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def divide(a, b):\n"
        "    return a / b\n",
        require_human_review=False,
    )

    assert result["review_completed"] is False


def test_no_findings_does_not_require_human_review(monkeypatch):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success()

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def greet():\n"
        "    return 'hello'\n",
        require_human_review=True,
    )

    assert result["findings"] == []
    assert result["review_required"] is False


def test_summary_concern_without_findings_is_not_reported_as_passed(
    monkeypatch,
):
    def mock_ai(
        _source_code,
        verification_feedback=None,
    ):
        return make_ai_success(
            findings=[],
            summary=(
                "The division function may raise ZeroDivisionError "
                "because it does not handle a zero divisor."
            ),
        )

    monkeypatch.setattr(graph, "review_python_code", mock_ai)

    result = invoke_graph(
        "def divide(a, b):\n"
        "    return a / b\n",
        require_human_review=False,
    )

    assert result["status"] != "analysis_passed"
    assert not any(
        finding.get("source") == "gemini_ai"
        for finding in result["findings"]
    )
    assert result["review_required"] is False
    assert "summary/findings inconsistency" in result["summary"].lower()
