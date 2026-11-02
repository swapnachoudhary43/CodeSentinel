from verification import verify_finding, verify_findings


def test_division_risk_requires_human_review():
    source = "def divide(a, b):\n    return a / b\n"

    finding = {
        "source": "gemini_ai",
        "rule_id": "DIV0-001",
        "severity": "warning",
        "line": 2,
        "message": "Potential ZeroDivisionError if b is zero.",
        "reasoning": "The divisor is not checked.",
    }

    result = verify_finding(source, finding)

    assert result["verification"] == "UNRESOLVED"
    assert "division expression" in result["verification_evidence"]
    assert result["rule_id"] == "DIV0-001"


def test_unrelated_ai_finding_remains_unverified():
    source = "def add(a, b):\n    return a + b\n"

    finding = {
        "source": "gemini_ai",
        "rule_id": "AI-LOGIC-001",
        "severity": "warning",
        "line": 2,
        "message": "Potential logic issue.",
        "reasoning": "The result may not match the intended behavior.",
    }

    result = verify_finding(source, finding)

    assert result["verification"] == "NOT_CHECKED"


def test_ruff_finding_is_marked_as_tool_reported():
    finding = {
        "source": "ruff",
        "rule_id": "F841",
        "severity": "warning",
        "line": 2,
        "message": "Local variable is assigned to but never used.",
    }

    result = verify_finding("unused = 1\n", finding)

    assert result["verification"] == "NOT_CHECKED"


def test_multiple_findings_are_processed():
    findings = [
        {
            "source": "gemini_ai",
            "rule_id": "AI-001",
            "severity": "warning",
            "message": "Potential issue.",
            "reasoning": "The code may be incorrect.",
        },
        {
            "source": "ruff",
            "rule_id": "F841",
            "severity": "warning",
            "message": "Unused variable.",
        },
    ]

    results = verify_findings("x = 1\n", findings)

    assert len(results) == 2
    assert results[0]["verification"] == "NOT_CHECKED"
    assert results[1]["verification"] == "NOT_CHECKED"