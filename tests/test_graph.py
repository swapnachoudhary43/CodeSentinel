from graph import code_sentinel_graph


def test_valid_python_passes_analysis():
    result = code_sentinel_graph.invoke(
        {
            "source_code": "def add(a, b):\n    return a + b\n",
            "language": "python",
        }
    )

    assert result["status"] == "analysis_passed"
    assert result["findings"] == []
    assert result["line_count"] == 2
    assert "does not establish logical correctness" in result["summary"]


def test_invalid_python_reports_syntax_error():
    result = code_sentinel_graph.invoke(
        {
            "source_code": "def add(a, b)\n    return a + b\n",
            "language": "python",
        }
    )

    assert result["status"] == "issues_found"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule_id"] == "PY-SYNTAX"
    assert result["findings"][0]["line"] == 1
    assert "Ruff was not run" in result["summary"]


def test_empty_source_is_rejected():
    result = code_sentinel_graph.invoke(
        {"source_code": "   ", "language": "python"}
    )

    assert result["status"] == "invalid_input"
    assert result["findings"] == []
    assert "non-empty string" in result["errors"][0]


def test_unsupported_language_is_rejected():
    result = code_sentinel_graph.invoke(
        {"source_code": "const answer = 42;", "language": "javascript"}
    )

    assert result["status"] == "invalid_input"
    assert "Python source only" in result["errors"][0]


def test_oversized_source_is_rejected():
    result = code_sentinel_graph.invoke(
        {"source_code": "x" * 50_001, "language": "python"}
    )

    assert result["status"] == "invalid_input"
    assert "character limit" in result["errors"][0]


def test_ruff_reports_unused_local_variable():
    result = code_sentinel_graph.invoke(
        {
            "source_code": (
                "def greet(name):\n"
                "    unused = 1\n"
                "    return 'hello'\n"
            ),
            "language": "python",
        }
    )

    assert result["status"] == "issues_found"
    assert any(
        finding["rule_id"] == "F841"
        and finding["line"] == 2
        and finding["source"] == "ruff"
        for finding in result["findings"]
    )


def test_ruff_finding_contains_location_and_message():
    result = code_sentinel_graph.invoke(
        {
            "source_code": (
                "def greet(name):\n"
                "    unused = 1\n"
                "    return 'hello'\n"
            ),
            "language": "python",
        }
    )

    finding = next(
        item for item in result["findings"] if item["source"] == "ruff"
    )

    assert finding["rule_id"]
    assert finding["line"] == 2
    assert finding["column"] is not None
    assert finding["message"]
