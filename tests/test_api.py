import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import api
import graph
from graph import build_graph


@pytest.fixture()
def client(monkeypatch):
    def mock_ai(_source_code, verification_feedback=None, language="python"):
        return {
            "ok": True,
            "findings": [],
            "summary": "No actionable issue was identified by this review.",
            "error": None,
        }

    monkeypatch.setattr(graph, "review_python_code", mock_ai)
    api.checkpointer = InMemorySaver()
    api.api_graph = build_graph(checkpointer=api.checkpointer)
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_analyze_valid_python(client):
    response = client.post(
        "/api/analyze",
        json={
            "source_code": "def greet():\n    return 'hello'\n",
            "language": "python",
            "require_human_review": False,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "analysis_passed"
    assert body["thread_id"]
    assert isinstance(body["findings"], list)
    assert isinstance(body["audit_log"], list)
    assert body["pending"] is False


def test_analyze_missing_source_is_rejected(client):
    response = client.post("/api/analyze", json={"language": "python"})

    assert response.status_code == 422


def test_analyze_empty_source_is_rejected(client):
    response = client.post(
        "/api/analyze",
        json={"source_code": "", "language": "python"},
    )

    assert response.status_code == 422


def test_analyze_whitespace_is_invalid_input(client):
    response = client.post(
        "/api/analyze",
        json={"source_code": "   ", "language": "python"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "invalid_input"


def test_analyze_wrong_type_is_rejected(client):
    response = client.post(
        "/api/analyze",
        json={"source_code": 12, "language": "python"},
    )

    assert response.status_code == 422


def test_analyze_malformed_json_is_rejected(client):
    response = client.post(
        "/api/analyze",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_unsupported_language_is_reported(client):
    response = client.post(
        "/api/analyze",
        json={"source_code": "print(1)", "language": "ruby"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "invalid_input"
    assert body["errors"]


def test_large_source_is_rejected(client):
    response = client.post(
        "/api/analyze",
        json={"source_code": "x = 1\n" * 20000, "language": "python"},
    )

    assert response.status_code == 422


def test_repeated_analyses_have_distinct_threads(client):
    payload = {
        "source_code": "def greet():\n    return 'hello'\n",
        "language": "python",
        "require_human_review": False,
    }
    first = client.post("/api/analyze", json=payload).json()
    second = client.post("/api/analyze", json=payload).json()

    assert first["thread_id"] != second["thread_id"]


def test_missing_thread_returns_404(client):
    response = client.get("/api/review/missing-thread")

    assert response.status_code == 404


def test_invalid_review_decision_returns_422(client):
    created = client.post(
        "/api/analyze",
        json={
            "source_code": "def divide(a, b):\n    return a / b\n",
            "language": "python",
            "require_human_review": True,
        },
    ).json()

    response = client.post(
        f"/api/review/{created['thread_id']}",
        json={"decision": "ship-it"},
    )

    assert response.status_code == 422
    assert "approve_with_warnings" in response.json()["detail"]


def test_human_decisions_are_stored(client):
    created = client.post(
        "/api/analyze",
        json={
            "source_code": "def divide(a, b):\n    return a / b\n",
            "language": "python",
            "require_human_review": True,
        },
    )
    body = created.json()
    assert created.status_code == 200
    assert body["pending"] is True
    assert any(item["rule_id"] == "LOGIC-DIV0" for item in body["findings"])

    submitted = client.post(
        f"/api/review/{body['thread_id']}",
        json={
            "decision": "approve_with_warnings",
            "feedback": "The divisor is still unchecked.",
        },
    )
    saved = submitted.json()
    assert submitted.status_code == 200
    assert saved["review_completed"] is True
    assert saved["review_decision"] == "approve_with_warnings"
    assert saved["pending"] is False
    assert "does not independently" in saved["summary"]

    fetched = client.get(f"/api/review/{body['thread_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["review_decision"] == "approve_with_warnings"
    assert fetched.json()["review_feedback"] == "The divisor is still unchecked."


def test_review_without_pause_returns_409(client):
    created = client.post(
        "/api/analyze",
        json={
            "source_code": "def greet():\n    return 'hello'\n",
            "language": "python",
            "require_human_review": False,
        },
    ).json()

    response = client.post(
        f"/api/review/{created['thread_id']}",
        json={"decision": "reject"},
    )

    assert response.status_code == 409


def test_missing_gemini_key_is_partial(monkeypatch):
    monkeypatch.setattr("ai_agent.get_gemini_api_key", lambda: "")
    monkeypatch.setattr("config.get_gemini_api_key", lambda: "")
    api.checkpointer = InMemorySaver()
    api.api_graph = build_graph(checkpointer=api.checkpointer)
    local_client = TestClient(api.app)

    response = local_client.post(
        "/api/analyze",
        json={
            "source_code": "def greet():\n    return 'hello'\n",
            "language": "python",
            "require_human_review": False,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "partial_analysis"
    assert "INSUFFICIENT EVIDENCE" in body["summary"]
    assert any("API key" in error for error in body["errors"])
    assert "sk-" not in json.dumps(body)


def test_ruff_unavailable_is_not_a_pass(client, monkeypatch):
    monkeypatch.setattr(
        graph,
        "run_ruff",
        lambda _source: {"ok": False, "findings": [], "error": "Ruff executable not found."},
    )

    response = client.post(
        "/api/analyze",
        json={
            "source_code": "def greet():\n    return 'hello'\n",
            "language": "python",
            "require_human_review": False,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "partial_analysis"
    assert body["agent_status"]["static_analysis_agent"] == "unavailable"
    specialized = body["specialized_agent_results"]["static_analysis_agent"]
    assert specialized["execution_status"] == "UNAVAILABLE"
    assert "INSUFFICIENT EVIDENCE" in body["summary"]


def test_stream_emits_real_agent_events(client):
    with client.stream(
        "POST",
        "/api/analyze/stream",
        json={
            "source_code": "def greet():\n    return 'hello'\n",
            "language": "python",
            "require_human_review": False,
        },
    ) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())

    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in payload.splitlines()
        if line.startswith("data:")
    ]
    agents = [event.get("agent") for event in events if event.get("type") == "agent_update"]
    complete = next(event for event in events if event.get("type") == "complete")

    assert "input_validation" in agents
    assert "syntax_checker" in agents
    assert "static_analysis_agent" in agents
    assert "report_agent" in agents
    assert complete["result"]["status"] == "analysis_passed"


def test_public_error_redacts_api_key(monkeypatch):
    monkeypatch.setattr(api, "get_gemini_api_key", lambda: "secret-key-value")

    message = api.public_error(RuntimeError("request failed for secret-key-value"))

    assert "secret-key-value" not in message
    assert "[redacted]" in message
