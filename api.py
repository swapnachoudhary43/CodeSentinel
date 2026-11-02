import json
import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import get_gemini_api_key
from graph import REVIEW_DECISIONS, build_graph

app = FastAPI(
    title="CodeSentinel API",
    description="Multi-agent code intelligence and verification platform.",
    version="0.2.0",
)

frontend_url = os.getenv("FRONTEND_URL", "").strip()

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# One checkpointer instance is shared by all API requests.
checkpointer = InMemorySaver()
api_graph = build_graph(checkpointer=checkpointer)


@app.get("/")
def root() -> dict[str, str]:
    """Show a welcome message and links to API resources."""
    return {
        "name": "CodeSentinel",
        "description": "Multi-Agent Code Intelligence & Verification Platform",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
    }


class AnalysisRequest(BaseModel):
    source_code: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Python source code to analyze.",
    )
    language: str = Field(
        default="python",
        description="Programming language of the submitted source. Supported values: python, javascript, java, c, c++.",
    )
    require_human_review: bool = Field(
        default=True,
        description="Pause for human review when findings are produced.",
    )


class ReviewRequest(BaseModel):
    decision: str = Field(
        ...,
        description=(
            "Human review decision: approve, approve_with_warnings, "
            "request_changes, reject, or defer."
        ),
    )
    feedback: str = Field(
        default="",
        max_length=2000,
        description="Optional reviewer feedback.",
    )


def public_error(exc: Exception) -> str:
    """Return a client-safe error without embedded credentials."""
    message = f"{type(exc).__name__}: {exc}"
    secret = get_gemini_api_key()
    if secret and secret in message:
        message = message.replace(secret, "[redacted]")
    return message[:500]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def get_thread_config(thread_id: str) -> dict[str, Any]:
    """Build the LangGraph checkpoint configuration."""
    return {"configurable": {"thread_id": thread_id}}


def serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a predictable API response from the graph state."""
    return {
        "status": result.get("status", "unknown"),
        "summary": result.get("summary", ""),
        "findings": result.get("findings", []),
        "errors": result.get("errors", []),
        "ai_summary": result.get("ai_summary", ""),
        "line_count": result.get("line_count", 0),
        "context": result.get("context", {}),
        "agent_status": result.get("agent_status", {}),
        "agent_results": result.get("agent_results", {}),
        "specialized_agent_results": result.get("specialized_agent_results", {}),
        "edge_case_recommendations": result.get("edge_case_recommendations", {}),
        "audit_log": result.get("audit_log", []),
        "review_required": result.get("review_required", False),
        "review_decision": result.get("review_decision", ""),
        "review_feedback": result.get("review_feedback", ""),
        "review_completed": result.get("review_completed", False),
        "thread_id": result.get("thread_id"),
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "healthy",
        "service": "codesentinel-api",
    }


@app.post("/api/analyze")
def analyze_code(request: AnalysisRequest) -> dict[str, Any]:
    """Start a new analysis workflow."""
    thread_id = str(uuid4())
    config = get_thread_config(thread_id)

    initial_state = {
        "source_code": request.source_code,
        "language": request.language,
        "require_human_review": request.require_human_review,
    }

    try:
        result = api_graph.invoke(initial_state, config=config)
        response = serialize_result(result)

        snapshot = api_graph.get_state(config)
        response["pending"] = bool(snapshot.next)
        if snapshot.next:
            response["status"] = "pending_review"
            response["review_required"] = True

        response["thread_id"] = thread_id
        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis workflow failed: {public_error(exc)}",
        ) from exc


def _result_from_snapshot(thread_id: str, config: dict[str, Any]) -> dict[str, Any]:
    snapshot = api_graph.get_state(config)
    response = serialize_result(snapshot.values or {})
    response["pending"] = bool(snapshot.next)
    if snapshot.next:
        response["status"] = "pending_review"
        response["review_required"] = True
    response["thread_id"] = thread_id
    return response


def _analysis_events(request: AnalysisRequest, thread_id: str) -> Iterator[str]:
    config = get_thread_config(thread_id)
    initial_state = {
        "source_code": request.source_code,
        "language": request.language,
        "require_human_review": request.require_human_review,
    }

    try:
        for update in api_graph.stream(initial_state, config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                if str(node_name).startswith("__"):
                    continue
                latest: dict[str, Any] = {}
                if isinstance(node_update, dict):
                    audit_log = node_update.get("audit_log") or []
                    if audit_log and isinstance(audit_log[-1], dict):
                        latest = audit_log[-1]
                yield _sse({
                    "type": "agent_update",
                    "node": node_name,
                    "agent": latest.get("agent") or node_name,
                    "status": latest.get("status") or "completed",
                })

        yield _sse({
            "type": "complete",
            "result": _result_from_snapshot(thread_id, config),
        })
    except Exception as exc:
        yield _sse({
            "type": "error",
            "detail": f"Analysis workflow failed: {public_error(exc)}",
        })


@app.post("/api/analyze/stream")
def analyze_code_stream(request: AnalysisRequest) -> StreamingResponse:
    """Stream real workflow events, then the final analysis state."""
    thread_id = str(uuid4())
    return StreamingResponse(
        _analysis_events(request, thread_id),
        media_type="text/event-stream",
    )


@app.get("/api/review/{thread_id}")
def get_review_status(thread_id: str) -> dict[str, Any]:
    """Retrieve the latest state of an analysis workflow."""
    config = get_thread_config(thread_id)

    try:
        snapshot = api_graph.get_state(config)

        if not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail="Analysis thread not found.",
            )

        response = serialize_result(snapshot.values)
        response["thread_id"] = thread_id
        response["pending"] = bool(snapshot.next)
        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve review status: {public_error(exc)}",
        ) from exc


@app.post("/api/review/{thread_id}")
def submit_review(
    thread_id: str,
    request: ReviewRequest,
) -> dict[str, Any]:
    """Submit a human decision and resume the paused workflow."""
    decision = request.decision.strip().lower()

    if decision not in REVIEW_DECISIONS:
        allowed = ", ".join(sorted(REVIEW_DECISIONS))
        raise HTTPException(
            status_code=422,
            detail=f"Decision must be one of: {allowed}.",
        )

    config = get_thread_config(thread_id)

    try:
        snapshot = api_graph.get_state(config)

        if not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail="Analysis thread not found.",
            )

        if not snapshot.next:
            raise HTTPException(
                status_code=409,
                detail="This workflow is not waiting for human review.",
            )

        result = api_graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "feedback": request.feedback,
                }
            ),
            config=config,
        )

        response = serialize_result(result)
        response["thread_id"] = thread_id
        response["pending"] = False
        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resume review workflow: {public_error(exc)}",
        ) from exc
