import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph import build_graph

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
        description="Human review decision: approve or reject.",
    )
    feedback: str = Field(
        default="",
        max_length=2000,
        description="Optional reviewer feedback.",
    )


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
        "agent_status": result.get("agent_status", {}),
        "agent_results": result.get("agent_results", {}),
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
        if snapshot.next:
            response["status"] = "pending_review"
            response["review_required"] = True

        response["thread_id"] = thread_id
        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis workflow failed: {exc}",
        ) from exc


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
            detail=f"Could not retrieve review status: {exc}",
        ) from exc


@app.post("/api/review/{thread_id}")
def submit_review(
    thread_id: str,
    request: ReviewRequest,
) -> dict[str, Any]:
    """Submit a human decision and resume the paused workflow."""
    decision = request.decision.strip().lower()

    if decision not in {"approve", "reject"}:
        raise HTTPException(
            status_code=422,
            detail="Decision must be either 'approve' or 'reject'.",
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
            detail=f"Could not resume review workflow: {exc}",
        ) from exc
