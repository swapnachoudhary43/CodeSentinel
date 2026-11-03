"""Run the CodeSentinel workflow in a terminal for the live evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import REVIEW_DECISIONS, build_graph, normalize_language

graph = build_graph(checkpointer=InMemorySaver())


def read_submission(file_path: str | None) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")

    print("Paste the instructor's source. Finish with a line containing only END.")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def print_update(update: dict) -> None:
    for node_name, node_update in update.items():
        if str(node_name).startswith("__"):
            continue
        agent = node_name
        status = "completed"
        if isinstance(node_update, dict):
            audit_log = node_update.get("audit_log") or []
            if audit_log and isinstance(audit_log[-1], dict):
                agent = audit_log[-1].get("agent") or node_name
                status = audit_log[-1].get("status") or status
        print(f"[{agent}] {status}")


def stream_until_pause(payload, config: dict) -> dict:
    for update in graph.stream(payload, config, stream_mode="updates"):
        if isinstance(update, dict):
            print_update(update)
    snapshot = graph.get_state(config)
    return snapshot.values or {}


def print_findings(state: dict) -> None:
    findings = state.get("findings") or []
    print(f"\nFindings: {len(findings)}")
    for finding in findings:
        verification = finding.get("verification") or "NOT_CHECKED"
        location = finding.get("line") or "-"
        print(
            f"- {finding.get('rule_id') or finding.get('source')} "
            f"line {location} [{verification}]"
        )
        message = finding.get("message") or ""
        if message:
            print(f"  {message}")


def print_report(state: dict) -> None:
    print("\nStatus:", state.get("status", "unknown"))
    print("Summary:", state.get("summary", ""))
    if state.get("review_decision"):
        print("Decision:", state.get("review_decision"))
        print("A recorded decision does not prove the code is correct or secure.")
    print("\nAudit log")
    for entry in state.get("audit_log") or []:
        print(f"- {entry.get('agent')}: {entry.get('status')}")


def ask_decision() -> dict:
    allowed = ", ".join(sorted(REVIEW_DECISIONS))
    print("\nThe workflow is paused for a human decision.")
    print(f"Choose one of: {allowed}")
    while True:
        decision = input("Decision: ").strip().lower()
        if decision in REVIEW_DECISIONS:
            break
        print(f"Decision must be one of: {allowed}.")
    feedback = input("Feedback (optional): ").strip()
    return {"decision": decision, "feedback": feedback}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeSentinel live in the terminal."
    )
    parser.add_argument("file", nargs="?", help="Source file to review.")
    parser.add_argument(
        "--language",
        default="",
        help="python, javascript, java, c, or c++. Prompted when omitted.",
    )
    parser.add_argument(
        "--decision",
        default="",
        help=(
            "Record this human decision without prompting. "
            "Omit it during the live evaluation so the instructor can choose."
        ),
    )
    parser.add_argument("--feedback", default="", help="Optional decision feedback.")
    args = parser.parse_args()

    language = args.language.strip().lower()
    if not language:
        language = input(
            "Language [python, javascript, java, c, c++]: "
        ).strip().lower() or "python"
    if normalize_language(language) is None:
        raise SystemExit(f"Unsupported language: {language}")

    source = read_submission(args.file)
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("\nRunning the workflow\n")
    state = stream_until_pause(
        {
            "source_code": source,
            "language": language,
            "require_human_review": True,
        },
        config,
    )
    print_findings(state)

    if graph.get_state(config).next:
        decision = args.decision.strip().lower()
        if decision and decision not in REVIEW_DECISIONS:
            allowed = ", ".join(sorted(REVIEW_DECISIONS))
            raise SystemExit(f"Decision must be one of: {allowed}.")
        resume = (
            {"decision": decision, "feedback": args.feedback}
            if decision
            else ask_decision()
        )
        state = stream_until_pause(Command(resume=resume), config)

    print_report(state)


if __name__ == "__main__":
    main()
