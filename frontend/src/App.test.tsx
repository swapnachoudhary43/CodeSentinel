import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const analysisResult = {
  status: "pending_review",
  summary: "Analysis reported 1 finding.",
  findings: [
    {
      source: "logic_agent",
      rule_id: "LOGIC-DIV0",
      severity: "warning",
      line: 2,
      message: "Division expression found.",
      verification: "UNRESOLVED",
      verification_evidence: "The divisor is unknown.",
    },
    {
      source: "ruff",
      rule_id: "F841",
      severity: "warning",
      line: 3,
      message: "Local variable is unused.",
      verification: "NOT_CHECKED",
    },
  ],
  errors: [],
  line_count: 2,
  thread_id: "thread-1",
  review_required: true,
  review_decision: "",
  review_feedback: "",
  review_completed: false,
  context: { detected_language: "python" },
  agent_status: { logic_bug_agent: "completed" },
  agent_results: {},
  specialized_agent_results: {},
  edge_case_recommendations: {},
  audit_log: [{ agent: "logic_bug_agent", status: "completed" }],
  pending: true,
};

function sse(events: object[]) {
  const body = events
    .map((event) => `data: ${JSON.stringify(event)}\n\n`)
    .join("");
  return new Response(body, { status: 200 });
}

function clickRunAnalysis() {
  fireEvent.click(screen.getByRole("button", { name: /Run analysis/ }));
}

describe("CodeSentinel workspace", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        const target = String(url);

        if (target.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "healthy" }), {
            status: 200,
          });
        }

        if (target.endsWith("/api/analyze/stream")) {
          const request = JSON.parse(String(init?.body));
          expect(request.require_human_review).toBe(true);
          expect(request.language).toBe("python");
          return sse([
            {
              type: "agent_update",
              agent: "input_validation",
              status: "completed",
            },
            { type: "complete", result: analysisResult },
          ]);
        }

        if (target.includes("/api/review/thread-1") && init?.method === "POST") {
          const request = JSON.parse(String(init.body));
          return new Response(
            JSON.stringify({
              ...analysisResult,
              status: "issues_found",
              summary: "Human review decision recorded.",
              pending: false,
              review_required: false,
              review_completed: true,
              review_decision: request.decision,
              review_feedback: request.feedback,
            }),
            { status: 200 },
          );
        }

        if (target.includes("/api/review/thread-1")) {
          return new Response(
            JSON.stringify({
              ...analysisResult,
              review_completed: true,
              review_decision: "defer",
              pending: false,
            }),
            { status: 200 },
          );
        }

        return new Response("missing", { status: 404 });
      }),
    );
  });

  it("rejects an empty editor before calling the API", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    clickRunAnalysis();

    expect((await screen.findByRole("alert")).textContent).toContain(
      "Enter some Python code",
    );
    const calls = vi.mocked(fetch).mock.calls.filter((call) =>
      String(call[0]).includes("/api/analyze"),
    );
    expect(calls).toHaveLength(0);
  });

  it("renders streamed findings and sends one analysis request", async () => {
    render(<App />);

    clickRunAnalysis();

    expect(
      await screen.findByText("Division expression found."),
    ).toBeTruthy();
    expect(screen.getAllByText(/logic bug agent/i).length).toBeGreaterThan(0);

    const calls = vi.mocked(fetch).mock.calls.filter((call) =>
      String(call[0]).includes("/api/analyze/stream"),
    );
    expect(calls).toHaveLength(1);
    expect(
      (screen.getByRole("button", { name: "Defer" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("filters tool reports separately from potential findings", async () => {
    render(<App />);
    clickRunAnalysis();
    await screen.findByText("Division expression found.");

    fireEvent.click(screen.getByRole("button", { name: /Tool reports/ }));
    expect(screen.getByText("Local variable is unused.")).toBeTruthy();
    expect(screen.queryByText("Division expression found.")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Potential/ }));
    expect(screen.getByText("Division expression found.")).toBeTruthy();
    expect(screen.queryByText("Local variable is unused.")).toBeNull();
  });

  it("records a human review decision from the backend response", async () => {
    render(<App />);
    clickRunAnalysis();
    await screen.findByRole("button", { name: "Reject" });

    fireEvent.click(screen.getByRole("button", { name: "Reject" }));

    expect(await screen.findByText(/Decision recorded: reject/i)).toBeTruthy();
  });

  it("shows a backend failure instead of a completed workflow", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).endsWith("/health")) {
          return new Response(JSON.stringify({ status: "healthy" }), {
            status: 200,
          });
        }
        throw new TypeError("Failed to fetch");
      }),
    );

    render(<App />);
    clickRunAnalysis();

    expect((await screen.findByRole("alert")).textContent).toContain(
      "Failed to fetch",
    );
    expect(screen.queryByText("Analysis results")).toBeNull();
  });

  it("changes the example when the language changes", () => {
    render(<App />);

    fireEvent.change(screen.getByLabelText("Select source language"), {
      target: { value: "javascript" },
    });

    const editor = screen.getByLabelText(
      "JavaScript source code",
    ) as HTMLTextAreaElement;
    expect(editor.value).toContain("function divide");
  });
});
