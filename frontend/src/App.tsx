import { useMemo, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const SUPPORTED_LANGUAGES = [
  "python",
  "javascript",
  "java",
  "c",
  "c++",
] as const;

const DEFAULT_CODE = `def divide(a, b):
    return a / b

def calculate(values):
    total = 0
    for value in values:
        total += value
    return total
`;

type Finding = {
  source: string;
  rule_id: string;
  severity: string;
  line: number | null;
  column?: number | null;
  message: string;
  reasoning?: string;
  verification?: string;
  verification_evidence?: string;
};

type AnalysisResult = {
  status: string;
  summary: string;
  findings: Finding[];
  errors: string[];
  line_count: number;
  ai_summary?: string | null;
  thread_id: string;
  review_required: boolean;
  review_decision: string;
  review_feedback: string;
  review_completed: boolean;
  agent_status: Record<string, string>;
  agent_results: Record<string, unknown>;
  audit_log: Array<Record<string, unknown>>;
  pending?: boolean;
};

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected error occurred while analyzing the code.";
}

function formatLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getLanguageLabel(language: string): string {
  if (language === "c++") return "C++";
  if (language === "javascript") return "JavaScript";
  if (language === "python") return "Python";
  if (language === "java") return "Java";
  if (language === "c") return "C";
  return formatLabel(language || "Python");
}

function getLanguageFileName(language: string): string {
  if (language === "javascript") return "main.js";
  if (language === "java") return "Main.java";
  if (language === "c") return "main.c";
  if (language === "c++") return "main.cpp";
  return "main.py";
}

function getDefaultCodeForLanguage(language: string): string {
  if (language === "javascript") {
    return `function divide(a, b) {
  return a / b;
}

function calculate(values) {
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total;
}`;
  }

  if (language === "java") {
    return `class Main {
    public static double divide(double a, double b) {
        return a / b;
    }

    public static int calculate(int[] values) {
        int total = 0;
        for (int value : values) {
            total += value;
        }
        return total;
    }
}`;
  }

  if (language === "c") {
    return `#include <stdio.h>

int calculate(int values[], int length) {
    int total = 0;
    for (int i = 0; i < length; i++) {
        total += values[i];
    }
    return total;
}

int main(void) {
    int values[] = {1, 2, 3};
    printf("%d\n", calculate(values, 3));
    return 0;
}`;
  }

  if (language === "c++") {
    return `#include <iostream>
#include <vector>

double divide(double a, double b) {
    return a / b;
}

int calculate(const std::vector<int>& values) {
    int total = 0;
    for (int value : values) {
        total += value;
    }
    return total;
}

int main() {
    std::vector<int> values = {1, 2, 3};
    std::cout << calculate(values) << std::endl;
    return 0;
}`;
  }

  return DEFAULT_CODE;
}

function getSourceLabel(source: string): string {
  if (source === "gemini_ai") return "Gemini AI";
  if (source === "ruff") return "Ruff";
  if (source === "python_compile") return "Python Syntax";

  return formatLabel(source || "Analysis Tool");
}

function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    analysis_passed: "Analysis passed",
    issues_found: "Issues found",
    partial_analysis: "Partial analysis",
    analysis_error: "Analysis error",
    invalid_input: "Invalid input",
  };

  return labels[status] || formatLabel(status || "Unknown status");
}

function getVerificationLabel(verification?: string): string {
  if (!verification) return "Not assessed";

  const labels: Record<string, string> = {
    tool_reported: "Reported by tool",
    verified: "Verified",
    not_verified: "Not independently verified",
    requires_human_review: "Requires human review",
  };

  return labels[verification] || formatLabel(verification);
}

function isPotentialFinding(finding: Finding): boolean {
  return (
    finding.source === "gemini_ai" ||
    finding.verification === "requires_human_review" ||
    finding.verification === "not_verified"
  );
}

function App() {
  const [sourceCode, setSourceCode] = useState(DEFAULT_CODE);
  const [selectedLanguage, setSelectedLanguage] = useState<
    (typeof SUPPORTED_LANGUAGES)[number]
  >("python");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [filter, setFilter] = useState("all");
  const [reviewFeedback, setReviewFeedback] = useState("");
  const [reviewLoading, setReviewLoading] = useState(false);

  const lineCount = useMemo(
    () => (sourceCode.length ? sourceCode.split("\n").length : 0),
    [sourceCode],
  );

  const potentialCount = result
    ? result.findings.filter(isPotentialFinding).length
    : 0;

  const toolReportedCount = result
    ? result.findings.length - potentialCount
    : 0;

  const visibleFindings = useMemo(() => {
    if (!result) return [];

    if (filter === "potential") {
      return result.findings.filter(isPotentialFinding);
    }

    if (filter === "tools") {
      return result.findings.filter(
        (finding) => !isPotentialFinding(finding),
      );
    }

    return result.findings;
  }, [result, filter]);

  function parseAnalysisResult(payload: unknown): AnalysisResult {
    if (!payload || typeof payload !== "object") {
      throw new Error("The API returned an invalid response.");
    }

    const data = payload as Partial<AnalysisResult>;

    if (
      typeof data.status !== "string" ||
      typeof data.summary !== "string" ||
      !Array.isArray(data.findings) ||
      !Array.isArray(data.errors) ||
      typeof data.thread_id !== "string"
    ) {
      throw new Error(
        "The API response is missing required analysis fields.",
      );
    }

    return {
      status: data.status,
      summary: data.summary,
      findings: data.findings.filter(
        (finding): finding is Finding =>
          typeof finding === "object" &&
          finding !== null &&
          typeof finding.message === "string",
      ),
      errors: data.errors.map(String),
      line_count:
        typeof data.line_count === "number"
          ? data.line_count
          : lineCount,
      ai_summary:
        typeof data.ai_summary === "string"
          ? data.ai_summary
          : null,
      thread_id: data.thread_id,
      review_required: Boolean(data.review_required),
      review_decision: data.review_decision || "",
      review_feedback: data.review_feedback || "",
      review_completed: Boolean(data.review_completed),
      agent_status: data.agent_status || {},
      agent_results: data.agent_results || {},
      audit_log: Array.isArray(data.audit_log)
        ? data.audit_log
        : [],
      pending: Boolean(data.pending),
    };
  }

  async function fetchReviewState(
    threadId: string,
  ): Promise<AnalysisResult> {
    const response = await fetch(
      `${API_URL}/api/review/${encodeURIComponent(threadId)}`,
    );

    const data: unknown = await response.json().catch(() => null);

    if (!response.ok) {
      const detail =
        data &&
        typeof data === "object" &&
        "detail" in data &&
        typeof data.detail === "string"
          ? data.detail
          : `Could not retrieve review state (HTTP ${response.status}).`;

      throw new Error(detail);
    }

    return parseAnalysisResult(data);
  }

  async function runAnalysis() {
    setRequestError("");
    setResult(null);
    setFilter("all");
    setReviewFeedback("");

    if (!sourceCode.trim()) {
      setRequestError(
        `Enter some ${getLanguageLabel(selectedLanguage)} code before running the analysis.`,
      );
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_code: sourceCode,
          language: selectedLanguage,
          require_human_review: true,
        }),
      });

      const data: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          data &&
          typeof data === "object" &&
          "detail" in data &&
          typeof data.detail === "string"
            ? data.detail
            : `The API returned HTTP ${response.status}.`;

        throw new Error(detail);
      }

      setResult(parseAnalysisResult(data));
    } catch (error) {
      setRequestError(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  async function submitHumanReview(
    decision: "approve" | "reject",
  ) {
    if (!result?.thread_id || reviewLoading) return;

    setRequestError("");
    setReviewLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/review/${encodeURIComponent(result.thread_id)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            decision,
            feedback: reviewFeedback.trim(),
          }),
        },
      );

      const data: unknown = await response.json().catch(() => null);

      if (response.ok) {
        setResult(parseAnalysisResult(data));
        return;
      }

      // If the workflow was already resumed, retrieve its latest state.
      if (response.status === 409) {
        const latest = await fetchReviewState(result.thread_id);

        if (
          latest.review_completed &&
          latest.review_decision === decision &&
          !latest.pending
        ) {
          setResult(latest);
          return;
        }
      }

      const detail =
        data &&
        typeof data === "object" &&
        "detail" in data &&
        typeof data.detail === "string"
          ? data.detail
          : `The review request returned HTTP ${response.status}.`;

      throw new Error(detail);
    } catch (error) {
      setRequestError(getErrorMessage(error));

      // Refresh the state so a recorded decision is not lost in the UI.
      try {
        const latest = await fetchReviewState(result.thread_id);
        setResult(latest);
      } catch {
        // Preserve the original request error.
      }
    } finally {
      setReviewLoading(false);
    }
  }

  function loadExample() {
    setSourceCode(getDefaultCodeForLanguage(selectedLanguage));
    setResult(null);
    setRequestError("");
    setFilter("all");
    setReviewFeedback("");
  }

  function clearCode() {
    setSourceCode("");
    setResult(null);
    setRequestError("");
    setFilter("all");
    setReviewFeedback("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">CS</div>

          <div>
            <div className="brand-name">CodeSentinel</div>
            <div className="brand-caption">Code intelligence</div>
          </div>
        </div>

        <div className="sidebar-section-label">WORKSPACE</div>

        <nav className="side-nav" aria-label="Main navigation">
          <button className="nav-item active" type="button">
            <span className="nav-icon">⌂</span>
            <span>Code Analysis</span>
          </button>

          <button
            className="nav-item"
            type="button"
            onClick={loadExample}
          >
            <span className="nav-icon">＋</span>
            <span>New Analysis</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-bottom-mark">✳</div>
          <p>Better code.</p>
          <p>Safer tomorrow.</p>
          <span>Analyze · Verify · Improve</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>Code Analysis</strong>
          </div>

          <div className="topbar-right">
            <span className="connection-dot" />
            <span>Local analysis</span>
            <div className="avatar" aria-label="User initials">
              SC
            </div>
          </div>
        </header>

        <div className="content-wrap">
          <section className="welcome-row">
            <div className="welcome-copy">
              <span className="eyebrow">
                <span className="eyebrow-dot" />
                INTELLIGENT CODE REVIEW
              </span>

              <h1>
                Your code,
                <br />
                <span>under a closer look.</span>
              </h1>

              <p>
                Analyze Python, JavaScript, Java, C, or C++ code with
                static checks and AI-assisted review. Inspect findings,
                understand the evidence, and decide what needs attention.
              </p>
            </div>

            <div className="welcome-art" aria-hidden="true">
              <div className="art-ring art-ring-one" />
              <div className="art-ring art-ring-two" />
              <div className="art-center">CS</div>
              <span className="art-node art-node-one">{"{}"}</span>
              <span className="art-node art-node-two">✓</span>
              <span className="art-node art-node-three">⌘</span>
            </div>
          </section>

          <section className="editor-card">
            <div className="card-heading">
              <div>
                <div className="section-kicker">
                  ANALYSIS WORKSPACE
                </div>
                <h2>Review your code</h2>
                <p>
                  Submit Python source code for analysis. Submitted code
                  is not executed by this workflow.
                </p>
              </div>

              <select
                aria-label="Select source language"
                className="language-badge language-select"
                value={selectedLanguage}
                onChange={(event) => {
                  const nextLanguage = event.target.value as (typeof SUPPORTED_LANGUAGES)[number];
                  setSelectedLanguage(nextLanguage);
                  setSourceCode(getDefaultCodeForLanguage(nextLanguage));
                  setResult(null);
                  setRequestError("");
                }}
              >
                {SUPPORTED_LANGUAGES.map((language) => (
                  <option key={language} value={language}>
                    {getLanguageLabel(language)}
                  </option>
                ))}
              </select>
            </div>

            <div className="editor-toolbar">
              <div className="file-label">
                <span className="file-icon">
                  {selectedLanguage === "python"
                  ? "Py"
                  : selectedLanguage === "javascript"
                    ? "JS"
                    : selectedLanguage === "java"
                      ? "Ja"
                      : selectedLanguage === "c++"
                        ? "C++"
                        : "C"}
                </span>
                <span>{getLanguageFileName(selectedLanguage)}</span>
                <span className="file-status-dot" />
              </div>

              <div className="editor-meta">
                {lineCount} {lineCount === 1 ? "line" : "lines"}
                <span className="meta-separator">·</span>
                {sourceCode.length} characters
              </div>
            </div>

            <div className="editor">
              <div className="line-numbers" aria-hidden="true">
                {Array.from(
                  { length: Math.max(lineCount, 1) },
                  (_, index) => (
                    <span key={index}>{index + 1}</span>
                  ),
                )}
              </div>

              <textarea
                aria-label={`${getLanguageLabel(selectedLanguage)} source code`}
                value={sourceCode}
                onChange={(event) => {
                  setSourceCode(event.target.value);
                  setResult(null);
                  setRequestError("");
                }}
                spellCheck={false}
                autoCapitalize="off"
                autoComplete="off"
                autoCorrect="off"
                placeholder={`Paste your ${getLanguageLabel(selectedLanguage)} code here...`}
              />
            </div>

            <div className="card-footer">
              <div className="editor-note">
                <span className="info-icon">i</span>
                <span>Code is analyzed, not executed.</span>
              </div>

              <div className="editor-actions">
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={clearCode}
                  disabled={loading}
                >
                  Clear
                </button>

                <button
                  className="button button-secondary"
                  type="button"
                  onClick={loadExample}
                  disabled={loading}
                >
                  Load example
                </button>

                <button
                  className="button button-primary"
                  type="button"
                  onClick={runAnalysis}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="button-spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Run analysis
                      <span className="button-arrow">→</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {requestError && (
              <div className="request-error" role="alert">
                <span className="error-symbol">!</span>
                <div>
                  <strong>Analysis could not be completed</strong>
                  <p>{requestError}</p>
                  <span>
                    Check that the backend is running at {API_URL}.
                  </span>
                </div>
              </div>
            )}
          </section>

          {loading && (
            <section className="loading-card" aria-live="polite">
              <span className="loading-spinner" />
              <div>
                <strong>Analyzing your code</strong>
                <p>
                  Running the available checks and preparing the review.
                </p>
              </div>
            </section>
          )}

          {result && !loading && (
            <section className="results-section">
              <div className="results-heading">
                <div>
                  <div className="section-kicker">REVIEW REPORT</div>
                  <h2>Analysis results</h2>
                  <p>Findings and status returned by the backend.</p>

                  <p className="thread-id">
                    Review ID: <code>{result.thread_id}</code>
                  </p>
                </div>

                <span
                  className={`status-badge status-${result.status}`}
                  role="status"
                >
                  <span className="status-dot" />
                  {getStatusLabel(result.status)}
                </span>
              </div>

              <div className="metrics-grid">
                <article className="metric-card">
                  <span className="metric-label">Total findings</span>
                  <strong>{result.findings.length}</strong>
                  <span className="metric-foot">
                    Returned by analysis
                  </span>
                </article>

                <article className="metric-card">
                  <span className="metric-label">Tool findings</span>
                  <strong>{toolReportedCount}</strong>
                  <span className="metric-foot">
                    Non-AI reports
                  </span>
                </article>

                <article className="metric-card">
                  <span className="metric-label">Potential findings</span>
                  <strong>{potentialCount}</strong>
                  <span className="metric-foot">
                    Need careful review
                  </span>
                </article>

                <article className="metric-card">
                  <span className="metric-label">Lines analyzed</span>
                  <strong>{result.line_count}</strong>
                  <span className="metric-foot">
                    Submitted source
                  </span>
                </article>
              </div>

              <article className="summary-card">
                <div className="summary-heading">
                  <span className="summary-icon">✳</span>
                  <div>
                    <span className="section-kicker">
                      ANALYSIS SUMMARY
                    </span>
                    <h3>What the checks reported</h3>
                  </div>
                </div>

                <p className="summary-text">{result.summary}</p>

                <p className="summary-disclaimer">
                  Findings are based on the checks available in this run.
                  They should be reviewed with the evidence below.
                </p>

                {result.ai_summary && (
                  <div className="ai-summary">
                    <strong>AI review summary</strong>
                    <p>{result.ai_summary}</p>
                  </div>
                )}
              </article>

              <article className="workflow-card">
                <div className="workflow-heading">
                  <div>
                    <span className="section-kicker">
                      HUMAN-IN-THE-LOOP
                    </span>
                    <h3>Review decision</h3>
                  </div>

                  <span
                    className={`workflow-state ${
                      result.pending || result.review_required
                        ? "is-pending"
                        : "is-complete"
                    }`}
                  >
                    {result.pending || result.review_required
                      ? "Awaiting your review"
                      : result.review_completed
                        ? `Review ${formatLabel(result.review_decision)}`
                        : "No review pending"}
                  </span>
                </div>

                <p className="workflow-description">
                  Review the findings and evidence before recording a
                  decision. This records your review; it does not modify
                  or execute the submitted source code.
                </p>

                {(result.pending || result.review_required) &&
                !result.review_completed ? (
                  <div className="review-controls">
                    <label htmlFor="review-feedback">
                      Reviewer feedback (optional)
                    </label>

                    <textarea
                      id="review-feedback"
                      rows={3}
                      maxLength={2000}
                      value={reviewFeedback}
                      disabled={reviewLoading}
                      onChange={(event) =>
                        setReviewFeedback(event.target.value)
                      }
                      placeholder="Explain your decision..."
                    />

                    <div className="review-actions">
                      <button
                        className="button button-secondary"
                        type="button"
                        disabled={reviewLoading}
                        onClick={() =>
                          submitHumanReview("reject")
                        }
                      >
                        {reviewLoading
                          ? "Submitting..."
                          : "Reject review"}
                      </button>

                      <button
                        className="button button-primary"
                        type="button"
                        disabled={reviewLoading}
                        onClick={() =>
                          submitHumanReview("approve")
                        }
                      >
                        {reviewLoading
                          ? "Submitting..."
                          : "Approve review"}
                      </button>
                    </div>
                  </div>
                ) : result.review_completed ? (
                  <div className="review-completed">
                    <strong>
                      Decision recorded:{" "}
                      {formatLabel(result.review_decision)}
                    </strong>

                    {result.review_feedback && (
                      <p>{result.review_feedback}</p>
                    )}
                  </div>
                ) : (
                  <p className="workflow-description">
                    This analysis did not pause for a human decision.
                  </p>
                )}
              </article>

              <article className="trace-card">
                <div className="findings-heading">
                  <div>
                    <div className="section-kicker">
                      EXECUTION TRACE
                    </div>
                    <h3>Agent workflow</h3>
                  </div>

                  <span className="findings-total">
                    {Object.keys(result.agent_status).length} stages
                  </span>
                </div>

                {Object.keys(result.agent_status).length > 0 ? (
                  <div className="agent-status-list">
                    {Object.entries(result.agent_status).map(
                      ([agent, status]) => (
                        <div
                          className="agent-status-row"
                          key={agent}
                        >
                          <span>{formatLabel(agent)}</span>
                          <strong>
                            {formatLabel(String(status))}
                          </strong>
                        </div>
                      ),
                    )}
                  </div>
                ) : (
                  <p className="workflow-description">
                    No agent status details were returned.
                  </p>
                )}

                <details className="audit-details">
                  <summary>
                    View audit log ({result.audit_log.length} events)
                  </summary>

                  {result.audit_log.length > 0 ? (
                    <ol className="audit-list">
                      {result.audit_log.map((entry, index) => (
                        <li key={index}>
                          <code>
                            {String(
                              entry.event ||
                                entry.action ||
                                entry.node ||
                                "Workflow event",
                            )}
                          </code>

                          <span>
                            {String(
                              entry.message ||
                                entry.details ||
                                entry.status ||
                                "",
                            )}
                          </span>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p>No audit events were returned.</p>
                  )}
                </details>
              </article>

              {result.errors.length > 0 && (
                <article className="errors-card">
                  <h3>
                    <span className="error-symbol">!</span>
                    Incomplete or failed checks
                  </h3>

                  <ul>
                    {result.errors.map((error, index) => (
                      <li key={`${index}-${error}`}>{error}</li>
                    ))}
                  </ul>
                </article>
              )}

              <section className="findings-section">
                <div className="findings-heading">
                  <div>
                    <div className="section-kicker">
                      DETAILED REVIEW
                    </div>
                    <h3>Findings</h3>
                  </div>

                  <span className="findings-total">
                    {result.findings.length} total
                  </span>
                </div>

                <div className="filter-row" aria-label="Filter findings">
                  <button
                    type="button"
                    className={`filter-chip ${
                      filter === "all" ? "selected" : ""
                    }`}
                    onClick={() => setFilter("all")}
                  >
                    All <span>{result.findings.length}</span>
                  </button>

                  <button
                    type="button"
                    className={`filter-chip ${
                      filter === "tools" ? "selected" : ""
                    }`}
                    onClick={() => setFilter("tools")}
                  >
                    Tool reports <span>{toolReportedCount}</span>
                  </button>

                  <button
                    type="button"
                    className={`filter-chip ${
                      filter === "potential" ? "selected" : ""
                    }`}
                    onClick={() => setFilter("potential")}
                  >
                    Potential <span>{potentialCount}</span>
                  </button>
                </div>

                {visibleFindings.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-mark">
                      {result.findings.length === 0 ? "✓" : "—"}
                    </div>

                    <h4>
                      {result.findings.length === 0
                        ? "No findings were returned"
                        : "No findings in this filter"}
                    </h4>

                    <p>
                      {result.findings.length === 0
                        ? "The available checks did not report findings. This does not guarantee the code is bug-free."
                        : "Choose another filter to see the remaining findings."}
                    </p>
                  </div>
                ) : (
                  <div className="findings-list">
                    {visibleFindings.map((finding, index) => {
                      const potential = isPotentialFinding(finding);

                      return (
                        <article
                          className={`finding-card ${
                            potential
                              ? "finding-potential"
                              : "finding-tool"
                          }`}
                          key={`${finding.source}-${finding.rule_id}-${finding.line}-${index}`}
                        >
                          <div className="finding-topline">
                            <div className="finding-source">
                              <span className="source-mark">
                                {potential ? "AI" : "T"}
                              </span>

                              <span>
                                {getSourceLabel(finding.source)}
                              </span>

                              <span className="finding-separator">
                                ·
                              </span>

                              <span className="rule-id">
                                {finding.rule_id || "General finding"}
                              </span>
                            </div>

                            <span
                              className={`severity-badge severity-${(
                                finding.severity || "info"
                              ).toLowerCase()}`}
                            >
                              {formatLabel(
                                finding.severity || "info",
                              )}
                            </span>
                          </div>

                          <h4>{finding.message}</h4>

                          <div className="finding-meta">
                            {finding.line != null ? (
                              <span>Line {finding.line}</span>
                            ) : (
                              <span>Line not specified</span>
                            )}

                            {finding.column != null && (
                              <span>
                                Column {finding.column}
                              </span>
                            )}
                          </div>

                          {finding.reasoning && (
                            <p className="finding-reasoning">
                              {finding.reasoning}
                            </p>
                          )}

                          {finding.verification && (
                            <div className="verification-note">
                              <strong>
                                Verification:{" "}
                                {getVerificationLabel(
                                  finding.verification,
                                )}
                              </strong>

                              {finding.verification_evidence && (
                                <p>
                                  {finding.verification_evidence}
                                </p>
                              )}
                            </div>
                          )}

                          {!finding.verification && potential && (
                            <div className="verification-note">
                              <strong>
                                Verification: Not independently verified
                              </strong>

                              <p>
                                Review the code and the finding before
                                treating it as a confirmed defect.
                              </p>
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>

              <div className="safety-note">
                <span className="safety-note-icon">i</span>
                <p>
                  CodeSentinel does not execute submitted source code
                  during this analysis. A finding may require manual
                  reproduction, tests, or additional verification.
                </p>
              </div>
            </section>
          )}

          {!result && !loading && !requestError && (
            <section className="initial-state">
              <div className="initial-state-icon">⌘</div>

              <div>
                <h3>Ready when you are</h3>
                <p>
                  Enter Python code above and select{" "}
                  <strong>Run analysis</strong> to see the backend’s
                  actual results here.
                </p>
              </div>
            </section>
          )}

          <footer className="page-footer">
            <span>CodeSentinel</span>
            <span>Analyze thoughtfully. Verify independently.</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;