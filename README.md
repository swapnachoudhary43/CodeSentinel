# CodeSentinel — AI-Powered Multi-Agent Code Review

CodeSentinel is a multi-language code review platform that uses a multi-agent workflow to identify potential code issues, perform lightweight static checks, verify findings where possible, and involve a human reviewer before completing the analysis.

The project uses **LangGraph** to coordinate the workflow, **Gemini** for AI-assisted code review, **Ruff** for static analysis, and a **React + TypeScript** frontend.

## Features

- **Input validation:** Accepts non-empty source code up to 50,000 characters for Python, JavaScript, Java, C, and C++.
- **Syntax checking:** Uses lightweight native syntax validation per language without executing submitted code.
- **Static analysis:** Runs Ruff for Python and keeps the workflow fast for non-Python languages with lightweight validation.
- **AI code review:** Uses Gemini to generate structured findings and explanations.
- **Finding verification:** Uses Python AST inspection for Python findings, with broader coverage depending on available language tooling.
- **Bounded reassessment:** Sends verification feedback to Gemini for at most one reassessment attempt.
- **Human-in-the-loop review:** Allows a reviewer to approve or reject findings and provide feedback.
- **Audit trail:** Records agent statuses, results, and timestamps.
- **Analysis report:** Produces a final summary of the workflow and its limitations.

## Technology Stack

### Backend
- Python
- FastAPI
- LangGraph
- Google Gemini API (`google-genai`)
- Ruff
- Python AST
- Lightweight syntax validation for JavaScript, Java, C, and C++
- Pydantic

### Frontend
- React
- TypeScript
- Vite
- CSS

## Project Structure

```text
CodeSentinel/
├── ai_agent.py
├── api.py
├── config.py
├── graph.py
├── state.py
├── tools.py
├── verification.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── tests/
│   ├── test_graph.py
│   └── test_verification.py
└── frontend/
    ├── public/
    ├── src/
    ├── index.html
    ├── package.json
    ├── package-lock.json
    └── vite.config.ts
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CodeSentinel
```

### 2. Create and activate a Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root using `.env.example` as a template.

Set:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Keep `.env` private. Do not commit it to Git.

### 5. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## Running CodeSentinel

Use two PowerShell terminals. Both commands start from the project root.

Terminal 1, API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2, workspace:

```powershell
npm run dev --prefix frontend -- --host 127.0.0.1 --port 5173
```

The Vite dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`. If the API port changes, set `BACKEND_PROXY_TARGET` before starting Vite.

Open the workspace:

```text
http://127.0.0.1:5173
```

Terminal-only live demo, with the instructor choosing the decision:

```powershell
.\.venv\Scripts\python.exe demo.py
```

Or review a file and record a decision:

```powershell
.\.venv\Scripts\python.exe demo.py .\sample.py --language python --decision defer
```

Backend API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## Multi-Agent Workflow

The workflow is coordinated using LangGraph.

1. **Input Validation:** Checks the submitted source, language, and input-size limit.
2. **Syntax Checker:** Uses Python compilation to identify syntax errors.
3. **Static Analysis Agent:** Runs Ruff and collects tool-generated findings.
4. **AI Review Agent:** Requests structured code-review findings from Gemini.
5. **Verification Agent:** Inspects findings using static evidence, including AST analysis for selected division-related concerns.
6. **AI Reassessment Agent:** Can request one reassessment using verification feedback.
7. **Human Review:** Pauses when findings exist and human review is requested. The reviewer can approve, approve with warnings, request changes, reject, or defer.
8. **Report Agent:** Produces the final summary and records the completed workflow.

The workflow also maintains shared state and an audit log containing agent execution information.

## Testing

Run backend tests:

```powershell
python -m pytest -v
```

Build the frontend:

```powershell
npm run build --prefix frontend
```

Run frontend lint:

```powershell
npm run lint --prefix frontend
```

## Deployment

The public frontend and API stay separate from the local dev proxy.

- Frontend: Vercel, with the project root set to `frontend`. Build command `npm run build`. Set `VITE_API_URL` to the Render origin, including `https://`, with no trailing path and no localhost address.
- Backend: Render, using `render.yaml`. Set `GEMINI_API_KEY` in the Render dashboard. Set `FRONTEND_URL` and `CORS_ORIGINS` to the Vercel origin. Do not put localhost in those values.
- Review threads on Render are stored in `data/codesentinel.sqlite` for the life of that service disk. A new deploy or a free-instance restart can clear them. Run one instance so a decision reaches the same checkpoint.

## Safety and Limitations

- Submitted Python source code is not executed.
- Syntax validation uses Python's compilation mechanism.
- Ruff receives source code through standard input.
- Gemini findings are AI-generated suggestions and should not automatically be treated as confirmed defects.
- Verification is limited to implemented static checks. For example, detecting a division expression does not prove that its divisor can be zero.
- Human approval or rejection records a reviewer's decision; it does not independently prove or disprove a finding.
- Passing the available checks does not guarantee that code is logically correct, secure, or free from runtime errors.
- The workflow accepts Python, JavaScript, Java, C, and C++ source code, with language-specific validation depending on available local tooling.
- Gemini-based review requires a valid API key and network access.

## Project Objective

CodeSentinel demonstrates how multiple specialized agents, external analysis tools, shared workflow state, bounded reassessment, and human feedback can work together in an AI-assisted software review process.