
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MAX_SOURCE_LENGTH = 50_000
TOOL_TIMEOUT_SECONDS = 15

SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "java",
    "c",
    "c++",
}


def _failure(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "findings": [],
        "error": message,
    }


def _run_command(
    command: list[str],
    *,
    input_text: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run a fixed analysis command with a timeout, never through a shell."""
    try:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_SECONDS,
            cwd=cwd,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure(
            f"Analysis tool exceeded its {TOOL_TIMEOUT_SECONDS}-second limit."
        )
    except OSError as exc:
        return _failure(
            f"Could not start analysis tool: {type(exc).__name__}: {exc}"
        )

    return {
        "ok": True,
        "returncode": result.returncode,
        "stdout": result.stdout[-20_000:],
        "stderr": result.stderr[-20_000:],
    }


def _tool_finding(
    source: str,
    rule_id: str,
    severity: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "rule_id": rule_id,
        "severity": severity,
        "line": line,
        "column": column,
        "message": message,
    }


def _diagnostic_finding(
    source: str,
    message: str,
    *,
    rule_id: str,
    line: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    return _tool_finding(
        source=source,
        rule_id=rule_id,
        severity="error",
        message=message,
        line=line,
        column=column,
    )


# ---------------------------------------------------------
# Python: Ruff
# ---------------------------------------------------------

def run_ruff(source_code: str) -> dict[str, Any]:
    """Run Ruff against submitted Python source via stdin."""
    executable_name = "ruff.exe" if os.name == "nt" else "ruff"
    executable = shutil.which(executable_name) or shutil.which("ruff")

    if executable is None:
        script_dir = Path(sys.executable).resolve().parent
        script_candidates = [
            script_dir / executable_name,
            script_dir / "Scripts" / executable_name,
            script_dir / "bin" / "ruff",
            script_dir / "bin" / "ruff.exe",
        ]
        for candidate in script_candidates:
            if candidate.is_file():
                executable = str(candidate)
                break

    if executable is None:
        return _failure(
            "Ruff executable not found in PATH or the active Python environment. "
            "Install project dependencies in the active environment."
        )

    result = _run_command(
        [
            str(executable),
            "check",
            "--output-format",
            "json",
            "--stdin-filename",
            "submitted.py",
            "-",
        ],
        input_text=source_code,
    )

    if not result["ok"]:
        return result

    if result["returncode"] not in (0, 1):
        return _failure(
            "Ruff failed: "
            + (result["stderr"] or result["stdout"] or "Unknown error")
        )

    try:
        diagnostics = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return _failure("Ruff returned invalid JSON.")

    findings = []

    for item in diagnostics:
        location = item.get("location") or {}
        end_location = item.get("end_location") or {}

        findings.append({
            "source": "ruff",
            "rule_id": item.get("code", "RUFF-UNKNOWN"),
            "severity": "warning",
            "line": location.get("row"),
            "column": location.get("column"),
            "end_line": end_location.get("row"),
            "end_column": end_location.get("column"),
            "message": item.get("message", "No message provided."),
        })

    return {
        "ok": True,
        "findings": findings,
        "error": None,
    }


# ---------------------------------------------------------
# C and C++ syntax checking
# ---------------------------------------------------------

def _check_c_family(
    source_code: str,
    language: str,
) -> dict[str, Any]:
    is_cpp = language == "c++"

    candidates = (
        ["clang++", "g++"]
        if is_cpp
        else ["clang", "gcc"]
    )

    compiler = next(
        (shutil.which(name) for name in candidates if shutil.which(name)),
        None,
    )

    if not compiler:
        return _failure(
            f"No suitable {'C++' if is_cpp else 'C'} compiler found. "
            f"Install one of: {', '.join(candidates)}."
        )

    with tempfile.TemporaryDirectory(prefix="codesentinel-") as temp_dir:
        extension = ".cpp" if is_cpp else ".c"
        source_path = Path(temp_dir) / f"submitted{extension}"

        try:
            source_path.write_text(source_code, encoding="utf-8")
        except OSError as exc:
            return _failure(f"Could not prepare source file: {exc}")

        command = [
            compiler,
            "-fsyntax-only",
            "-fno-diagnostics-color",
        ]

        if is_cpp:
            command.extend(["-std=c++17", "-x", "c++"])
        else:
            command.extend(["-std=c11", "-x", "c"])

        command.append(str(source_path))

        result = _run_command(command, cwd=temp_dir)

    if not result["ok"]:
        return result

    if result["returncode"] == 0:
        return {"ok": True, "findings": [], "error": None}

    output = result["stderr"] or result["stdout"]
    findings = []

    for line in output.splitlines():
        if " error:" not in line:
            continue

        findings.append(
            _diagnostic_finding(
                source="clang" if "clang" in Path(compiler).name else "gcc",
                rule_id="CXX-SYNTAX" if is_cpp else "C-SYNTAX",
                message=line.strip()[:1000],
            )
        )

    if not findings:
        findings.append(
            _diagnostic_finding(
                source="compiler",
                rule_id="CXX-SYNTAX" if is_cpp else "C-SYNTAX",
                message=output.strip()[:1000] or "Compilation check failed.",
            )
        )

    return {
        "ok": True,
        "findings": findings,
        "error": None,
    }


# ---------------------------------------------------------
# Java syntax/type checking
# ---------------------------------------------------------

def _check_java(source_code: str) -> dict[str, Any]:
    javac = shutil.which("javac")

    if not javac:
        return _failure("javac was not found. Install a JDK and add it to PATH.")

    with tempfile.TemporaryDirectory(prefix="codesentinel-") as temp_dir:
        source_path = Path(temp_dir) / "Main.java"

        try:
            source_path.write_text(source_code, encoding="utf-8")
        except OSError as exc:
            return _failure(f"Could not prepare Java source: {exc}")

        result = _run_command(
            [
                javac,
                "-proc:none",
                "-implicit:none",
                "-d",
                temp_dir,
                str(source_path),
            ],
            cwd=temp_dir,
        )

    if not result["ok"]:
        return result

    if result["returncode"] == 0:
        return {"ok": True, "findings": [], "error": None}

    output = result["stderr"] or result["stdout"]
    findings = []

    for line in output.splitlines():
        if ".java:" not in line:
            continue

        findings.append(
            _diagnostic_finding(
                source="javac",
                rule_id="JAVA-COMPILE",
                message=line.strip()[:1000],
            )
        )

    if not findings:
        findings.append(
            _diagnostic_finding(
                source="javac",
                rule_id="JAVA-COMPILE",
                message=output.strip()[:1000] or "Java compilation failed.",
            )
        )

    return {"ok": True, "findings": findings, "error": None}


# ---------------------------------------------------------
# JavaScript syntax checking
# ---------------------------------------------------------

def _check_javascript(source_code: str) -> dict[str, Any]:
    node = shutil.which("node")

    if not node:
        return _failure("Node.js was not found. Install Node.js and add it to PATH.")

    result = _run_command(
        [node, "--check", "-"],
        input_text=source_code,
    )

    if not result["ok"]:
        return result

    if result["returncode"] == 0:
        return {"ok": True, "findings": [], "error": None}

    output = result["stderr"] or result["stdout"]

    return {
        "ok": True,
        "findings": [
            _diagnostic_finding(
                source="node",
                rule_id="JS-SYNTAX",
                message=output.strip()[:1000] or "JavaScript syntax check failed.",
            )
        ],
        "error": None,
    }


# ---------------------------------------------------------
# Public language-aware syntax checker
# ---------------------------------------------------------

def run_syntax_check(
    source_code: str,
    language: str,
) -> dict[str, Any]:
    """Perform syntax/type checks without intentionally running user code."""
    if not isinstance(source_code, str) or not source_code.strip():
        return _failure("Source code must be a non-empty string.")

    if len(source_code) > MAX_SOURCE_LENGTH:
        return _failure(
            f"Source code exceeds the {MAX_SOURCE_LENGTH:,}-character limit."
        )

    normalized = language.strip().lower()

    if normalized not in SUPPORTED_LANGUAGES:
        return _failure(f"Unsupported language: {language}")

    if normalized == "python":
        try:
            compile(source_code, "<submitted_code>", "exec")
            return {"ok": True, "findings": [], "error": None}
        except SyntaxError as exc:
            return {
                "ok": True,
                "findings": [
                    _diagnostic_finding(
                        source="python_compile",
                        rule_id="PY-SYNTAX",
                        message=exc.msg,
                        line=exc.lineno,
                        column=exc.offset,
                    )
                ],
                "error": None,
            }
        except (ValueError, TypeError) as exc:
            return _failure(f"Python syntax check failed: {exc}")

    if normalized in ("c", "c++"):
        return _check_c_family(source_code, normalized)

    if normalized == "java":
        return _check_java(source_code)

    if normalized == "javascript":
        return _check_javascript(source_code)

    return _failure(f"No syntax checker is configured for {normalized}.")
