import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_ruff(source_code: str) -> dict[str, Any]:
    """Run Ruff against source supplied via stdin; never execute that source."""
    executable_name = "ruff.exe" if os.name == "nt" else "ruff"
    executable = Path(sys.executable).with_name(executable_name)

    if not executable.is_file():
        return {
            "ok": False,
            "findings": [],
            "error": (
                f"Ruff executable not found at {executable}. "
                "Install project dependencies in the active virtual environment."
            ),
        }

    command = [
        str(executable),
        "check",
        "--output-format",
        "json",
        "--stdin-filename",
        "submitted.py",
        "-",
    ]

    try:
        result = subprocess.run(
            command,
            input=source_code,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "findings": [],
            "error": "Ruff exceeded its 10-second execution limit.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "findings": [],
            "error": f"Could not start Ruff: {exc}",
        }

    if result.returncode not in (0, 1):
        detail = result.stderr.strip() or result.stdout.strip()
        return {
            "ok": False,
            "findings": [],
            "error": f"Ruff failed with exit code {result.returncode}: {detail}",
        }

    try:
        diagnostics = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "findings": [],
            "error": "Ruff returned output that could not be parsed as JSON.",
        }

    findings = []

    for diagnostic in diagnostics:
        location = diagnostic.get("location") or {}
        end_location = diagnostic.get("end_location") or {}

        findings.append(
            {
                "source": "ruff",
                "rule_id": diagnostic.get("code", "RUFF-UNKNOWN"),
                "severity": "warning",
                "line": location.get("row"),
                "column": location.get("column"),
                "end_line": end_location.get("row"),
                "end_column": end_location.get("column"),
                "message": diagnostic.get("message", "No message provided."),
            }
        )

    return {
        "ok": True,
        "findings": findings,
        "error": None,
    }
