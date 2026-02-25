"""Adapters for benchmarking local candidate model and external baselines."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from infer.schemas import TaskRequest, TaskResponse, ValidationResult


class CommandRunnerAdapter:
    """Runs a model adapter command that reads request JSON and prints response JSON."""

    def __init__(self, command: list[str]) -> None:
        self.command = command

    def run(self, request: TaskRequest) -> TaskResponse:
        proc = subprocess.run(
            self.command,
            input=json.dumps(request.to_dict()).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"adapter command failed ({proc.returncode}): {proc.stderr.decode('utf-8', errors='ignore')}"
            )
        payload = json.loads(proc.stdout.decode("utf-8"))
        return TaskResponse.create(
            status=payload["status"],
            patch=payload.get("patch", ""),
            files=[],
            validation=ValidationResult(
                apply_ok=payload["validation"]["apply_ok"],
                lint_ok=payload["validation"]["lint_ok"],
                build_ok=payload["validation"]["build_ok"],
                notes=payload["validation"].get("notes", []),
            ),
        )


class FrozenJsonBaseline:
    """Uses precomputed baseline results keyed by case id for offline parity checks."""

    def __init__(self, baseline_results: Path) -> None:
        self.by_id = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in baseline_results.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    def get(self, case_id: str) -> dict:
        if case_id not in self.by_id:
            raise KeyError(f"missing baseline case: {case_id}")
        return self.by_id[case_id]
