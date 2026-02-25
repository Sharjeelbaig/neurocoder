"""Narrow-domain benchmark harness for patch and page tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from infer.schemas import TaskRequest, TaskResponse


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    task: TaskRequest
    expected_apply_ok: bool = True
    expected_build_ok: bool = True
    expected_lint_ok: bool = True


@dataclass(slots=True)
class BenchmarkResult:
    id: str
    status: str
    apply_ok: bool
    build_ok: bool
    lint_ok: bool
    notes: list[str]


@dataclass(slots=True)
class Scorecard:
    total: int
    success_rate: float
    apply_rate: float
    build_rate: float
    lint_rate: float


class TaskRunner(Protocol):
    def run(self, request: TaskRequest) -> TaskResponse:
        ...


class LocalServiceRunner:
    def __init__(self, service: object) -> None:
        self.service = service

    def run(self, request: TaskRequest) -> TaskResponse:
        return self.service.handle(request)


class BenchmarkSuite:
    def __init__(self, cases: list[BenchmarkCase]) -> None:
        self.cases = cases

    @classmethod
    def from_jsonl(cls, path: Path) -> "BenchmarkSuite":
        cases: list[BenchmarkCase] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            task = TaskRequest.from_dict(payload["task"])
            cases.append(
                BenchmarkCase(
                    id=payload["id"],
                    task=task,
                    expected_apply_ok=payload.get("expected_apply_ok", True),
                    expected_build_ok=payload.get("expected_build_ok", True),
                    expected_lint_ok=payload.get("expected_lint_ok", True),
                )
            )
        return cls(cases)

    def run(self, runner: TaskRunner) -> tuple[list[BenchmarkResult], Scorecard]:
        results: list[BenchmarkResult] = []
        success_count = 0
        apply_count = 0
        build_count = 0
        lint_count = 0

        for case in self.cases:
            response = runner.run(case.task)
            result = BenchmarkResult(
                id=case.id,
                status=response.status,
                apply_ok=response.validation.apply_ok,
                build_ok=response.validation.build_ok,
                lint_ok=response.validation.lint_ok,
                notes=response.validation.notes,
            )
            results.append(result)

            success = (
                response.validation.apply_ok == case.expected_apply_ok
                and response.validation.build_ok == case.expected_build_ok
                and response.validation.lint_ok == case.expected_lint_ok
            )
            if success:
                success_count += 1
            if response.validation.apply_ok:
                apply_count += 1
            if response.validation.build_ok:
                build_count += 1
            if response.validation.lint_ok:
                lint_count += 1

        total = max(len(self.cases), 1)
        scorecard = Scorecard(
            total=len(self.cases),
            success_rate=success_count / total,
            apply_rate=apply_count / total,
            build_rate=build_count / total,
            lint_rate=lint_count / total,
        )
        return results, scorecard



def write_results(results: list[BenchmarkResult], scorecard: Scorecard, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(asdict(row), sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    (output_dir / "scorecard.json").write_text(
        json.dumps(asdict(scorecard), indent=2, sort_keys=True),
        encoding="utf-8",
    )
