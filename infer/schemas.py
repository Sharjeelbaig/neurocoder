"""Typed contracts for inference and training samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskType = Literal["page_generate", "patch_edit"]
TaskStatus = Literal["ok", "needs_retry", "failed"]

_ALLOWED_TASK_TYPES = {"page_generate", "patch_edit"}
_ALLOWED_STATUS = {"ok", "needs_retry", "failed"}


class SchemaValidationError(ValueError):
    """Raised when request/response payloads violate schema."""


@dataclass(slots=True)
class SourceFile:
    path: str
    content: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceFile":
        path = payload.get("path")
        content = payload.get("content")
        if not isinstance(path, str) or not path:
            raise SchemaValidationError("files[].path must be a non-empty string")
        if not isinstance(content, str):
            raise SchemaValidationError("files[].content must be a string")
        return cls(path=path, content=content)


@dataclass(slots=True)
class Constraints:
    framework: str = "react-tailwind"
    output: str = "unified_diff"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Constraints":
        if payload is None:
            payload = {}
        framework = payload.get("framework", "react-tailwind")
        output = payload.get("output", "unified_diff")
        if framework != "react-tailwind":
            raise SchemaValidationError("constraints.framework must be 'react-tailwind'")
        if output != "unified_diff":
            raise SchemaValidationError("constraints.output must be 'unified_diff'")
        return cls(framework=framework, output=output)


@dataclass(slots=True)
class TaskRequest:
    task_type: TaskType
    instruction: str
    files: list[SourceFile]
    constraints: Constraints = field(default_factory=Constraints)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRequest":
        task_type = payload.get("task_type")
        instruction = payload.get("instruction")
        files_payload = payload.get("files")

        if task_type not in _ALLOWED_TASK_TYPES:
            raise SchemaValidationError(
                "task_type must be one of: page_generate, patch_edit"
            )
        if not isinstance(instruction, str) or not instruction.strip():
            raise SchemaValidationError("instruction must be a non-empty string")
        if not isinstance(files_payload, list):
            raise SchemaValidationError("files must be a list")

        files = [SourceFile.from_dict(item) for item in files_payload]
        constraints = Constraints.from_dict(payload.get("constraints"))
        return cls(
            task_type=task_type,
            instruction=instruction.strip(),
            files=files,
            constraints=constraints,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    apply_ok: bool
    lint_ok: bool
    build_ok: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutputFile:
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskResponse:
    status: TaskStatus
    patch: str
    files: list[OutputFile]
    validation: ValidationResult

    @classmethod
    def create(
        cls,
        *,
        status: str,
        patch: str = "",
        files: list[OutputFile] | None = None,
        validation: ValidationResult | None = None,
    ) -> "TaskResponse":
        if status not in _ALLOWED_STATUS:
            raise SchemaValidationError("status must be one of: ok, needs_retry, failed")
        if files is None:
            files = []
        if validation is None:
            validation = ValidationResult(False, False, False, ["validation not executed"])
        return cls(status=status, patch=patch, files=files, validation=validation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "patch": self.patch,
            "files": [f.to_dict() for f in self.files],
            "validation": self.validation.to_dict(),
        }


@dataclass(slots=True)
class TrainExample:
    id: str
    source_license: str
    task_type: TaskType
    instruction: str
    context_files: list[SourceFile]
    target_patch: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainExample":
        if payload.get("task_type") not in _ALLOWED_TASK_TYPES:
            raise SchemaValidationError("train example task_type is invalid")
        context_payload = payload.get("context_files", [])
        if not isinstance(context_payload, list):
            raise SchemaValidationError("context_files must be a list")
        context_files = [SourceFile.from_dict(item) for item in context_payload]
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SchemaValidationError("metadata must be an object")
        return cls(
            id=str(payload.get("id", "")),
            source_license=str(payload.get("source_license", "")),
            task_type=payload["task_type"],
            instruction=str(payload.get("instruction", "")),
            context_files=context_files,
            target_patch=str(payload.get("target_patch", "")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_license": self.source_license,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "context_files": [f.__dict__ for f in self.context_files],
            "target_patch": self.target_patch,
            "metadata": self.metadata,
        }
