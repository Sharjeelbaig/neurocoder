"""Unified diff parsing, validation, and in-memory application."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Iterable

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffValidationError(ValueError):
    """Raised when unified diff format is invalid."""


@dataclass(slots=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass(slots=True)
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[Hunk]


@dataclass(slots=True)
class ApplyResult:
    ok: bool
    files: dict[str, str]
    notes: list[str]


def _normalize_path(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if raw_path == "/dev/null":
        return raw_path
    if raw_path.startswith("a/") or raw_path.startswith("b/"):
        return raw_path[2:]
    return raw_path


def _parse_hunk_header(line: str) -> tuple[int, int, int, int]:
    match = _HUNK_RE.match(line)
    if not match:
        raise DiffValidationError(f"invalid hunk header: {line}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    return old_start, old_count, new_start, new_count


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    if not diff_text.strip():
        raise DiffValidationError("patch is empty")

    lines = diff_text.splitlines()
    patches: list[FilePatch] = []
    current_patch: FilePatch | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git ") or line.startswith("index "):
            i += 1
            continue

        if line.startswith("--- "):
            old_path = _normalize_path(line[4:])
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise DiffValidationError("missing +++ header after ---")
            new_path = _normalize_path(lines[i][4:])
            current_patch = FilePatch(old_path=old_path, new_path=new_path, hunks=[])
            patches.append(current_patch)
            i += 1
            continue

        if line.startswith("@@ "):
            if current_patch is None:
                raise DiffValidationError("hunk encountered before file header")
            old_start, old_count, new_start, new_count = _parse_hunk_header(line)
            i += 1
            hunk_lines: list[str] = []

            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("@@ ") or nxt.startswith("--- ") or nxt.startswith("diff --git "):
                    break
                if nxt == "\\ No newline at end of file":
                    i += 1
                    continue
                if not nxt:
                    # unified diff uses explicit prefixes; blank logical lines still begin with prefix.
                    raise DiffValidationError("blank unprefixed line inside hunk")
                if nxt[0] not in {" ", "+", "-"}:
                    raise DiffValidationError(f"invalid hunk line prefix: {nxt}")
                hunk_lines.append(nxt)
                i += 1

            if not hunk_lines:
                raise DiffValidationError("hunk has no body lines")

            computed_old = sum(1 for item in hunk_lines if item[0] in {" ", "-"})
            computed_new = sum(1 for item in hunk_lines if item[0] in {" ", "+"})
            if computed_old != old_count:
                raise DiffValidationError(
                    f"old line count mismatch in hunk: expected {old_count}, got {computed_old}"
                )
            if computed_new != new_count:
                raise DiffValidationError(
                    f"new line count mismatch in hunk: expected {new_count}, got {computed_new}"
                )

            current_patch.hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )
            continue

        if not line.strip():
            i += 1
            continue

        raise DiffValidationError(f"unexpected diff line: {line}")

    if not patches:
        raise DiffValidationError("patch has no file sections")
    for patch in patches:
        if not patch.hunks:
            raise DiffValidationError(f"file patch for {patch.new_path} has no hunks")

    return patches


def validate_unified_diff(diff_text: str) -> tuple[bool, list[str]]:
    try:
        parse_unified_diff(diff_text)
        return True, []
    except DiffValidationError as exc:
        return False, [str(exc)]


def _split_lines(content: str) -> tuple[list[str], bool]:
    has_trailing_newline = content.endswith("\n")
    lines = content.splitlines()
    return lines, has_trailing_newline


def _join_lines(lines: list[str], trailing_newline: bool) -> str:
    if not lines:
        return ""
    joined = "\n".join(lines)
    return joined + ("\n" if trailing_newline else "")


def apply_unified_diff(files: dict[str, str], diff_text: str) -> ApplyResult:
    try:
        patches = parse_unified_diff(diff_text)
    except DiffValidationError as exc:
        return ApplyResult(ok=False, files=files.copy(), notes=[str(exc)])

    result = files.copy()
    notes: list[str] = []

    for file_patch in patches:
        if file_patch.old_path == "/dev/null":
            source_content = ""
            source_path = file_patch.new_path
        else:
            source_path = file_patch.old_path
            if source_path not in result:
                return ApplyResult(
                    ok=False,
                    files=files.copy(),
                    notes=[f"missing source file for patch: {source_path}"],
                )
            source_content = result[source_path]

        lines, trailing_newline = _split_lines(source_content)
        offset = 0

        for hunk in file_patch.hunks:
            cursor = hunk.old_start - 1 + offset
            if cursor < 0 or cursor > len(lines):
                return ApplyResult(
                    ok=False,
                    files=files.copy(),
                    notes=[f"hunk start out of range for {source_path}"],
                )

            for hunk_line in hunk.lines:
                prefix = hunk_line[0]
                payload = hunk_line[1:]

                if prefix == " ":
                    if cursor >= len(lines) or lines[cursor] != payload:
                        return ApplyResult(
                            ok=False,
                            files=files.copy(),
                            notes=[
                                f"context mismatch in {source_path} near line {cursor + 1}: "
                                f"expected {payload!r}"
                            ],
                        )
                    cursor += 1
                elif prefix == "-":
                    if cursor >= len(lines) or lines[cursor] != payload:
                        return ApplyResult(
                            ok=False,
                            files=files.copy(),
                            notes=[
                                f"delete mismatch in {source_path} near line {cursor + 1}: "
                                f"expected {payload!r}"
                            ],
                        )
                    lines.pop(cursor)
                    offset -= 1
                elif prefix == "+":
                    lines.insert(cursor, payload)
                    cursor += 1
                    offset += 1
                else:
                    return ApplyResult(
                        ok=False,
                        files=files.copy(),
                        notes=[f"invalid hunk prefix {prefix!r} in {source_path}"],
                    )

        target_path = file_patch.new_path if file_patch.new_path != "/dev/null" else file_patch.old_path
        if file_patch.new_path == "/dev/null":
            if file_patch.old_path in result:
                del result[file_patch.old_path]
                notes.append(f"deleted file: {file_patch.old_path}")
            continue

        result[target_path] = _join_lines(lines, trailing_newline)
        if file_patch.old_path != file_patch.new_path and file_patch.old_path in result:
            del result[file_patch.old_path]
            notes.append(f"renamed file: {file_patch.old_path} -> {file_patch.new_path}")

    return ApplyResult(ok=True, files=result, notes=notes)


def generate_unified_diff(path: str, original: str, updated: str) -> str:
    old_lines = original.splitlines(keepends=True)
    new_lines = updated.splitlines(keepends=True)
    diff_iter: Iterable[str] = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff_iter).strip() + "\n"
