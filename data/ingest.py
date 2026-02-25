"""Data ingestion pipeline with SPDX policy enforcement and provenance manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

from data.classifier import classify_react_tailwind, is_code_file
from data.licenses import detect_spdx, is_license_allowed


@dataclass(slots=True)
class FileRecord:
    source_repo: str
    relative_path: str
    sha256: str
    bytes: int
    spdx: str
    react_like: bool
    tailwind_like: bool
    domain_match: bool
    ingested_at: str


@dataclass(slots=True)
class IngestSummary:
    accepted_files: int
    skipped_duplicates: int
    skipped_oversized: int
    skipped_non_code: int
    skipped_license: int
    total_bytes: int
    repos_seen: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ingest_sources(
    source_roots: Iterable[Path],
    output_dir: Path,
    max_file_bytes: int = 256_000,
) -> IngestSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    license_audit_path = output_dir / "license_audit.json"

    seen_hashes: set[str] = set()
    records: list[FileRecord] = []
    license_audit: dict[str, dict[str, str | bool]] = {}

    skipped_duplicates = 0
    skipped_oversized = 0
    skipped_non_code = 0
    skipped_license = 0
    total_bytes = 0

    roots = sorted({Path(root).resolve() for root in source_roots})
    for repo_root in roots:
        spdx = detect_spdx(repo_root)
        allowed = is_license_allowed(spdx)
        license_audit[str(repo_root)] = {
            "spdx": spdx or "UNKNOWN",
            "allowed": allowed,
        }
        if not allowed:
            skipped_license += 1
            continue

        for path in sorted(repo_root.rglob("*")):
            if not path.is_file():
                continue
            if not is_code_file(path):
                skipped_non_code += 1
                continue

            content = path.read_text(encoding="utf-8", errors="ignore")
            file_bytes = len(content.encode("utf-8"))
            if file_bytes > max_file_bytes:
                skipped_oversized += 1
                continue

            digest = _sha256(content)
            if digest in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(digest)

            labels = classify_react_tailwind(content)
            rel_path = str(path.relative_to(repo_root))
            record = FileRecord(
                source_repo=str(repo_root),
                relative_path=rel_path,
                sha256=digest,
                bytes=file_bytes,
                spdx=spdx or "UNKNOWN",
                react_like=labels["react_like"],
                tailwind_like=labels["tailwind_like"],
                domain_match=labels["domain_match"],
                ingested_at=_utc_now(),
            )
            records.append(record)
            total_bytes += file_bytes

    with manifest_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    summary = IngestSummary(
        accepted_files=len(records),
        skipped_duplicates=skipped_duplicates,
        skipped_oversized=skipped_oversized,
        skipped_non_code=skipped_non_code,
        skipped_license=skipped_license,
        total_bytes=total_bytes,
        repos_seen=len(roots),
    )

    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    license_audit_path.write_text(json.dumps(license_audit, indent=2, sort_keys=True), encoding="utf-8")
    return summary
