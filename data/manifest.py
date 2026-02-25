"""Helpers to read/write dataset manifests and provenance metadata."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class ManifestRecord:
    source_repo: str
    relative_path: str
    sha256: str
    bytes: int
    spdx: str
    react_like: bool
    tailwind_like: bool
    domain_match: bool
    ingested_at: str



def load_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        records.append(ManifestRecord(**payload))
    return records


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")
