"""Run permissive-license dataset ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from data.ingest import ingest_sources

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyMoE data ingestion")
    parser.add_argument("sources", nargs="+", help="Source repository paths")
    parser.add_argument("--out", default="datasets/snapshot_v1", help="Output directory")
    args = parser.parse_args()

    roots = [Path(path) for path in args.sources]
    summary = ingest_sources(roots, Path(args.out))
    print(summary)


if __name__ == "__main__":
    main()
