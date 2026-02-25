"""Run narrow benchmark suite against local service."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from eval.benchmark import BenchmarkSuite, LocalServiceRunner, write_results
from infer.service import HeuristicModelAdapter, TaskService

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyMoE benchmark")
    parser.add_argument("suite", help="Path to benchmark JSONL")
    parser.add_argument("--out", default="benchmarks/results/local")
    args = parser.parse_args()

    suite = BenchmarkSuite.from_jsonl(Path(args.suite))
    runner = LocalServiceRunner(TaskService(adapter=HeuristicModelAdapter()))
    results, scorecard = suite.run(runner)
    write_results(results, scorecard, Path(args.out))
    print(scorecard)


if __name__ == "__main__":
    main()
