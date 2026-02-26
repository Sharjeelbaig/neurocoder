"""Run or print the full NeuroCoder vNext build session in batches.

This script is designed for long coding/modeling sessions:
- ground-up dataset build,
- optional external blend dataset build,
- from-scratch SFT training,
- alignment pass,
- packaging,
- inference regression checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _run(cmd: Sequence[str], cwd: Path, execute: bool, log_file: Path) -> int:
    rendered = " ".join(cmd)
    print(f"$ {rendered}")
    if not execute:
        return 0

    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {rendered}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=fh, stderr=fh, text=True)
    return proc.returncode


def _ensure_merged_jsonl(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroCoder vNext execution session")
    parser.add_argument("--execute", action="store_true", help="Run commands. Default prints only.")
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    py = sys.executable

    profile = {
        "fast": {
            "landing": 1200,
            "patch": 900,
            "reason": 300,
            "train_steps": 300,
            "align_steps": 220,
            "batch_size": 6,
        },
        "full": {
            "landing": 8000,
            "patch": 6000,
            "reason": 2000,
            "train_steps": 1200,
            "align_steps": 800,
            "batch_size": 8,
        },
    }[args.profile]

    out_root = repo / "artifacts" / f"vnext_{args.profile}"
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "session.log"

    groundup_jsonl = repo / "datasets" / "groundup" / "neurocoder_v3.jsonl"
    groundup_txt = repo / "datasets" / "groundup" / "neurocoder_v3.txt"
    merged_jsonl = repo / "datasets" / "curriculum" / f"vnext_{args.profile}_merged.jsonl"
    trained_dir = repo / "artifacts" / f"trained_vnext_{args.profile}"
    release_dir = repo / "artifacts" / f"release_vnext_{args.profile}"
    suite_report = repo / "benchmarks" / "results" / f"inference_suite_vnext_{args.profile}.json"

    commands: list[list[str]] = [
        [
            py,
            str(repo / "scripts" / "build_groundup_dataset_v3.py"),
            "--landing-count",
            str(profile["landing"]),
            "--patch-count",
            str(profile["patch"]),
            "--reasoning-count",
            str(profile["reason"]),
            "--seed",
            str(args.seed),
        ],
        [
            py,
            str(repo / "scripts" / "build_hf_mix_dataset.py"),
            "--out",
            str(repo / "datasets" / "curriculum" / f"hf_mix_{args.profile}.txt"),
            "--manifest",
            str(repo / "datasets" / "curriculum" / f"hf_mix_{args.profile}_manifest.json"),
            "--per-dataset",
            "300" if args.profile == "fast" else "1000",
            "--max-chars",
            "1800",
            "--seed",
            str(args.seed),
        ],
    ]

    for idx, cmd in enumerate(commands, start=1):
        rc = _run(cmd, cwd=repo, execute=args.execute, log_file=logs)
        if rc != 0:
            raise SystemExit(f"step {idx} failed with exit={rc}. Check {logs}")

    if args.execute:
        _ensure_merged_jsonl(
            paths=[
                groundup_jsonl,
                repo / "datasets" / "curriculum" / "sft_v2.jsonl",
            ],
            out_path=merged_jsonl,
        )
    else:
        print(f"$ merge-jsonl {groundup_jsonl} + datasets/curriculum/sft_v2.jsonl -> {merged_jsonl}")

    post_merge: list[list[str]] = [
        [
            py,
            str(repo / "scripts" / "train_sft_model.py"),
            "--dataset",
            str(merged_jsonl),
            "--out-dir",
            str(trained_dir),
            "--vocab-size",
            "12000",
            "--seq-len",
            "640",
            "--hidden-size",
            "384",
            "--num-layers",
            "12",
            "--num-heads",
            "8",
            "--num-experts",
            "4",
            "--steps",
            str(profile["train_steps"]),
            "--batch-size",
            str(profile["batch_size"]),
            "--lr",
            "1.5e-4",
            "--seed",
            str(args.seed),
        ],
        [
            py,
            str(repo / "scripts" / "align_responses.py"),
            "--model-dir",
            str(trained_dir),
            "--dataset",
            str(groundup_txt),
            "--steps",
            str(profile["align_steps"]),
            "--batch-size",
            str(profile["batch_size"]),
            "--seq-len",
            "320",
            "--lr",
            "7e-5",
            "--seed",
            str(args.seed),
        ],
        [
            py,
            str(repo / "scripts" / "package_release.py"),
            "--tokenizer",
            str(trained_dir / "tokenizer.json"),
            "--weights",
            str(trained_dir / "model.safetensors"),
            "--model-config",
            str(trained_dir / "model_config.json"),
            "--out",
            str(release_dir),
            "--model-name",
            "neurocoder",
        ],
        [
            py,
            str(repo / "scripts" / "test_inference_suite.py"),
            "--model-dir",
            str(release_dir / "hf"),
            "--out",
            str(suite_report),
        ],
    ]

    for offset, cmd in enumerate(post_merge, start=len(commands) + 1):
        rc = _run(cmd, cwd=repo, execute=args.execute, log_file=logs)
        if rc != 0:
            raise SystemExit(f"step {offset} failed with exit={rc}. Check {logs}")

    summary = {
        "profile": args.profile,
        "execute": args.execute,
        "trained_dir": str(trained_dir),
        "release_dir": str(release_dir),
        "suite_report": str(suite_report),
        "log": str(logs),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
