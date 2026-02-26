"""Run deterministic inference checks for NeuroCoder CLI outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Pattern


@dataclass(slots=True)
class Case:
    name: str
    prompt: str
    must_contain: tuple[str, ...] = ()
    must_match: tuple[Pattern[str], ...] = ()


CASES: tuple[Case, ...] = (
    Case(
        name="greeting_hi",
        prompt="hi",
        must_contain=("Hello! I am NeuroCoder",),
    ),
    Case(
        name="greeting_how_are_you",
        prompt="how are you?",
        must_contain=("I am doing well",),
    ),
    Case(
        name="python_reverse_string",
        prompt="Write a python function to reverse a string",
        must_contain=("def reverse_string", "return value[::-1]"),
    ),
    Case(
        name="landing_page_default",
        prompt="Generate a landing page for marketing agency",
        must_contain=("<!DOCTYPE html>", "<title>GrowthSprint Landing</title>", "Velocity"),
    ),
    Case(
        name="landing_page_custom_title",
        prompt='Generate a landing page for marketing agency, but title should be "Velocity Landing"',
        must_contain=("<title>Velocity Landing</title>",),
    ),
    Case(
        name="patch_edit_blue",
        prompt="Provide a unified diff to change hero button color to blue-500",
        must_contain=("--- a/src/components/Hero.tsx", "+++ b/src/components/Hero.tsx", "bg-blue-500"),
    ),
    Case(
        name="reasoning_linear_equation",
        prompt="Solve 1148583*a = 1148360*a - 5352",
        must_contain=("<thinking>", "<answer>-24</answer>"),
    ),
    Case(
        name="reasoning_bus_cost",
        prompt="A school has 252 students and 8 teachers going on a trip. They hire 41-seater buses. Rental is 300000 and toll is 7500 per bus. Give final cost only in tags.",
        must_contain=("<answer>2152500</answer>",),
    ),
    Case(
        name="reasoning_entailment",
        prompt="If Melisandre did not agree with Thorn and Thorn thought the same thing as Mance, did Thorn agree with Mance?",
        must_contain=("<answer>no</answer>",),
    ),
    Case(
        name="translation_persian",
        prompt="Translate to Persian: So she was again in Mathare, without income, without skills and without money.",
        must_match=(re.compile(r"[\u0600-\u06FF]"),),
    ),
    Case(
        name="arithmetic_thinking",
        prompt="think step by step and solve 17 * 8 + 3",
        must_contain=("<thinking>", "<answer>139</answer>"),
    ),
)


def run_prompt(model_dir: Path, prompt: str, disable_fallback: bool) -> str:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "infer_neurocoder.py"),
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "260",
    ]
    if disable_fallback:
        cmd.append("--disable-fallback")
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"inference failed for prompt: {prompt}")
    return proc.stdout.strip()


def evaluate_case(case: Case, output: str) -> list[str]:
    errors: list[str] = []
    for needle in case.must_contain:
        if needle not in output:
            errors.append(f"missing substring: {needle}")
    for pattern in case.must_match:
        if pattern.search(output) is None:
            errors.append(f"regex mismatch: {pattern.pattern}")
    return errors


def run_suite(model_dir: Path, disable_fallback: bool) -> dict[str, object]:
    mode = "disable_fallback" if disable_fallback else "with_fallback"
    rows: list[dict[str, object]] = []
    passed = 0
    for case in CASES:
        output = run_prompt(model_dir, case.prompt, disable_fallback=disable_fallback)
        errors = evaluate_case(case, output)
        ok = not errors
        if ok:
            passed += 1
        rows.append(
            {
                "name": case.name,
                "ok": ok,
                "errors": errors,
                "output_preview": output[:280],
            }
        )
    total = len(CASES)
    return {
        "mode": mode,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 1.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroCoder inference regression suite")
    parser.add_argument("--model-dir", default="artifacts/release_hfstd_v1/hf")
    parser.add_argument("--out", default="benchmarks/results/inference_suite_latest.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model_dir": str(model_dir),
        "runs": [
            run_suite(model_dir, disable_fallback=True),
            run_suite(model_dir, disable_fallback=False),
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    for run in report["runs"]:
        if run["passed"] != run["total"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
