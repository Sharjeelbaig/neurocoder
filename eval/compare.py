"""Baseline comparison utilities for narrow-task parity reporting."""

from __future__ import annotations

from dataclasses import dataclass

from eval.benchmark import Scorecard


@dataclass(slots=True)
class ParityReport:
    candidate_weighted: float
    baseline_weighted: float
    parity_ratio: float
    pass_threshold: float
    passed: bool



def _weighted(score: Scorecard) -> float:
    # Weighted blend aligned with project acceptance gates.
    return (
        0.40 * score.success_rate
        + 0.25 * score.apply_rate
        + 0.20 * score.build_rate
        + 0.15 * score.lint_rate
    )


def compare_to_baseline(
    candidate: Scorecard,
    baseline: Scorecard,
    pass_threshold: float = 0.95,
) -> ParityReport:
    candidate_weighted = _weighted(candidate)
    baseline_weighted = _weighted(baseline)
    parity_ratio = candidate_weighted / baseline_weighted if baseline_weighted > 0 else 0.0
    return ParityReport(
        candidate_weighted=candidate_weighted,
        baseline_weighted=baseline_weighted,
        parity_ratio=parity_ratio,
        pass_threshold=pass_threshold,
        passed=parity_ratio >= pass_threshold,
    )
