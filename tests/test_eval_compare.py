from __future__ import annotations

import unittest

from eval.benchmark import Scorecard
from eval.compare import compare_to_baseline


class EvalCompareTests(unittest.TestCase):
    def test_parity_ratio(self) -> None:
        candidate = Scorecard(total=10, success_rate=0.9, apply_rate=0.9, build_rate=0.85, lint_rate=0.95)
        baseline = Scorecard(total=10, success_rate=0.95, apply_rate=0.95, build_rate=0.9, lint_rate=0.95)
        report = compare_to_baseline(candidate, baseline)
        self.assertGreater(report.parity_ratio, 0.8)
        self.assertLess(report.parity_ratio, 1.1)


if __name__ == "__main__":
    unittest.main()
