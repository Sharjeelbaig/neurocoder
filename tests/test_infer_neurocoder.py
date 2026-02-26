from __future__ import annotations

import unittest

from scripts.infer_neurocoder import (
    build_landing_page_html,
    build_patch_from_prompt,
    solve_linear_equation,
    stable_recovery_response,
)


class InferNeuroCoderTests(unittest.TestCase):
    def test_linear_equation_sign(self) -> None:
        prompt = "Solve 1148583*a = 1148360*a - 5352"
        output = solve_linear_equation(prompt)
        self.assertIsNotNone(output)
        self.assertIn("<answer>-24</answer>", output)

    def test_landing_title_override(self) -> None:
        prompt = 'Generate a landing page for marketing agency, title should be "Velocity Landing"'
        html = build_landing_page_html(prompt)
        self.assertIn("<title>Velocity Landing</title>", html)

    def test_patch_color_request(self) -> None:
        diff = build_patch_from_prompt("change hero button color to blue-500 and return unified diff")
        self.assertIn("bg-blue-500", diff)
        self.assertIn("hover:bg-blue-600", diff)

    def test_greeting_stable_response(self) -> None:
        text = stable_recovery_response("hi")
        self.assertTrue(text.lower().startswith("hello"))


if __name__ == "__main__":
    unittest.main()
