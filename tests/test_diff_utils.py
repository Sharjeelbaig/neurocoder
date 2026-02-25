from __future__ import annotations

import unittest

from infer.diff_utils import apply_unified_diff, parse_unified_diff, validate_unified_diff


PATCH = """--- a/src/Hero.tsx
+++ b/src/Hero.tsx
@@ -1 +1 @@
-export default function Hero(){return <button className='bg-blue-500'>Hi</button>}
+export default function Hero(){return <button className='bg-emerald-500'>Hi</button>}
"""


class DiffUtilsTests(unittest.TestCase):
    def test_validate_and_parse(self) -> None:
        ok, notes = validate_unified_diff(PATCH)
        self.assertTrue(ok)
        self.assertEqual(notes, [])
        parsed = parse_unified_diff(PATCH)
        self.assertEqual(len(parsed), 1)

    def test_apply_patch(self) -> None:
        files = {
            "src/Hero.tsx": "export default function Hero(){return <button className='bg-blue-500'>Hi</button>}"
        }
        result = apply_unified_diff(files, PATCH)
        self.assertTrue(result.ok)
        self.assertIn("bg-emerald-500", result.files["src/Hero.tsx"])

    def test_invalid_patch(self) -> None:
        ok, notes = validate_unified_diff("not a patch")
        self.assertFalse(ok)
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
