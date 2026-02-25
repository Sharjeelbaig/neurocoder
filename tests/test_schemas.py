from __future__ import annotations

import unittest

from infer.schemas import SchemaValidationError, TaskRequest, TaskResponse, ValidationResult


class TaskSchemaTests(unittest.TestCase):
    def test_valid_task_request(self) -> None:
        payload = {
            "task_type": "patch_edit",
            "instruction": "change color",
            "files": [{"path": "src/Hero.tsx", "content": "const x = 1;"}],
            "constraints": {"framework": "react-tailwind", "output": "unified_diff"},
        }
        req = TaskRequest.from_dict(payload)
        self.assertEqual(req.task_type, "patch_edit")
        self.assertEqual(req.files[0].path, "src/Hero.tsx")

    def test_invalid_framework(self) -> None:
        payload = {
            "task_type": "patch_edit",
            "instruction": "change color",
            "files": [{"path": "src/Hero.tsx", "content": "const x = 1;"}],
            "constraints": {"framework": "nextjs", "output": "unified_diff"},
        }
        with self.assertRaises(SchemaValidationError):
            TaskRequest.from_dict(payload)

    def test_response_to_dict(self) -> None:
        response = TaskResponse.create(
            status="ok",
            patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
            validation=ValidationResult(True, True, True, []),
        )
        payload = response.to_dict()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("validation", payload)


if __name__ == "__main__":
    unittest.main()
