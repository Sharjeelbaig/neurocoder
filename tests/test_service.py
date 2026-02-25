from __future__ import annotations

import unittest

from infer.schemas import TaskRequest
from infer.service import HeuristicModelAdapter, TaskService


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TaskService(adapter=HeuristicModelAdapter())

    def test_patch_edit_returns_diff(self) -> None:
        request = TaskRequest.from_dict(
            {
                "task_type": "patch_edit",
                "instruction": "change button color to emerald-500",
                "files": [
                    {
                        "path": "src/Hero.tsx",
                        "content": "export default function Hero(){return <button className='bg-blue-500 text-white'>Hi</button>}",
                    }
                ],
                "constraints": {"framework": "react-tailwind", "output": "unified_diff"},
            }
        )
        response = self.service.handle(request)
        self.assertIn(response.status, {"ok", "needs_retry"})
        self.assertTrue(response.patch.startswith("--- "))

    def test_page_generate_returns_files(self) -> None:
        request = TaskRequest.from_dict(
            {
                "task_type": "page_generate",
                "instruction": "Create SaaS landing page",
                "files": [],
                "constraints": {"framework": "react-tailwind", "output": "unified_diff"},
            }
        )
        response = self.service.handle(request)
        self.assertEqual(response.status, "ok")
        self.assertGreaterEqual(len(response.files), 2)


if __name__ == "__main__":
    unittest.main()
