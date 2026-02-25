"""CI check for request/response schema contracts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from infer.schemas import TaskRequest, TaskResponse, ValidationResult



def main() -> None:
    payload = {
        "task_type": "patch_edit",
        "instruction": "change button color to emerald-500",
        "files": [{"path": "src/Hero.tsx", "content": "export default function Hero(){return <button className='bg-blue-500'>Hi</button>}"}],
        "constraints": {"framework": "react-tailwind", "output": "unified_diff"},
    }
    request = TaskRequest.from_dict(payload)

    response = TaskResponse.create(
        status="ok",
        patch="--- a/src/Hero.tsx\n+++ b/src/Hero.tsx\n@@ -1 +1 @@\n-export default function Hero(){return <button className='bg-blue-500'>Hi</button>}\n+export default function Hero(){return <button className='bg-emerald-500'>Hi</button>}\n",
        validation=ValidationResult(apply_ok=True, lint_ok=True, build_ok=True, notes=[]),
    )

    assert request.to_dict()["task_type"] == "patch_edit"
    encoded = json.dumps(response.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["status"] == "ok"
    assert "validation" in decoded


if __name__ == "__main__":
    main()
