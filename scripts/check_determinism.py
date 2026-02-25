"""CI check that runtime outputs are deterministic for fixed inputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer.schemas import TaskRequest
from infer.service import HeuristicModelAdapter, TaskService



def main() -> None:
    service = TaskService(adapter=HeuristicModelAdapter())
    payload = {
        "task_type": "patch_edit",
        "instruction": "change button color to emerald-500",
        "files": [{"path": "src/Hero.tsx", "content": "export default function Hero(){return <button className='bg-blue-500'>Hi</button>}"}],
        "constraints": {"framework": "react-tailwind", "output": "unified_diff"},
    }
    request = TaskRequest.from_dict(payload)
    first = service.handle(request)
    second = service.handle(request)

    assert first.patch == second.patch
    assert first.status == second.status
    assert first.validation.apply_ok == second.validation.apply_ok


if __name__ == "__main__":
    main()
