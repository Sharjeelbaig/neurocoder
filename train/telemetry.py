"""Expert utilization telemetry and collapse alarms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExpertTelemetry:
    steps_below_threshold: int = 0
    collapse_events: int = 0

    def update(self, expert_load: list[float], threshold: float, patience: int) -> tuple[bool, str | None]:
        if not expert_load:
            return False, None
        min_load = min(expert_load)
        if min_load < threshold:
            self.steps_below_threshold += 1
        else:
            self.steps_below_threshold = 0

        if self.steps_below_threshold >= patience:
            self.collapse_events += 1
            self.steps_below_threshold = 0
            return True, f"expert collapse alarm: min_load={min_load:.6f}"

        return False, None
