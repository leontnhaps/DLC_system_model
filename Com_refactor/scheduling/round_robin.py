"""Round-robin scheduling scaffold."""

from scheduling.base import SchedulingAlgorithm


class RoundRobinScheduler(SchedulingAlgorithm):
    """Deterministic round-robin selector."""

    def select_next(self, candidates, state=None):
        items = list(candidates or [])
        if not items:
            return None, {"index": 0}

        index = 0
        if isinstance(state, dict):
            try:
                index = int(state.get("index", 0))
            except Exception:
                index = 0

        index %= len(items)
        selected = items[index]
        next_state = {"index": (index + 1) % len(items)}
        return selected, next_state


__all__ = ["RoundRobinScheduler"]
