"""Minimal scheduling workflow scaffold."""

from scheduling.round_robin import RoundRobinScheduler


class SchedulingWorkflow:
    """Small delegating wrapper around a scheduling algorithm."""

    def __init__(self, scheduler=None):
        self.scheduler = scheduler or RoundRobinScheduler()
        self.context = {}

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return self.scheduler

    def set_context(self, **kwargs):
        self.context.update(kwargs)
        return self.context

    def choose_next(self, candidates, state=None):
        return self.scheduler.select_next(candidates, state=state)

    def order_target_ids(self, targets):
        """Return target ids ordered by pan from high to low in one direction."""
        items = list((targets or {}).items())
        items.sort(key=lambda item: (-float(item[1][0]), int(item[0])))
        return [track_id for track_id, _target in items]


__all__ = ["SchedulingWorkflow"]
