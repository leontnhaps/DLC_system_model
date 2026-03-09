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


__all__ = ["SchedulingWorkflow"]
