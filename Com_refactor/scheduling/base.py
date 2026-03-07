"""Base interface for scheduling algorithms."""

from abc import ABC, abstractmethod


class SchedulingAlgorithm(ABC):
    """Minimal interface for selecting the next candidate."""

    @abstractmethod
    def select_next(self, candidates, state=None):
        """Return ``(selected_candidate, next_state)``."""
        raise NotImplementedError


__all__ = ["SchedulingAlgorithm"]
