"""Lightweight event bus wrapper for UI queue compatibility."""

import queue


class EventBus:
    def __init__(self, q: queue.Queue[tuple[str, object]]):
        self._q = q

    def publish(self, tag: str, payload: object) -> None:
        self._q.put((tag, payload))

    def get_nowait(self) -> tuple[str, object]:
        return self._q.get_nowait()
