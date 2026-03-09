"""Application runtime state container."""

from dataclasses import dataclass


@dataclass
class AppState:
    laser_state: bool = False
    preview_active: bool = False
