"""Small UI-thread scheduling helpers (stdlib-only)."""


def call_on_ui_thread(root, fn, *args, **kwargs):
    """Schedule ``fn(*args, **kwargs)`` on the UI thread via ``root.after``."""
    return root.after(0, lambda: fn(*args, **kwargs))


__all__ = ["call_on_ui_thread"]
