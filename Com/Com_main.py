"""Compatibility entry wrapper for the GUI application."""

from app.window import ComApp, main

__all__ = ["ComApp", "main"]


if __name__ == "__main__":
    main()
