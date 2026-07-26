"""The SSTVAE desktop application (PySide6).

Only `settings` is importable without Qt installed; everything else in
this package pulls in PySide6.
"""

__all__ = ["main"]


def main() -> int:
    from .app import main as _main

    return _main()
