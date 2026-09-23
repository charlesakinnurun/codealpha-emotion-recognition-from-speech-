"""Structured logging helpers.

A single console formatter is used everywhere so pipeline output stays
predictable and greppable. File-backed experiment logging lives in the
training module.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _logical_name(name: str) -> str:
    """Strip a redundant package prefix (``emotion_recognition.``) if present."""
    prefix = "emotion_recognition."
    return name[len(prefix) :] if name.startswith(prefix) else name


def get_logger(name: str) -> logging.Logger:
    """Return the project logger for ``name``, configured once.

    Logs are written to stderr so that freed stdout can be reserved for
    structured outputs (e.g. dataframes printed by CLI tools).
    """
    global _configured  # noqa: PLW0603 - module-level one-time setup
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger("emotion_recognition")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _configured = True
    return logging.getLogger(f"emotion_recognition.{_logical_name(name)}")


__all__ = ["get_logger"]
