"""Scheduler + KV-cache tracing for the learning visualizer.

Enable by setting ``MINISGL_TRACE_SCHEDULER_PATH`` to an output file and running
with overlap scheduling disabled (``MINISGL_DISABLE_OVERLAP_SCHEDULING=1``) so
each recorded step is one clean schedule -> forward -> commit unit.
"""

from __future__ import annotations

from .schema import BatchEntry, CacheEvent, RadixNodeDump, StepSnapshot, TraceMeta
from .tracer import Tracer, get_tracer, reset_tracer

__all__ = [
    "BatchEntry",
    "CacheEvent",
    "RadixNodeDump",
    "StepSnapshot",
    "TraceMeta",
    "Tracer",
    "get_tracer",
    "reset_tracer",
]
