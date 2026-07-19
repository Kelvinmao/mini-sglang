"""Serializable records for the scheduler + KV-cache visualizer trace.

A trace is one JSON document: a :class:`TraceMeta` header plus a list of
:class:`StepSnapshot`. Each snapshot captures one scheduler iteration as a
*hybrid* record: the resulting state (batch composition, radix-tree structure,
page/token accounting) together with the ordered list of cache events that
produced it. The replay viewer draws the snapshot and animates the events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# One cache event is a small JSON object tagged by ``kind``. Kept as a plain
# dict so the engine hooks can emit lightweight payloads without importing this
# module; see ``tracer`` for the emit helpers that build them.
CacheEvent = Dict[str, Any]


@dataclass
class BatchEntry:
    """
    One request as it appeared in a scheduled forward batch.

    Lengths are snapshotted before the forward pass mutates them, so
    ``extend_len`` reflects the tokens this step actually computed.
    """

    uid: int
    table_idx: int
    phase: str  # "prefill" | "decode" | "chunked"
    cached_len: int
    device_len: int
    extend_len: int


@dataclass
class RadixNodeDump:
    """
    One compressed edge of the radix prefix tree at snapshot time.
    """

    uuid: int
    parent: Optional[int]
    length: int
    ref_count: int
    protected: bool
    tokens: List[int]  # first few token ids on the edge, for display


@dataclass
class StepSnapshot:
    """
    One scheduler iteration: resulting state plus the events that produced it.
    """

    index: int
    phase: str
    description: str
    batch: List[BatchEntry]
    tree: List[RadixNodeDump]
    evictable_size: int
    protected_size: int
    used_pages: int
    free_pages: int
    num_pages: int
    page_size: int
    events: List[CacheEvent] = field(default_factory=list)


@dataclass
class TraceMeta:
    """
    Header describing the run that produced a trace.
    """

    model: str
    page_size: int
    num_pages: int
    cache_type: str
    overlap_disabled: bool
    created_at: str
    version: int = 1
