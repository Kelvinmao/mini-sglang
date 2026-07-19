"""Env-gated recorder that captures scheduler + KV-cache activity to a trace.

The tracer is a process-global singleton. When ``MINISGL_TRACE_SCHEDULER_PATH``
is empty it is disabled and every hook is a single boolean check that returns
immediately, so the scheduler hot loop is unaffected. When enabled, the
scheduler calls :meth:`set_batch` right after a batch is scheduled and
:meth:`commit_step` at the end of the iteration, while the cache and radix tree
call the ``emit_*`` helpers as they mutate. :meth:`dump` writes the collected
trace as a single JSON document for the replay viewer.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from minisgl.env import ENV

from .schema import BatchEntry, CacheEvent, RadixNodeDump, StepSnapshot, TraceMeta

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.scheduler.cache import CacheManager


class Tracer:
    """
    Collect per-iteration scheduler/KV-cache snapshots in memory.
    """

    def __init__(self, path: str) -> None:
        """
        Create a tracer, enabled only when ``path`` is a non-empty string.

        Args:
            path: Destination trace file; empty disables all recording.
        """

        self._path = path
        self._enabled = bool(path)
        self._meta: Optional[TraceMeta] = None
        self._steps: List[StepSnapshot] = []
        self._events: List[CacheEvent] = []
        self._batch: Optional[List[BatchEntry]] = None
        self._batch_phase: str = ""
        self._index = 0

    @property
    def enabled(self) -> bool:
        """
        Return whether recording is active.
        """

        return self._enabled

    @property
    def path(self) -> str:
        """
        Return the configured output path.
        """

        return self._path

    @property
    def steps(self) -> List[StepSnapshot]:
        """
        Return the recorded steps (mainly for tests).
        """

        return self._steps

    def set_meta(
        self,
        *,
        model: str,
        page_size: int,
        num_pages: int,
        cache_type: str,
        overlap_disabled: bool,
    ) -> None:
        """
        Record the run-level header once at scheduler startup.
        """

        if not self._enabled:
            return
        self._meta = TraceMeta(
            model=model,
            page_size=page_size,
            num_pages=num_pages,
            cache_type=cache_type,
            overlap_disabled=overlap_disabled,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ---- cache event emit helpers (called from cache.py / radix_cache.py) ----

    def emit_match(self, *, input_len: int, matched_len: int, node: int) -> None:
        """
        Record a prefix-cache lookup and how much of it was reused.
        """

        if not self._enabled:
            return
        self._events.append(
            {"kind": "match", "input_len": input_len, "matched_len": matched_len, "node": node}
        )

    def emit_insert(self, *, already_cached: int, insert_len: int, node: int, edge_len: int) -> None:
        """
        Record a prefix insertion and the size of the new tree edge, if any.
        """

        if not self._enabled:
            return
        self._events.append(
            {
                "kind": "insert",
                "already_cached": already_cached,
                "insert_len": insert_len,
                "node": node,
                "edge_len": edge_len,
            }
        )

    def emit_evict(self, *, requested: int, evicted_len: int, nodes: List[int]) -> None:
        """
        Record an eviction pass and which leaf nodes it removed.
        """

        if not self._enabled:
            return
        self._events.append(
            {"kind": "evict", "requested": requested, "evicted_len": evicted_len, "nodes": nodes}
        )

    def emit_alloc(self, *, pages: int) -> None:
        """
        Record a page allocation for the uncached region of a batch.
        """

        if not self._enabled:
            return
        self._events.append({"kind": "alloc", "pages": pages})

    # ---- step lifecycle (called from the scheduler loop) ----

    def set_batch(self, batch: Batch) -> None:
        """
        Snapshot batch composition before the forward pass mutates lengths.
        """

        if not self._enabled:
            return
        entries: List[BatchEntry] = []
        for req in batch.reqs:
            phase = "chunked" if type(req).__name__ == "ChunkedReq" else batch.phase
            entries.append(
                BatchEntry(
                    uid=req.uid,
                    table_idx=req.table_idx,
                    phase=phase,
                    cached_len=req.cached_len,
                    device_len=req.device_len,
                    extend_len=req.extend_len,
                )
            )
        self._batch = entries
        self._batch_phase = batch.phase

    def commit_step(self, cache_manager: CacheManager, description: str = "") -> None:
        """
        Finalize the current iteration into a snapshot and reset buffers.

        When no batch ran this iteration, any stray events are dropped so they
        do not leak into the next step.
        """

        if not self._enabled:
            return
        if self._batch is None:
            self._events = []
            return

        tree = _dump_tree(cache_manager)
        size_info = cache_manager.prefix_cache.size_info
        free_pages = len(cache_manager.free_slots)
        snapshot = StepSnapshot(
            index=self._index,
            phase=self._batch_phase,
            description=description or _describe(self._batch_phase, self._batch, self._events),
            batch=self._batch,
            tree=tree,
            evictable_size=size_info.evictable_size,
            protected_size=size_info.protected_size,
            used_pages=cache_manager.num_pages - free_pages,
            free_pages=free_pages,
            num_pages=cache_manager.num_pages,
            page_size=cache_manager.page_size,
            events=self._events,
        )
        self._steps.append(snapshot)
        self._index += 1
        self._events = []
        self._batch = None

    def dump(self, path: Optional[str] = None) -> str:
        """
        Write the collected trace to disk as a single JSON document.

        Args:
            path: Override for the configured output path.

        Returns:
            The path the trace was written to.
        """

        import json
        import os

        out_path = path or self._path
        if not out_path:
            raise ValueError("Tracer has no output path configured")
        document = {
            "meta": asdict(self._meta) if self._meta is not None else None,
            "steps": [asdict(step) for step in self._steps],
        }
        parent = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
        return out_path


def _dump_tree(cache_manager: CacheManager) -> List[RadixNodeDump]:
    """
    Read the radix-tree structure from the cache manager when available.
    """

    dump_fn = getattr(cache_manager.prefix_cache, "dump_tree", None)
    if dump_fn is None:
        return []
    return dump_fn()


def _describe(phase: str, batch: List[BatchEntry], events: List[CacheEvent]) -> str:
    """
    Build a one-line human summary of what happened this step.
    """

    tokens = sum(entry.extend_len for entry in batch)
    parts = [f"{phase}: {len(batch)} req(s), {tokens} token(s)"]
    matched = sum(e["matched_len"] for e in events if e["kind"] == "match")
    if matched:
        parts.append(f"reused {matched}")
    evicted = sum(e["evicted_len"] for e in events if e["kind"] == "evict")
    if evicted:
        parts.append(f"evicted {evicted}")
    return "; ".join(parts)


_TRACER: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """
    Return the process-global tracer, creating it from the env on first use.
    """

    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer(str(ENV.TRACE_SCHEDULER_PATH.value))
    return _TRACER


def reset_tracer(path: Optional[str] = None) -> Tracer:
    """
    Replace the global tracer, mainly for tests and explicit re-configuration.

    Args:
        path: Output path for the new tracer; empty/None disables it.

    Returns:
        The freshly created tracer.
    """

    global _TRACER
    _TRACER = Tracer(path or "")
    return _TRACER
