"""
Tests for the scheduler/KV-cache Tracer that backs the visualizer.

These run on CPU with the radix prefix cache and do not require a GPU. They
cover schema-valid recording, the zero-overhead disabled path, event capture for
match/insert/evict, and JSON dumping.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

import minisgl.core as core
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.scheduler.cache import CacheManager
from minisgl.trace import get_tracer, reset_tracer


class _DummyHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global context and tracer around each test."""
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx
    reset_tracer("")  # leave tracing disabled for other tests


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    page_table = torch.empty((1,))
    core.set_global_ctx(core.Context(page_size=page_size))
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _make_req(uid: int, device_len: int) -> Req:
    return Req(
        input_ids=torch.arange(device_len, dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=1,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=_DummyHandle(cached_len=0),
    )


def test_records_schema_valid_step(tmp_path: Path):
    tracer = reset_tracer(str(tmp_path / "trace.json"))
    assert tracer.enabled

    cm = _make_cache_manager(num_pages=64, page_size=1)
    ids = torch.arange(16, dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, ids)  # emits an insert event
    cm.prefix_cache.match_prefix(ids)  # emits a match event

    tracer.set_batch(Batch(reqs=[_make_req(0, device_len=16)], phase="prefill"))
    tracer.commit_step(cm)

    assert len(tracer.steps) == 1
    step = tracer.steps[0]
    assert step.index == 0
    assert step.phase == "prefill"
    assert step.page_size == 1
    assert step.num_pages == 64
    assert step.batch[0].uid == 0
    assert step.batch[0].extend_len == 16
    assert any(e["kind"] == "insert" for e in step.events)
    assert any(e["kind"] == "match" for e in step.events)
    assert step.tree, "tree dump should include at least the root node"

    # The whole snapshot must be JSON-serializable for the viewer.
    json.dumps(dataclasses.asdict(step))


def test_disabled_tracer_records_nothing():
    tracer = reset_tracer("")  # empty path disables tracing
    assert not tracer.enabled

    cm = _make_cache_manager(num_pages=64, page_size=1)
    ids = torch.arange(16, dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, ids)
    cm.prefix_cache.match_prefix(ids)

    tracer.set_batch(Batch(reqs=[_make_req(0, device_len=16)], phase="prefill"))
    tracer.commit_step(cm)

    assert tracer.steps == []
    # No events should have been buffered while disabled.
    assert get_tracer()._events == []


def test_evict_event_recorded(tmp_path: Path):
    tracer = reset_tracer(str(tmp_path / "trace.json"))
    cm = _make_cache_manager(num_pages=4, page_size=1)

    cm._allocate(4)  # exhaust free pages
    ids = torch.arange(2, dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, ids)  # make 2 pages evictable
    cm._allocate(1)  # forces eviction -> emits an evict event

    tracer.set_batch(Batch(reqs=[_make_req(0, device_len=2)], phase="decode"))
    tracer.commit_step(cm)

    evicts = [e for e in tracer.steps[0].events if e["kind"] == "evict"]
    assert evicts and evicts[0]["evicted_len"] >= 1


def test_commit_without_batch_is_dropped(tmp_path: Path):
    tracer = reset_tracer(str(tmp_path / "trace.json"))
    cm = _make_cache_manager(num_pages=64, page_size=1)
    ids = torch.arange(16, dtype=torch.int32)
    cm.prefix_cache.match_prefix(ids)  # a stray event, no batch scheduled

    tracer.commit_step(cm)

    assert tracer.steps == []
    assert tracer._events == []  # stray events cleared, not leaked to next step


def test_dump_writes_single_json(tmp_path: Path):
    path = tmp_path / "trace.json"
    tracer = reset_tracer(str(path))
    tracer.set_meta(
        model="test-model", page_size=1, num_pages=64, cache_type="radix", overlap_disabled=True
    )

    cm = _make_cache_manager(num_pages=64, page_size=1)
    ids = torch.arange(16, dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, ids)
    tracer.set_batch(Batch(reqs=[_make_req(0, device_len=16)], phase="prefill"))
    tracer.commit_step(cm)

    out = tracer.dump()
    data = json.loads(Path(out).read_text())
    assert data["meta"]["model"] == "test-model"
    assert data["meta"]["overlap_disabled"] is True
    assert len(data["steps"]) == 1
    assert data["steps"][0]["index"] == 0
