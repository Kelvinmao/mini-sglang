from __future__ import annotations

import torch

from minisgl.core import Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.scheduler.cache import CacheManager


# BOT_TODO [should][maintainability]: This helper duplicates the request/cache-handle scaffold in test_decode_manager.py; move the shared test Req factory into tests/core/conftest.py or a small tests/core helper module so future scheduler tests do not drift in how they construct Req state.
class DummyCacheHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


def make_req(
    *,
    table_idx: int,
    cached_len: int,
    device_len: int,
    output_len: int = 1,
) -> Req:
    return Req(
        input_ids=torch.arange(device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=DummyCacheHandle(cached_len=cached_len),
    )


def assert_page_aligned(values: torch.Tensor, page_size: int) -> None:
    if len(values) > 0:
        assert torch.all(values % page_size == 0)


def test_allocate_paged_preserves_cached_prefix_entries() -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    page_table[0, :page_size] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert page_table[0, :page_size].tolist() == [100, 101, 102, 103]


def test_allocate_paged_writes_only_uncached_pages() -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    page_table[0, :page_size] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert page_table[0, page_size : page_size * 3].tolist() == list(range(8))
    assert torch.all(page_table[0, page_size * 3 :] == -1)


def test_allocate_paged_consumes_expected_number_of_pages() -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert len(manager.free_slots) == 6
    assert manager.free_slots.tolist() == [8, 12, 16, 20, 24, 28]
    assert_page_aligned(manager.free_slots, page_size)


# BOT_TODO [should][tests]: Add a non-page-aligned cached_len case, e.g. decode-style cached_len=10/device_len=13 with page_size=4; the current tests only cover aligned cached_len boundaries, so they can miss regressions in the ceil(cached_len/page_size) allocation boundary used during decode growth.
def test_allocate_paged_can_allocate_multiple_request_rows() -> None:
    page_size = 4
    page_table = torch.full((2, 16), -1, dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req0 = make_req(table_idx=0, cached_len=0, device_len=3)
    req1 = make_req(table_idx=1, cached_len=4, device_len=9)

    manager.allocate_paged([req0, req1])

    assert page_table[0, :page_size].tolist() == [0, 1, 2, 3]
    assert page_table[1, page_size : page_size * 3].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert len(manager.free_slots) == 5
