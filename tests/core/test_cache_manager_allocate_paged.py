from __future__ import annotations

import torch

from minisgl.scheduler.cache import CacheManager


def assert_page_aligned(values: torch.Tensor, page_size: int) -> None:
    if len(values) > 0:
        assert torch.all(values % page_size == 0)


def test_allocate_paged_preserves_cached_prefix_entries(make_req) -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    page_table[0, :page_size] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert page_table[0, :page_size].tolist() == [100, 101, 102, 103]


def test_allocate_paged_writes_only_uncached_pages(make_req) -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    page_table[0, :page_size] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert page_table[0, page_size : page_size * 3].tolist() == list(range(8))
    assert torch.all(page_table[0, page_size * 3 :] == -1)


def test_allocate_paged_consumes_expected_number_of_pages(make_req) -> None:
    page_size = 4
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=4, device_len=10)

    manager.allocate_paged([req])

    assert len(manager.free_slots) == 6
    assert manager.free_slots.tolist() == [8, 12, 16, 20, 24, 28]
    assert_page_aligned(manager.free_slots, page_size)


def test_allocate_paged_handles_non_page_aligned_cached_len(make_req) -> None:
    page_size = 4
    page_table = torch.full((1, 20), -1, dtype=torch.int32)
    page_table[0, : page_size * 3] = torch.arange(100, 112, dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req = make_req(table_idx=0, cached_len=10, device_len=13)

    manager.allocate_paged([req])

    assert page_table[0, : page_size * 3].tolist() == list(range(100, 112))
    assert page_table[0, page_size * 3 : page_size * 4].tolist() == [0, 1, 2, 3]
    assert torch.all(page_table[0, page_size * 4 :] == -1)
    assert len(manager.free_slots) == 7


def test_allocate_paged_can_allocate_multiple_request_rows(make_req) -> None:
    page_size = 4
    page_table = torch.full((2, 16), -1, dtype=torch.int32)
    manager = CacheManager(num_pages=8, page_size=page_size, page_table=page_table, type="naive")
    req0 = make_req(table_idx=0, cached_len=0, device_len=3)
    req1 = make_req(table_idx=1, cached_len=4, device_len=9)

    manager.allocate_paged([req0, req1])

    assert page_table[0, :page_size].tolist() == [0, 1, 2, 3]
    assert page_table[1, page_size : page_size * 3].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert len(manager.free_slots) == 5
