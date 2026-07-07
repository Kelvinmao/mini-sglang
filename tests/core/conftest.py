from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from minisgl.core import Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle


class DummyCacheHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


@pytest.fixture
def make_req() -> Callable[..., Req]:
    def _make_req(
        uid: int | None = None,
        *,
        table_idx: int | None = None,
        cached_len: int = 0,
        device_len: int | None = None,
        output_len: int = 1,
        prompt_len: int = 1,
    ) -> Req:
        if table_idx is None:
            assert uid is not None
            table_idx = uid
        if uid is None:
            uid = table_idx
        if device_len is None:
            device_len = prompt_len

        return Req(
            input_ids=torch.arange(device_len, dtype=torch.int32),
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=output_len,
            uid=uid,
            sampling_params=SamplingParams(max_tokens=output_len),
            cache_handle=DummyCacheHandle(cached_len=cached_len),
        )

    return _make_req
