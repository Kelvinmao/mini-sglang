"""Physical multi-head-attention KV-cache pool."""

from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Layer-major KV-cache storage for dense attention models.

    The backing tensor is shaped as ``(K/V, layer, page, offset, local_kv_head,
    head_dim)``. Tensor-parallel ranks store only their local KV heads unless
    the model has fewer KV heads than ranks, in which case heads are replicated.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Allocate the physical KV-cache buffer.
        """

        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """
        Return the key-cache slice for one layer.
        """

        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """
        Return the value-cache slice for one layer.
        """

        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        Store freshly computed K/V rows into physical cache locations.
        """

        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        """
        Return the CUDA device that owns the cache.
        """

        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the cache tensor dtype.
        """

        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        """
        Return the number of model layers stored in the cache.
        """

        return self._num_layers
