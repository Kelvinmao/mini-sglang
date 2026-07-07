"""No-op prefix cache used when radix reuse is disabled."""

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """
    Empty handle used when prefix reuse is disabled.
    """

    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        """
        Create a zero-length cache handle.
        """

        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        """
        Return no matched physical indices.
        """

        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    """
    Prefix-cache implementation that intentionally never stores prefixes.
    """

    def __init__(self, device: torch.device):
        """
        Initialize the empty cache on ``device``.
        """

        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        No-op because naive cache handles never own reusable pages.
        """

        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Return an empty match for every prompt.
        """

        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Ignore inserted prefixes and return an empty handle.
        """

        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        """
        Return no indices for zero-size eviction and reject real eviction.
        """

        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        """
        No-op reset for the empty cache.
        """

        pass

    @property
    def size_info(self) -> SizeInfo:
        """
        Return zero protected and evictable size.
        """

        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        """
        No-op integrity check for the empty cache.
        """

        pass
