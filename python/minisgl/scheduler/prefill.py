"""Prefill admission and chunking policy.

Pending prompts enter the scheduler through this module. The manager tries to
reuse cached prefixes, reserves enough KV-cache capacity for the request's
future decode tokens, and splits long prompts into multiple prefill chunks when
the per-iteration token budget would otherwise be exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """
    Request fragment representing an unfinished prefill chunk.

    Chunked requests are real forward inputs but are not sampled. After each
    chunk, the corresponding ``PendingReq`` keeps this object so the next
    scheduler iteration can continue from the same table slot and cache handle.
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        Reject host-token append because chunked prefill does not sample.
        """

        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        """
        Prevent partial prefill chunks from entering the decode manager.
        """

        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """
    Stateful helper that greedily admits pending requests into one prefill batch.

    ``token_budget`` limits prompt extension work for this iteration, while
    ``reserved_size`` accounts for decode tokens already promised to running
    requests. Both are updated as requests are admitted.
    """

    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """
        Reserve table and cache resources for a new pending request.

        Args:
            req: Pending request without an assigned table slot.

        Returns:
            Locked prefix-cache handle and allocated table index, or ``None`` if
            admission would exceed current capacity.
        """

        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        # Capacity can change when locking moves matched pages from evictable to
        # protected. Re-check before consuming a request table slot.
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # Cached prefix tokens and physical KV locations are copied into the
            # request slot up front. The scheduler allocates only the uncached
            # tail immediately before running the batch.
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """
        Copy the next prompt chunk into the request slot and build a ``Req``.

        Args:
            pending_req: Original request state.
            cache_handle: Locked prefix-cache handle for the cached prefix.
            table_idx: Allocated table slot.
            cached_len: Prefix length already backed by KV-cache locations.

        Returns:
            ``Req`` for a complete prefill or ``ChunkedReq`` for a partial chunk.
        """

        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """
        Try to admit one pending request or its next chunk into this batch.

        Args:
            pending_req: Request at the head of the pending queue.

        Returns:
            Request object ready for prefill, or ``None`` if admission failed.
        """

        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """
    Queue manager for requests that still need prompt prefill.
    """

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """
        Add a user request to the pending prefill queue.

        Args:
            req: Tokenized request from the tokenizer process.
        """

        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """
        Build the next prefill batch subject to token and cache budgets.

        Args:
            prefill_budget: Maximum uncached prompt tokens to process now.

        Returns:
            Prefill batch, or ``None`` when no pending request can run.
        """

        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """
        Remove a pending request and return any partially allocated chunk.

        Args:
            uid: User request id.

        Returns:
            Chunked request whose resources must be freed, if one existed.
        """

        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """
        Return whether at least one request is waiting for prefill.
        """

        return len(self.pending_list) > 0
