"""Decode queue management for requests with initialized KV cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """
    Tracks active requests that can generate one additional token per iteration.
    """

    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """
        Merge newly forwarded requests into the decode set and drop completed ones.

        Args:
            reqs: Requests whose forward pass just completed.
        """

        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """
        Remove a request from the decode set.

        Args:
            req: Request that finished or was aborted.
        """

        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """
        Abort a running decode request by user id.

        Args:
            uid: User request id.

        Returns:
            Removed request whose resources must be freed, if found.
        """

        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """
        Estimate KV slots reserved by requests currently decoding.
        """

        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """
        Return a stable-order decode batch for all currently runnable requests.
        """

        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """
        Return whether at least one request can decode.
        """

        return len(self.running_reqs) > 0
