"""Per-request table-slot allocation for scheduler-managed device tables."""

import torch


class TableManager:
    """
    Allocates per-request rows in the scheduler token and page tables.

    A table row is the stable handle used by kernels and schedulers while a
    request is live. Token ids and page-table entries are stored separately so
    prefill/decode scheduling can update them without mutating the request's CPU
    token history until a sampled token is committed.
    """

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        """
        Initialize table slots and the device token pool.

        Args:
            max_running_reqs: Maximum number of non-dummy request rows.
            page_table: GPU page table allocated by the engine.
        """

        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """
        Return the number of unassigned request rows.
        """

        return len(self._free_slots)

    def allocate(self) -> int:
        """
        Allocate and return one request table row.
        """

        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """
        Return a request table row to the free list.

        Args:
            slot: Table index previously returned by ``allocate``.
        """

        self._free_slots.append(slot)
