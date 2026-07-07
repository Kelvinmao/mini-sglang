"""Attention backend interfaces.

Mini-SGLang keeps model layers independent of the concrete attention kernel.
Each backend converts a scheduler ``Batch`` into kernel-specific metadata and
implements the same forward/capture/replay lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """
    Backend-specific metadata attached to a scheduled batch.
    """

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    """
    Common lifecycle for attention kernels.

    ``prepare_metadata`` runs during scheduling, ``forward`` runs inside model
    layers, and the capture/replay methods provide fixed-address metadata for
    CUDA graph decode.
    """

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    """
    Dispatch prefill and decode to different concrete attention backends.
    """

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        """
        Create a hybrid backend from phase-specific implementations.
        """

        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        Run the backend selected for the batch phase.
        """

        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """
        Prepare phase-specific metadata for a batch.
        """

        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        Initialize decode backend capture buffers.
        """

        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """
        Prepare decode backend metadata before graph capture.
        """

        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """
        Prepare decode backend metadata before graph replay.
        """

        self.decode_backend.prepare_for_replay(batch)
