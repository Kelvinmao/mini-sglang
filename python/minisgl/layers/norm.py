from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

        major, _ = torch.cuda.get_device_capability()
        self.use_triton = major < 8

        if self.use_triton:
            from flashinfer.triton.norm import rms_norm
            self.rmsnorm = rms_norm
        else:
            from flashinfer import rmsnorm
            self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_triton:
            return self.rmsnorm(x, self.weight, self.eps)

        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        self.rmsnorm(x2, self.weight, out, self.eps)
        return out.reshape(shape)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if not self.use_triton:
            self.rmsnorm(x, self.weight, self.eps, out=x)
            return

        # `x` may be a non-contiguous view (e.g. a column slice of a fused QKV
        # tensor in QK-norm). In that case `reshape` returns a copy, so we must
        # explicitly write the normalized result back into `x`.
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.empty_like(x2)
        self.rmsnorm(x2, self.weight, out, self.eps)
        x.copy_(out.view_as(x))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

        major, _ = torch.cuda.get_device_capability()
        self.use_triton = major < 8

        if self.use_triton:
            from flashinfer.triton.norm import (
                rms_norm,
                rms_norm_add_residual,
            )

            self.rmsnorm = rms_norm
            self.fused_add_rmsnorm = rms_norm_add_residual
        else:
            from flashinfer import fused_add_rmsnorm, rmsnorm

            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if not self.use_triton:
                return self.rmsnorm(x, self.weight, self.eps), x

            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            out = torch.empty_like(x2)
            self.rmsnorm(x2, self.weight, out, self.eps)
            return out.reshape(shape), x

        if not self.use_triton:
            self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual

        x2 = x.reshape(-1, x.shape[-1])
        r2 = residual.reshape(-1, residual.shape[-1])

        self.fused_add_rmsnorm(
            x2,
            r2,
            self.weight,
            self.eps,
            x_out=x2,
        )

        return x, residual