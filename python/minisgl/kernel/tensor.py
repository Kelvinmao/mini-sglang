"""Small TVM FFI tensor test wrapper used by kernel tests."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_test_tensor_module() -> Module:
    """
    Load the tensor test extension once per process.
    """

    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    Call the C++ tensor test helper.
    """

    return _load_test_tensor_module().test(x, y)
