# Mini-SGLang Resources

## Knowledge

- [Local: README.md](./README.md)
  Start here for the project purpose, supported usage modes, and the list of major serving optimizations.
- [Local: docs/structures.md](./docs/structures.md)
  Use for the high-level process architecture, request lifecycle, and module map.
- [Local: docs/features.md](./docs/features.md)
  Use for feature-specific concepts such as tensor parallelism, chunked prefill, attention backends, CUDA graph replay, radix cache, and overlap scheduling.
- [Local: python/minisgl/core.py](./python/minisgl/core.py)
  Use for the central request, batch, context, and sampling data structures that most runtime code depends on.
- [Local: python/minisgl/server/launch.py](./python/minisgl/server/launch.py)
  Use for understanding how Mini-SGLang starts its cooperating processes.
- [Local: python/minisgl/server/args.py](./python/minisgl/server/args.py)
  Use for understanding CLI flags, defaults, model path handling, tensor parallel size, cache type, attention backend, and shell mode.
- [Local: python/minisgl/env.py](./python/minisgl/env.py)
  Use for runtime environment variables such as overlap scheduling, shell sampling defaults, FlashInfer tensor-core usage, and PyNCCL buffer size.
- [Local: python/minisgl/scheduler/scheduler.py](./python/minisgl/scheduler/scheduler.py)
  Use for understanding how requests are accepted, scheduled, and advanced through prefill and decode.
- [Local: python/minisgl/engine/engine.py](./python/minisgl/engine/engine.py)
  Use for understanding the single-worker execution path around model forward passes, KV cache, attention backend, and sampling.
- [Local: python/minisgl/scheduler/prefill.py](./python/minisgl/scheduler/prefill.py)
  Use for understanding prompt admission, chunked prefill, and how prefix-cache matches become request state.
- [Local: python/minisgl/scheduler/cache.py](./python/minisgl/scheduler/cache.py)
  Use for understanding scheduler-owned physical KV-page allocation and page-table writes.
- [Local: python/minisgl/scheduler/decode.py](./python/minisgl/scheduler/decode.py)
  Use for direct policy tests around the running decode request set and decode batch ordering.
- [Local: python/minisgl/scheduler/table.py](./python/minisgl/scheduler/table.py)
  Use for direct tests around request table-row allocation and freeing.
- [Local: python/minisgl/kvcache/radix_cache.py](./python/minisgl/kvcache/radix_cache.py)
  Use for understanding the compressed prefix tree that implements radix prefix reuse.
- [Local: python/minisgl/kernel/radix.py](./python/minisgl/kernel/radix.py)
  Use for the Python wrapper around the radix-cache token comparison helper.
- [Local: python/minisgl/kernel/csrc/src/radix.cpp](./python/minisgl/kernel/csrc/src/radix.cpp)
  Use for the C++ implementation of common-prefix comparison used during radix-tree walks.
- [Local: tests/core/test_cache_allocate.py](./tests/core/test_cache_allocate.py)
  Use for concrete page-alignment expectations around cache allocation and eviction.
- [Local: tests/core/test_decode_manager.py](./tests/core/test_decode_manager.py)
  Use as the first focused policy-test example for scheduler decode behavior.
- [Local: tests/core/test_cache_manager_allocate_paged.py](./tests/core/test_cache_manager_allocate_paged.py)
  Use as a focused example for testing scheduler-owned KV page-table allocation behavior.
- [Local: pyproject.toml](./pyproject.toml)
  Use for install dependencies, Python version support, and dev/test tool configuration.
- [Local: Dockerfile](./Dockerfile)
  Use for the containerized CUDA runtime setup and cache-directory conventions.
- [External: Apple Metal PyTorch guide](https://developer.apple.com/metal/pytorch/)
  Use for official Apple guidance on PyTorch acceleration through Metal Performance Shaders.
- [External: PyTorch MPS backend docs](https://docs.pytorch.org/docs/2.12/notes/mps.html)
  Use for official PyTorch behavior and availability of the <code>mps</code> device.
- [External: SGLang Apple Silicon with Metal docs](https://docs.sglang.io/docs/hardware-platforms/apple_metal)
  Use for upstream SGLang's Apple Silicon direction; treat it as inspiration, not as a drop-in Mini-SGLang port.
- [External: FlashInfer installation docs](https://docs.flashinfer.ai/installation.html)
  Use for understanding why the current FlashInfer dependency points toward CUDA/Linux rather than macOS/MPS.
- [External: SGLang GitHub repository](https://github.com/sgl-project/sglang)
  Use as the parent project for comparing Mini-SGLang's simplified design against production SGLang.
- [External: SGLang LMSYS blog, RadixAttention section](https://lmsys.org/blog/2024-01-17-sglang/)
  Use for the original intuition behind prefix-sharing and radix attention.
- [External: Sarathi-Serve paper](https://arxiv.org/abs/2403.02310)
  Use when learning why chunked prefill helps long-context serving.
- [External: NanoFlow paper](https://arxiv.org/abs/2408.12757)
  Use when learning why overlap scheduling matters for serving throughput.

## Wisdom (Communities)

- [SGLang GitHub discussions and issues](https://github.com/sgl-project/sglang/issues)
  Use for seeing real bug reports, performance questions, and production tradeoffs around the larger system.

## Gaps

- Add run-environment notes after the learner confirms GPU, CUDA, and Docker/WSL2 setup.
- Add targeted test references as scheduler and KV-cache lessons become concrete.
