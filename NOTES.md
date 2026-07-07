# Notes

- The learner asked to learn the ideas in the Mini-SGLang repository.
- Teaching should emphasize the actual code path in this repo, not a generic LLM-serving overview.
- Mission clarified: run Mini-SGLang, understand LLM serving, and modify scheduler/KV-cache logic.
- Lesson 2 should set up lesson 3 on radix-cache internals.
- Lesson 3 should set up a practical run/debug lesson.
- Local preflight on 2026-07-07: Python 3.12.12 and uv 0.11.1 are available; active interpreter lacks torch; nvidia-smi reports GPU access blocked by the operating system.
- Lesson 4 should set up a targeted modification/testing workflow for scheduler and cache code.
- Lesson 5 should set up a guided miniature change exercise, likely around DecodeManager or CacheManager.
- Lesson 6 added focused DecodeManager tests; next guided exercise should use CacheManager.
- Verification for lesson 6 on 2026-07-07: `python -m pytest tests/core/test_decode_manager.py -q` failed because pytest-cov is missing; rerun with `-o addopts=''` reached collection and failed because torch is missing.
- Lesson 7 added focused CacheManager allocate_paged tests. This completes the planned teaching sequence for tracing, running/debugging, and making small scheduler/KV-cache changes with tests.
- Verification for lesson 7 on 2026-07-07: `python -m pytest tests/core/test_cache_manager_allocate_paged.py -q` failed because pytest-cov is missing; rerun with `-o addopts=''` reached collection and failed because torch is missing.
- Follow-up Mac Mini question answered on 2026-07-07: current Mini-SGLang does not run as-is on macOS/Apple Silicon; a pragmatic adaptation should be a single-device CPU/MPS educational backend before any Metal kernel optimization.
- Follow-up commenting question answered on 2026-07-07: add comments selectively for invariants, intent, and hidden constraints; avoid line-by-line tutorial comments in production source.
- Code-review TODOs completed on 2026-07-07: shared scheduler test request setup moved to `tests/core/conftest.py`; CacheManager tests now cover non-page-aligned cached_len allocation; durable lessons point here for dated verification blockers.
- Verification for TODO completion on 2026-07-07: `python -m py_compile tests/core/conftest.py tests/core/test_decode_manager.py tests/core/test_cache_manager_allocate_paged.py` passed; HTML parse smoke passed; focused pytest commands are blocked at collection because the active interpreter lacks `torch`.
