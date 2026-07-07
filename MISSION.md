# Mission: Mini-SGLang

## Why
The learner wants to run Mini-SGLang, understand how an LLM serving system moves requests through tokenization, scheduling, model execution, KV-cache reuse, and streaming output, and become comfortable modifying scheduler and KV-cache logic.

## Success looks like
- Run Mini-SGLang locally or in Docker and know which processes should start.
- Trace one generation request from API input to streamed output through the relevant source files.
- Explain the difference between prefill and decode in this codebase.
- Identify which scheduler and KV-cache files to edit for admission policy, decode batching, prefix reuse, and page allocation changes.
- Make small scheduler or KV-cache changes with targeted tests instead of guessing.

## Constraints
- Use this repository as the primary teaching workspace.
- Keep lessons short and tied to actual source files.
- Teach toward code modification, not only conceptual familiarity.

## Out of scope
- Full production SGLang internals unless needed to explain a Mini-SGLang design choice.
- CUDA kernel authoring as a first learning goal.
- Model architecture training details unrelated to serving.
