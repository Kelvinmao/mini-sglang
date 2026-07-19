# Lab: Visualize the Scheduler & KV-Cache

In this lab you will produce a **trace** of Mini-SGLang's scheduler and radix
KV-cache, then replay it step-by-step in a browser to *see* prefix reuse,
chunked prefill, page allocation, and LRU eviction as they happen.

There are two ways to generate a trace:

| Path | Needs GPU? | Needs install? | What it shows |
| --- | --- | --- | --- |
| **A. CPU simulator** | No | No | A pure-Python re-implementation of the radix cache driven by a scripted workload. Fast, deterministic, runs anywhere. |
| **B. Real model** | Yes | Yes (`uv pip install -e .`) | An actual Qwen3-0.6B run through the real scheduler, recorded by the env-gated tracer. |

Both paths emit the same `visualizer/sample-trace.json` and use the same viewer.
Start with Path A to learn the tool, then try Path B for the real thing.

---

## Prerequisites

- Linux (or WSL2). The viewer is plain HTML/JS and works in any modern browser.
- Python 3.10+ on your `PATH` (Path A needs nothing else).
- For Path B only: an NVIDIA GPU + the Mini-SGLang environment (see Step B1).

All commands below run from the repository root:

```bash
cd /path/to/mini-sglang
```

---

## Path A — CPU simulator (no GPU, no install)

### A1. Generate the trace

```bash
python visualizer/gen_trace.py
```

This writes `visualizer/sample-trace.json` and prints how many steps it
recorded. The generator is pure standard library — it reimplements the radix
tree from [python/minisgl/kvcache/radix_cache.py](../python/minisgl/kvcache/radix_cache.py)
and drives a scripted workload (two prompts sharing a prefix, several long
distinct prompts, a small page budget) so you get reuse **and** eviction.

Useful knobs:

```bash
python visualizer/gen_trace.py --num-pages 40 --chunk 4 --output-len 3
```

- `--num-pages` — total pages in the pool. Smaller ⇒ more eviction pressure.
- `--chunk` — prefill chunk size ⇒ controls chunked-prefill granularity.
- `--output-len` — decode tokens generated per request.

### A2. Open the viewer

The viewer auto-loads `sample-trace.json` over HTTP, so serve the folder:

```bash
cd visualizer && python -m http.server 8000
```

Open <http://localhost:8000/> in your browser. Jump to
[Step 3 — Read the viewer](#step-3--read-the-viewer).

> Do **not** open `index.html` via `file://` — the auto-load `fetch` is
> CORS-blocked there. Either serve over `http://` as above, or use the
> **Load trace…** button / drag-and-drop the JSON onto the page.

---

## Path B — Real model on GPU

### B1. Build the environment (one time)

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

This installs torch, FlashInfer, `sgl-kernel`, and friends (several GB). The
CUDA kernels are JIT-compiled on first use, so you need a matching **CUDA
toolkit** (`nvcc`) available; check your driver's CUDA version with
`nvidia-smi`.

### B2. Generate a real trace

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/offline/trace_radix_demo.py \
    --output visualizer/sample-trace.json
```

This loads Qwen3-0.6B and runs a curated scenario (shared-prefix prompts, long
distinct prompts, a small page budget) through the **real** scheduler. The
driver sets two environment variables for you before importing Mini-SGLang:

- `MINISGL_TRACE_SCHEDULER_PATH=<output>` — a non-empty path enables the tracer
  (empty ⇒ no tracing, no overhead).
- `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` — required, so each recorded step is
  one clean *schedule → forward → commit* unit.

GPU-safety notes:

- Defaults are deliberately small: `--num-pages 384`, `--max-extend-tokens 32`,
  CUDA graph disabled. They keep memory tiny and legible.
- `--max-extend-tokens 32` keeps FlashInfer's prefill tile at 64-wide so it fits
  64 KB-shared-memory GPUs (Turing / RTX 20-series). On newer GPUs (Ada, Hopper)
  you can safely raise it.

### B3. Open the viewer

Same as Path A:

```bash
cd visualizer && python -m http.server 8000   # then open http://localhost:8000/
```

---

## Step 3 — Read the viewer

The page replays one scheduler iteration at a time. Controls live in the top bar:

- **⟨ / ⟩** — step backward / forward one iteration.
- **▶** — play / pause auto-advance. (Spacebar also toggles play.)
- **scrubber** — jump to any step; the label shows the step index and a
  one-line description (e.g. `prefill: 1 req(s), 32 token(s); reused 128`).

Three panels update on every step:

### Radix prefix tree (left)

Each node is one compressed edge of the prefix tree. Colors (see the legend):

- **protected** — `ref_count > 0`; in use by an active request, cannot be evicted.
- **evictable** — `ref_count == 0`; cached but reclaimable by LRU.
- **matched** — an edge that was hit by prefix matching *this* step (reuse).
- **inserted** — a new edge added *this* step.

Watch two shared-prefix requests: the first **inserts** the prefix, the second
**matches** it instead of recomputing — that's the radix cache paying off.

### Batch composition (top right)

The requests scheduled into the current forward pass. For each entry:

- `cached_len` — tokens served from the KV cache (prefix reuse).
- `device_len` — tokens already resident on device.
- `extend_len` — new tokens computed this step. During **chunked prefill** a
  long prompt shows up across several steps, each with a small `extend_len`.

### Memory & pages (bottom right)

- The stat block and bar show `used / free / total` pages.
- **Free pages over time** is a sparkline of the page budget across all steps.
  When it dips and you see nodes disappear from the tree, that's **LRU
  eviction** reclaiming evictable leaves under pressure.

---

## What to look for

Try to spot each behaviour as you step through the trace:

1. **Prefix reuse** — second shared-prefix request has a large `cached_len` and
   a `matched` edge; its `extend_len` covers only the *new* suffix.
2. **Chunked prefill** — one long prompt spread over multiple prefill steps,
   each adding a bounded `extend_len`.
3. **Allocation** — decode steps allocate one page per request per token; watch
   `used_pages` climb.
4. **Eviction** — when free pages run low, evictable leaves vanish from the tree
   and the sparkline recovers.

---

## How a step is recorded

Each step in `sample-trace.json` is a hybrid *snapshot + events* record:

```jsonc
{
  "index", "phase", "description",
  "batch":  [{ "uid", "phase", "cached_len", "device_len", "extend_len" }],
  "tree":   [{ "uuid", "parent", "length", "ref_count", "protected", "tokens" }],
  "evictable_size", "protected_size", "used_pages", "free_pages",
  "num_pages", "page_size",
  "events": [{ "kind": "match | insert | evict | alloc", "..." : "..." }]
}
```

The snapshot is authoritative state; the events are what changed. The viewer
draws the snapshot and highlights the events. Where each field comes from:

| Field | Source |
| --- | --- |
| Batch composition | `normal_loop` in [python/minisgl/scheduler/scheduler.py](../python/minisgl/scheduler/scheduler.py) |
| `match` / `insert` / `evict` events | [python/minisgl/kvcache/radix_cache.py](../python/minisgl/kvcache/radix_cache.py) |
| `alloc` events | `allocate_paged` in [python/minisgl/scheduler/cache.py](../python/minisgl/scheduler/cache.py) |
| Tree snapshot | `dump_tree` on the radix cache |
| Recording glue | [python/minisgl/trace/tracer.py](../python/minisgl/trace/tracer.py) |

For a deeper map of the pipeline and hook points, see
[reference/0010-scheduler-cache-visualizer.html](../reference/0010-scheduler-cache-visualizer.html).

---

## Troubleshooting

- **Blank viewer / "No trace loaded".** You opened `index.html` via `file://`.
  Serve with `python -m http.server` and use `http://localhost:8000/`, or use
  the **Load trace…** button.
- **`ModuleNotFoundError: No module named 'minisgl'` (Path B).** The environment
  isn't active/installed. Run Step B1 (`uv venv … && uv pip install -e .`) and
  `source .venv/bin/activate`.
- **JIT/kernel compile error on GPU (Path B).** Usually a missing or mismatched
  CUDA toolkit (`nvcc`). Install one matching your driver's CUDA version.
- **Prefill crashes on an older GPU (Turing / RTX 20-series).** Keep
  `--max-extend-tokens` ≤ 32 so FlashInfer picks the 64-wide prefill tile that
  fits 64 KB of shared memory.
- **No eviction visible.** Lower `--num-pages` (Path A) to increase cache
  pressure so LRU eviction kicks in.
