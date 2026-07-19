"""Curated offline driver that records a scheduler + KV-cache trace.

Run this on a GPU to produce ``visualizer/sample-trace.json``, the default
artifact the replay viewer loads. The scenario is built from explicit token-id
prompts so the radix-tree behaviour is deterministic and independent of the
tokenizer:

* two prompts share a long prefix  -> prefix match + reuse,
* several long distinct prompts     -> chunked prefill + cache pressure,
* a small page budget               -> forced eviction,
* a fixed decode length             -> clean decode steps.

Tracing requires overlap scheduling disabled; this script sets both environment
variables before importing Mini-SGLang so each recorded step is one clean
schedule -> forward -> commit unit.

Note for older GPUs (Turing / RTX 20-series, 64 KB shared memory per block):
FlashInfer chooses its prefill tile from the *packed* query length
(``chunk x GQA_group_size``); a packed length > 64 selects the 128-wide tile,
which needs 65616 bytes and does not fit. Qwen3-0.6B has GQA group size 2 and
head_dim 128, so the chunk must be <= 32 (packed <= 64 -> 64-wide tile, ~49 KB).
The default below is 32 for this reason; raise ``--max-extend-tokens`` on newer
GPUs. (The ``fa`` backend does not help here: sgl-kernel FlashAttention targets
newer architectures.)
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch


def build_scenario() -> List[List[int]]:
    """
    Build deterministic token-id prompts that exercise reuse and eviction.
    """

    shared_prefix = list(range(1, 129))  # 128-token system prompt shared by two requests
    reqs: List[List[int]] = [
        shared_prefix + list(range(1000, 1064)),  # establishes the shared prefix
        shared_prefix + list(range(2000, 2064)),  # reuses the shared prefix
        list(range(3000, 3256)),  # distinct long prompt -> chunked prefill
        list(range(4000, 4256)),  # distinct long prompt -> more cache pressure
        list(range(5000, 5256)),  # distinct long prompt -> forces eviction
    ]
    return reqs


def main() -> None:
    """
    Parse arguments, run the scenario, and dump the trace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "visualizer", "sample-trace.json"),
        help="Trace output path (defaults to visualizer/sample-trace.json).",
    )
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--num-pages",
        type=int,
        default=384,
        help="Small on purpose so distinct prompts force eviction.",
    )
    parser.add_argument(
        "--max-extend-tokens",
        type=int,
        default=32,
        help="Prefill budget per step; below prompt length to force chunked prefill. "
        "<= 32 keeps FlashInfer's prefill tile at 64-wide so it fits 64 KB GPUs "
        "(packed query length = chunk x GQA group size must stay <= 64).",
    )
    parser.add_argument("--max-tokens", type=int, default=8, help="Decode tokens per request.")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--attn",
        default="",
        help="Attention backend override, e.g. 'fa,fi' (prefill,decode). "
        "Empty lets Mini-SGLang auto-select.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model compute dtype. Default float16 because Turing (sm_75) GPUs "
        "have no native bfloat16 support and the FlashInfer prefill kernel fails "
        "to launch in bf16 there. Use bfloat16 only on Ampere+ (sm_80+).",
    )
    args = parser.parse_args()

    output = os.path.abspath(args.output)

    # Environment must be set before importing Mini-SGLang so the env registry
    # and the tracer observe them at construction time.
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    os.environ["MINISGL_TRACE_SCHEDULER_PATH"] = output

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from minisgl.trace import get_tracer

    llm_kwargs = dict(
        page_size=args.page_size,
        num_page_override=args.num_pages,
        max_extend_tokens=args.max_extend_tokens,
        max_seq_len_override=args.max_seq_len,
        cuda_graph_max_bs=0,  # disable CUDA graph for a small, legible demo run
    )
    if args.attn:
        llm_kwargs["attention_backend"] = args.attn

    llm = LLM(
        args.model,
        dtype=torch.float16,
        memory_ratio=0.7,
        **llm_kwargs,
    )

    prompts = build_scenario()
    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
    llm.generate(prompts, sampling_params)

    path = get_tracer().dump()
    print(f"Wrote {len(get_tracer().steps)} steps to {path}")


if __name__ == "__main__":
    main()
