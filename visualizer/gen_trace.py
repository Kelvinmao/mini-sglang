"""Generate a scheduler + KV-cache trace WITHOUT a GPU or the model.

Some GPUs (e.g. Turing / RTX 20-series) cannot run Mini-SGLang's attention
kernels, so ``trace_radix_demo.py`` crashes before producing a trace. The
scheduler behaviour worth visualizing — prefix matching, chunked prefill
admission, page allocation, and LRU eviction — lives in the CPU-side radix
cache, not on the GPU. This script reimplements that radix-cache algorithm in
pure Python (mirroring ``python/minisgl/kvcache/radix_cache.py``) and drives a
scripted workload through it, emitting the same ``trace.json`` the viewer reads.

No dependencies beyond the standard library: runs anywhere.

    python visualizer/gen_trace.py            # writes visualizer/sample-trace.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Radix prefix tree (page_size = 1: one token per page), mirroring
# python/minisgl/kvcache/radix_cache.py.
# ---------------------------------------------------------------------------


class Node:
    """One compressed edge of the radix tree."""

    _counter = 0

    def __init__(self, key: List[int], value: List[int], parent: Optional["Node"], tic: int):
        self.key = list(key)  # token ids on this edge
        self.value = list(value)  # physical page ids on this edge
        self.children: Dict[int, "Node"] = {}
        self.parent = parent
        self.ref = 0
        self.tic = tic
        self.uuid = Node._counter
        Node._counter += 1

    @property
    def length(self) -> int:
        return len(self.key)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return not self.children


def _common_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RadixTree:
    """Pure-Python radix prefix cache with LRU eviction."""

    def __init__(self) -> None:
        self.clock = 0
        self.root = Node([], [], None, self._tic())
        self.root.ref = 1  # root is always protected

    def _tic(self) -> int:
        self.clock += 1
        return self.clock

    def _split(self, node: Node, pos: int) -> Node:
        """Split ``node``'s edge at ``pos``; return the new prefix node."""
        parent = node.parent
        assert parent is not None
        new_node = Node(node.key[:pos], node.value[:pos], parent, node.tic)
        new_node.ref = node.ref
        parent.children[new_node.key[0]] = new_node
        node.key = node.key[pos:]
        node.value = node.value[pos:]
        node.parent = new_node
        new_node.children[node.key[0]] = node
        return new_node

    def _walk(self, tokens: List[int]) -> Tuple[Node, int]:
        matched = 0
        node = self.root
        tic = self._tic()
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                return node, matched
            node = child
            m = _common_len(node.key, tokens[matched:])
            matched += m
            if m != node.length:
                node = self._split(node, m)
                node.tic = tic
                return node, matched
            node.tic = tic
        return node, matched

    def match(self, tokens: List[int]) -> Tuple[Node, int]:
        return self._walk(tokens)

    def insert(self, tokens: List[int], pages: List[int]) -> Tuple[int, Node, int]:
        """Insert a prefix; return (already_cached_len, node, new_edge_len)."""
        node, matched = self._walk(tokens)
        edge_len = 0
        if matched != len(tokens):
            new_node = Node(tokens[matched:], pages[matched:], node, self._tic())
            node.children[new_node.key[0]] = new_node
            node = new_node
            edge_len = new_node.length
        return matched, node, edge_len

    def lock(self, node: Node, unlock: bool = False) -> None:
        while not node.is_root():
            node.ref += -1 if unlock else 1
            assert node.ref >= 0
            node = node.parent  # type: ignore[assignment]

    def evict(self, size: int) -> Tuple[List[int], List[int]]:
        """Evict LRU unprotected leaves; return (freed_pages, evicted_uuids)."""
        leaves = [n for n in self._all_nodes() if n.is_leaf() and n.ref == 0 and not n.is_root()]
        leaves.sort(key=lambda n: n.tic)
        freed: List[int] = []
        uuids: List[int] = []
        i = 0
        while len(freed) < size:
            assert i < len(leaves), "not enough evictable cache"
            node = leaves[i]
            i += 1
            freed.extend(node.value)
            uuids.append(node.uuid)
            parent = node.parent
            assert parent is not None
            del parent.children[node.key[0]]
            if parent.is_leaf() and parent.ref == 0 and not parent.is_root():
                leaves.append(parent)  # newly exposed leaf becomes evictable
        return freed, uuids

    def _all_nodes(self) -> List[Node]:
        out: List[Node] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def sizes(self) -> Tuple[int, int]:
        protected = evictable = 0
        for n in self._all_nodes():
            if n.is_root():
                continue
            if n.ref > 0:
                protected += n.length
            else:
                evictable += n.length
        return protected, evictable

    def dump(self, preview: int = 8) -> List[dict]:
        nodes = []
        for n in self._all_nodes():
            nodes.append(
                {
                    "uuid": n.uuid,
                    "parent": None if n.is_root() else n.parent.uuid,  # type: ignore[union-attr]
                    "length": n.length,
                    "ref_count": n.ref,
                    "protected": n.ref > 0,
                    "tokens": n.key[:preview],
                }
            )
        return nodes


# ---------------------------------------------------------------------------
# Physical page pool with eviction, mirroring CacheManager._allocate.
# ---------------------------------------------------------------------------


class PagePool:
    def __init__(self, num_pages: int, tree: RadixTree):
        self.num_pages = num_pages
        self.free: List[int] = list(range(num_pages))
        self.tree = tree
        self.last_evict: Optional[dict] = None

    def allocate(self, n: int) -> List[int]:
        self.last_evict = None
        if n > len(self.free):
            need = n - len(self.free)
            freed, uuids = self.tree.evict(need)
            self.free.extend(freed)
            self.last_evict = {"kind": "evict", "requested": need, "evicted_len": len(freed), "nodes": uuids}
        out = self.free[:n]
        self.free = self.free[n:]
        return out

    def release(self, pages: List[int]) -> None:
        self.free.extend(pages)


# ---------------------------------------------------------------------------
# Trace recorder.
# ---------------------------------------------------------------------------


@dataclass
class Recorder:
    tree: RadixTree
    pool: PagePool
    steps: List[dict] = field(default_factory=list)
    _events: List[dict] = field(default_factory=list)

    def emit(self, event: Optional[dict]) -> None:
        if event:
            self._events.append(event)

    def step(self, phase: str, batch: List[dict], description: str) -> None:
        protected, evictable = self.tree.sizes()
        free_pages = len(self.pool.free)
        self.steps.append(
            {
                "index": len(self.steps),
                "phase": phase,
                "description": description,
                "batch": batch,
                "tree": self.tree.dump(),
                "evictable_size": evictable,
                "protected_size": protected,
                "used_pages": self.pool.num_pages - free_pages,
                "free_pages": free_pages,
                "num_pages": self.pool.num_pages,
                "page_size": 1,
                "events": self._events,
            }
        )
        self._events = []


# ---------------------------------------------------------------------------
# Scenario: scripted requests exercising reuse, chunked prefill, eviction.
# ---------------------------------------------------------------------------


@dataclass
class Req:
    uid: int
    table_idx: int
    prompt: List[int]
    output_len: int
    cached_len: int = 0
    device_len: int = 0
    pages: List[int] = field(default_factory=list)  # page id per token position
    handle: Optional[Node] = None


def _batch_entry(req: Req, phase: str, extend: int) -> dict:
    return {
        "uid": req.uid,
        "table_idx": req.table_idx,
        "phase": phase,
        "cached_len": req.device_len - extend,
        "device_len": req.device_len,
        "extend_len": extend,
    }


def run_scenario(num_pages: int, chunk: int, output_len: int, out_path: str) -> int:
    tree = RadixTree()
    pool = PagePool(num_pages, tree)
    rec = Recorder(tree, pool)

    shared = list(range(101, 113))  # 12-token shared system prompt
    requests = [
        Req(0, 0, shared + [201, 202, 203, 204], output_len),  # establishes shared prefix
        Req(1, 1, shared + [211, 212, 213, 214], output_len),  # reuses shared prefix
        Req(2, 2, list(range(301, 317)), output_len),  # distinct -> pressure
        Req(3, 3, list(range(401, 417)), output_len),  # distinct -> pressure
        Req(4, 4, list(range(501, 517)), output_len),  # distinct -> forces eviction
    ]

    def admit_and_prefill(req: Req) -> None:
        # 1. Match the longest cached prefix (scheduler excludes the last token).
        node, matched = tree.match(req.prompt[:-1])
        tree.lock(node)
        req.handle = node
        req.cached_len = matched
        req.device_len = matched
        # Physical pages backing the matched prefix (walk node -> root).
        chain: List[Node] = []
        walk: Optional[Node] = node
        while walk is not None and not walk.is_root():
            chain.append(walk)
            walk = walk.parent
        req.pages = []
        for nd in reversed(chain):
            req.pages.extend(nd.value)
        rec.emit({"kind": "match", "input_len": len(req.prompt) - 1, "matched_len": matched, "node": node.uuid})

        # 2. Prefill the uncached remainder in chunks.
        pos = matched
        while pos < len(req.prompt):
            end = min(pos + chunk, len(req.prompt))
            extend = end - pos
            new_pages = pool.allocate(extend)
            rec.emit(pool.last_evict)
            rec.emit({"kind": "alloc", "pages": extend})
            req.pages.extend(new_pages)
            req.device_len = end
            phase_name = "prefill" if end == len(req.prompt) else "chunked"
            desc = _describe("prefill", [(req, extend)], rec._events)
            rec.step(phase_name, [_batch_entry(req, phase_name, extend)], desc)
            pos = end

        # 3. Cache the full prompt prefix so later requests can reuse it.
        already, new_node, edge_len = tree.insert(req.prompt, req.pages)
        tree.lock(req.handle, unlock=True)
        tree.lock(new_node)
        req.handle = new_node
        rec.emit({"kind": "insert", "already_cached": already, "insert_len": len(req.prompt), "node": new_node.uuid, "edge_len": edge_len})

    def decode_group(group: List[Req]) -> None:
        for _ in range(output_len):
            batch = []
            for req in group:
                new_pages = pool.allocate(1)
                rec.emit(pool.last_evict)
                rec.emit({"kind": "alloc", "pages": 1})
                req.pages.append(new_pages[0])
                req.device_len += 1
                batch.append(_batch_entry(req, "decode", 1))
            rec.step("decode", batch, f"decode: {len(group)} req(s), one token each")

    def finish(req: Req) -> None:
        # Unlock the prefix (now evictable) and release decode tail pages.
        tree.lock(req.handle, unlock=True)  # type: ignore[arg-type]
        pool.release(req.pages[len(req.prompt):])

    # Phase 1: two requests that share a prefix, decoded together.
    admit_and_prefill(requests[0])
    admit_and_prefill(requests[1])
    decode_group([requests[0], requests[1]])
    finish(requests[0])
    finish(requests[1])

    # Phase 2: distinct requests that create pressure and force eviction.
    for req in requests[2:]:
        admit_and_prefill(req)
        decode_group([req])
        finish(req)

    meta = {
        "model": "cpu-simulator (no GPU)",
        "page_size": 1,
        "num_pages": num_pages,
        "cache_type": "radix",
        "overlap_disabled": True,
        "created_at": "generated by visualizer/gen_trace.py",
        "version": 1,
    }
    document = {"meta": meta, "steps": rec.steps}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    return len(rec.steps)


def _describe(phase: str, batch: List[Tuple[Req, int]], events: List[dict]) -> str:
    tokens = sum(extend for _, extend in batch)
    parts = [f"{phase}: {len(batch)} req(s), {tokens} token(s)"]
    matched = sum(e.get("matched_len", 0) for e in events if e.get("kind") == "match")
    if matched:
        parts.append(f"reused {matched}")
    evicted = sum(e.get("evicted_len", 0) for e in events if e.get("kind") == "evict")
    if evicted:
        parts.append(f"evicted {evicted}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "sample-trace.json"),
        help="Trace output path (defaults to visualizer/sample-trace.json).",
    )
    parser.add_argument("--num-pages", type=int, default=40, help="Small on purpose to force eviction.")
    parser.add_argument("--chunk", type=int, default=4, help="Prefill chunk size (chunked prefill).")
    parser.add_argument("--output-len", type=int, default=3, help="Decode tokens per request.")
    args = parser.parse_args()

    n = run_scenario(args.num_pages, args.chunk, args.output_len, os.path.abspath(args.output))
    print(f"Wrote {n} steps to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
