// Replay viewer for Mini-SGLang scheduler + KV-cache traces.
//
// A trace is a single JSON document: { meta, steps }, where each step is a
// hybrid record (resulting state + the cache events that produced it). This
// file loads a trace, drives a scrubber over the steps, and renders four
// coordinated panels: the radix prefix tree, batch composition, and memory.

const SVGNS = "http://www.w3.org/2000/svg";

const state = {
  trace: null,
  step: 0,
  playing: false,
  timer: null,
};

// ----------------------------------------------------------------------------
// Small SVG helpers
// ----------------------------------------------------------------------------
function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// ----------------------------------------------------------------------------
// Trace loading
// ----------------------------------------------------------------------------
function loadTrace(obj) {
  if (!obj || !Array.isArray(obj.steps)) {
    alert("This file does not look like a Mini-SGLang trace ({ meta, steps }).");
    return;
  }
  state.trace = obj;
  state.step = 0;
  stopPlaying();

  const scrubber = document.getElementById("scrubber");
  scrubber.max = Math.max(0, obj.steps.length - 1);
  scrubber.value = 0;

  document.getElementById("empty-hint").style.display = obj.steps.length ? "none" : "";
  renderMeta();
  renderStep(0);
}

function renderMeta() {
  const m = state.trace.meta || {};
  const line = [
    m.model ? `model ${m.model}` : null,
    m.cache_type ? `${m.cache_type} cache` : null,
    m.page_size != null ? `page ${m.page_size}` : null,
    m.num_pages != null ? `${m.num_pages} pages` : null,
    m.overlap_disabled ? "overlap off" : null,
    `${state.trace.steps.length} steps`,
  ]
    .filter(Boolean)
    .join(" · ");
  document.getElementById("meta-line").textContent = line;
}

// ----------------------------------------------------------------------------
// Step navigation
// ----------------------------------------------------------------------------
function goTo(i) {
  if (!state.trace) return;
  const n = state.trace.steps.length;
  state.step = Math.max(0, Math.min(n - 1, i));
  document.getElementById("scrubber").value = state.step;
  renderStep(state.step);
}

function stopPlaying() {
  state.playing = false;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  document.getElementById("btn-play").textContent = "▶";
}

function togglePlay() {
  if (!state.trace) return;
  if (state.playing) return stopPlaying();
  state.playing = true;
  document.getElementById("btn-play").textContent = "⏸";
  state.timer = setInterval(() => {
    if (state.step >= state.trace.steps.length - 1) return stopPlaying();
    goTo(state.step + 1);
  }, 900);
}

// ----------------------------------------------------------------------------
// Rendering a step
// ----------------------------------------------------------------------------
function renderStep(i) {
  const step = state.trace.steps[i];
  if (!step) return;
  document.getElementById("step-index").textContent = `Step ${step.index} · ${step.phase}`;
  document.getElementById("step-desc").textContent = step.description || "";
  renderTree(step);
  renderBatch(step);
  renderMemory(step);
}

// ---- Radix tree -------------------------------------------------------------
function layoutTree(nodes) {
  const map = new Map();
  nodes.forEach((n) => map.set(n.uuid, { ...n, children: [] }));
  let root = null;
  map.forEach((n) => {
    if (n.parent === null || n.parent === undefined) root = n;
    else if (map.has(n.parent)) map.get(n.parent).children.push(n);
  });
  map.forEach((n) => n.children.sort((a, b) => a.uuid - b.uuid));

  const xGap = 96;
  const yGap = 92;
  let leaf = 0;
  function assign(node, depth) {
    node.depth = depth;
    if (node.children.length === 0) {
      node.x = leaf * xGap;
      leaf += 1;
    } else {
      node.children.forEach((c) => assign(c, depth + 1));
      node.x = (node.children[0].x + node.children[node.children.length - 1].x) / 2;
    }
    node.y = depth * yGap;
  }
  if (root) assign(root, 0);
  return { root, list: [...map.values()] };
}

function reconcile(layer, data, keyFn, create, update) {
  const seen = new Set();
  data.forEach((d) => {
    const key = String(keyFn(d));
    seen.add(key);
    let el = layer.querySelector(`[data-key="${key}"]`);
    if (!el) {
      el = create(d);
      el.setAttribute("data-key", key);
      el.classList.add("enter");
      layer.appendChild(el);
    } else {
      el.classList.remove("exit");
    }
    update(el, d);
  });
  Array.from(layer.children).forEach((el) => {
    if (!seen.has(el.getAttribute("data-key")) && !el.classList.contains("exit")) {
      el.classList.add("exit");
      setTimeout(() => el.remove(), 360);
    }
  });
}

function renderTree(step) {
  const svgEl = document.getElementById("tree-svg");
  let edgeLayer = svgEl.querySelector("#edge-layer");
  let nodeLayer = svgEl.querySelector("#node-layer");
  if (!edgeLayer) {
    edgeLayer = svg("g", { id: "edge-layer" });
    nodeLayer = svg("g", { id: "node-layer" });
    svgEl.appendChild(edgeLayer);
    svgEl.appendChild(nodeLayer);
  }

  const { list } = layoutTree(step.tree || []);
  if (list.length === 0) {
    clear(edgeLayer);
    clear(nodeLayer);
    svgEl.setAttribute("viewBox", "0 0 100 100");
    return;
  }

  const pos = new Map(list.map((n) => [n.uuid, n]));
  const xs = list.map((n) => n.x);
  const margin = 60;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const maxDepth = Math.max(...list.map((n) => n.depth));
  const width = maxX - minX + margin * 2;
  const height = maxDepth * 92 + margin * 2;
  svgEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svgEl.setAttribute("width", width);
  svgEl.setAttribute("height", height);
  const px = (n) => n.x - minX + margin;
  const py = (n) => n.y + margin;

  // Which nodes were matched / inserted this step (for highlight classes).
  const matched = new Set();
  const inserted = new Set();
  (step.events || []).forEach((e) => {
    if (e.kind === "match" && e.matched_len > 0) matched.add(e.node);
    if (e.kind === "insert") inserted.add(e.node);
  });

  // Edges keyed by child uuid.
  const edges = list.filter((n) => n.parent != null && pos.has(n.parent));
  reconcile(
    edgeLayer,
    edges,
    (n) => `e${n.uuid}`,
    () => svg("path", { class: "edge" }),
    (el, n) => {
      const p = pos.get(n.parent);
      el.setAttribute("d", `M ${px(p)} ${py(p)} L ${px(n)} ${py(n)}`);
    }
  );

  // Nodes keyed by uuid.
  reconcile(
    nodeLayer,
    list,
    (n) => `n${n.uuid}`,
    () => {
      const g = svg("g", { class: "node" });
      g.appendChild(svg("circle", { r: 22 }));
      g.appendChild(svg("text", { class: "len", dy: "0.32em" }));
      g.appendChild(svg("text", { class: "sub", dy: "2.8em" }));
      return g;
    },
    (el, n) => {
      el.setAttribute("transform", `translate(${px(n)}, ${py(n)})`);
      const isRoot = n.parent == null;
      // Update only the state classes so enter/exit animation classes survive.
      el.classList.remove("root", "protected", "evictable", "matched");
      el.classList.add(isRoot ? "root" : n.protected ? "protected" : "evictable");
      if (matched.has(n.uuid)) el.classList.add("matched");
      const len = el.querySelector(".len");
      const sub = el.querySelector(".sub");
      len.textContent = isRoot ? "root" : String(n.length);
      sub.textContent = isRoot ? "" : `ref ${n.ref_count}`;
    }
  );
}

// ---- Batch composition ------------------------------------------------------
function renderBatch(step) {
  const list = document.getElementById("batch-list");
  clear(list);
  const batch = step.batch || [];
  if (batch.length === 0) {
    list.innerHTML = '<p class="desc">No requests scheduled this step.</p>';
    return;
  }
  const maxLen = Math.max(...batch.map((b) => b.device_len), 1);

  batch.forEach((b) => {
    const row = document.createElement("div");
    row.className = "batch-row";

    const uid = document.createElement("span");
    uid.className = "uid";
    uid.textContent = `#${b.uid}`;

    const badge = document.createElement("span");
    badge.className = `badge ${b.phase}`;
    badge.textContent = b.phase;

    const barWrap = document.createElement("div");
    barWrap.style.display = "flex";
    barWrap.style.alignItems = "center";
    barWrap.style.gap = "8px";

    const bar = document.createElement("div");
    bar.className = "tokbar";
    bar.style.width = `${(b.device_len / maxLen) * 100}%`;
    const cached = document.createElement("div");
    cached.className = "cached";
    cached.style.flexBasis = `${(b.cached_len / b.device_len) * 100}%`;
    const extend = document.createElement("div");
    extend.className = "extend";
    extend.style.flexBasis = `${(b.extend_len / b.device_len) * 100}%`;
    bar.appendChild(cached);
    bar.appendChild(extend);
    bar.title = `cached ${b.cached_len} + compute ${b.extend_len} = ${b.device_len}`;

    const count = document.createElement("span");
    count.className = "uid";
    count.textContent = `${b.cached_len}+${b.extend_len}`;

    barWrap.appendChild(bar);
    barWrap.appendChild(count);
    row.appendChild(uid);
    row.appendChild(badge);
    row.appendChild(barWrap);
    list.appendChild(row);
  });
}

// ---- Memory & pages ---------------------------------------------------------
function renderMemory(step) {
  const stats = document.getElementById("mem-stats");
  const capacity = step.num_pages * step.page_size;
  const usedTokens = step.used_pages * step.page_size;
  const freeTokens = step.free_pages * step.page_size;
  const otherUsed = Math.max(0, usedTokens - step.protected_size - step.evictable_size);

  stats.innerHTML = `
    <div>pages used <b>${step.used_pages}/${step.num_pages}</b></div>
    <div>free pages <b>${step.free_pages}</b></div>
    <div>protected <b>${step.protected_size}</b> tok</div>
    <div>evictable <b>${step.evictable_size}</b> tok</div>`;

  const bar = document.getElementById("mem-bar");
  clear(bar);
  bar.setAttribute("viewBox", "0 0 100 10");
  bar.setAttribute("preserveAspectRatio", "none");
  const segs = [
    [step.protected_size, "var(--protected)"],
    [step.evictable_size, "var(--evictable)"],
    [otherUsed, "var(--prefill)"],
    [freeTokens, "var(--free)"],
  ];
  let x = 0;
  segs.forEach(([val, color]) => {
    const w = capacity > 0 ? (val / capacity) * 100 : 0;
    if (w <= 0) return;
    bar.appendChild(svg("rect", { x, y: 0, width: w, height: 10, fill: color }));
    x += w;
  });

  renderSparkline();
}

function renderSparkline() {
  const spark = document.getElementById("mem-spark");
  clear(spark);
  spark.setAttribute("viewBox", "0 0 100 100");
  spark.setAttribute("preserveAspectRatio", "none");
  const steps = state.trace.steps;
  const n = steps.length;
  if (n === 0) return;
  const pts = steps.map((s, i) => {
    const x = n === 1 ? 0 : (i / (n - 1)) * 100;
    const frac = s.num_pages > 0 ? s.free_pages / s.num_pages : 0;
    return [x, 100 - frac * 100];
  });
  spark.appendChild(
    svg("polyline", {
      points: pts.map((p) => p.join(",")).join(" "),
      fill: "none",
      stroke: "var(--accent)",
      "stroke-width": 1.5,
      "vector-effect": "non-scaling-stroke",
    })
  );
  const cur = pts[state.step];
  spark.appendChild(
    svg("circle", { cx: cur[0], cy: cur[1], r: 2.5, fill: "var(--matched)", "vector-effect": "non-scaling-stroke" })
  );
}

// ----------------------------------------------------------------------------
// Wiring
// ----------------------------------------------------------------------------
function init() {
  document.getElementById("btn-prev").addEventListener("click", () => goTo(state.step - 1));
  document.getElementById("btn-next").addEventListener("click", () => goTo(state.step + 1));
  document.getElementById("btn-play").addEventListener("click", togglePlay);
  document.getElementById("scrubber").addEventListener("input", (e) => {
    stopPlaying();
    goTo(Number(e.target.value));
  });

  document.getElementById("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) readFile(file);
  });

  const zone = document.getElementById("drop-zone");
  ["dragover", "dragenter"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) readFile(file);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") goTo(state.step + 1);
    else if (e.key === "ArrowLeft") goTo(state.step - 1);
    else if (e.key === " ") {
      e.preventDefault();
      togglePlay();
    }
  });

  // Try the checked-in sample; ignored if opened via file:// (blocked by CORS).
  fetch("sample-trace.json")
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then(loadTrace)
    .catch(() => {});
}

function readFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      loadTrace(JSON.parse(reader.result));
    } catch (err) {
      alert("Could not parse JSON: " + err.message);
    }
  };
  reader.readAsText(file);
}

init();
