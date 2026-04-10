/* app.js — merLLM dashboard */

// ── Navigation ────────────────────────────────────────────────────────────────

document.querySelectorAll("nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "metrics") loadMetricsCharts().catch(() => {});
  });
});

document.querySelectorAll(".range-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".range-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    _metricsRange = btn.dataset.range;
    loadMetricsCharts().catch(() => {});
  });
});

// ── API helpers ───────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function del(path) {
  return api(path, { method: "DELETE" });
}

// ── Polling ───────────────────────────────────────────────────────────────────

let _pollTimer         = null;
let _activityTimer     = null;
let _activityEventSrc  = null;
let _notifEventSrc     = null;

function startPolling() {
  refresh();
  _pollTimer = setInterval(refresh, 10000);
  // Seed instance cards immediately via one-shot poll, then switch to SSE push
  refreshActivity();
  _startActivityStream();
  _startNotifStream();
  // My Day panel: load once on startup, refresh every 60s
  refreshMyDay();
  setInterval(refreshMyDay, 60000);
}

function _startNotifStream() {
  if (_notifEventSrc) return;
  _notifEventSrc = new EventSource("/api/merllm/events");
  _notifEventSrc.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      if (ev.type === "job_complete" || ev.type === "job_failed") {
        const title = ev.status === "completed"
          ? `✓ Batch job complete — ${ev.source_app || "merLLM"}`
          : `✗ Batch job failed — ${ev.source_app || "merLLM"}`;
        const body = ev.prompt_preview
          ? `"${ev.prompt_preview}…"` : "";
        // In-page toast
        _showJobToast(title, body, ev.status === "completed");
        // Browser notification if permitted
        if (Notification.permission === "granted") {
          new Notification(title, {body});
        }
      }
    } catch (_) {}
  };
  _notifEventSrc.onerror = () => {
    _notifEventSrc.close();
    _notifEventSrc = null;
    setTimeout(_startNotifStream, 15000);
  };
}

function _showJobToast(title, body, success) {
  const toast = document.createElement("div");
  toast.className = "job-toast " + (success ? "toast-ok" : "toast-err");
  toast.innerHTML = `<strong>${esc(title)}</strong>${body ? "<br><span>" + esc(body) + "</span>" : ""}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.classList.add("toast-visible"), 50);
  setTimeout(() => { toast.classList.remove("toast-visible"); setTimeout(() => toast.remove(), 400); }, 6000);
}

async function requestNotifPermission() {
  if (!("Notification" in window)) {
    alert("This browser does not support desktop notifications.");
    return;
  }
  const result = await Notification.requestPermission();
  const btn = document.getElementById("notif-permission-btn");
  if (btn) {
    if (result === "granted") {
      btn.textContent = "Browser notifications enabled";
      btn.disabled = true;
    } else {
      btn.textContent = "Permission denied — try browser settings";
    }
  }
}

function _startActivityStream() {
  if (_activityEventSrc) return; // already open
  _activityEventSrc = new EventSource("/api/merllm/activity/stream");

  _activityEventSrc.onmessage = (e) => {
    try {
      const a = JSON.parse(e.data);
      // SSE payload is the raw _activity_snapshot + per-gpu loaded models are
      // not included (those come from the poll).  Merge with last known loaded state.
      if (a.gpu0 !== undefined) _mergeAndRenderInstance("ollama-gpu0", "GPU 0", a.gpu0);
      if (a.gpu1 !== undefined) _mergeAndRenderInstance("ollama-gpu1", "GPU 1", a.gpu1);
    } catch (_) {}
  };

  _activityEventSrc.onerror = () => {
    // On error, close and fall back to polling; retry SSE after 10 seconds
    _activityEventSrc.close();
    _activityEventSrc = null;
    if (!_activityTimer) {
      _activityTimer = setInterval(refreshActivity, 2000);
    }
    setTimeout(() => {
      if (_activityTimer) { clearInterval(_activityTimer); _activityTimer = null; }
      _startActivityStream();
    }, 10000);
  };
}

// Per-GPU cache so SSE updates (which carry only active state) can still
// show loaded models and the instance URL from the last full poll.
const _instCache = {
  gpu0: { url: "", loaded: null },
  gpu1: { url: "", loaded: null },
};

function _mergeAndRenderInstance(elId, label, activeSnapshot) {
  // activeSnapshot is the raw _activity entry (or null) with elapsed_sec added.
  // Build an inst object matching what renderInstanceCard expects.
  const gpuKey = elId === "ollama-gpu0" ? "gpu0" : "gpu1";
  const cached = _instCache[gpuKey];
  renderInstanceCard(elId, label, {
    url:    cached.url,
    loaded: cached.loaded,
    active: activeSnapshot,
  });
}

async function refreshActivity() {
  // Full poll: fetches loaded models + active state. Updates cache.
  if (!document.getElementById("tab-overview").classList.contains("active")) return;
  try {
    const a = await api("/api/merllm/activity");
    _instCache.gpu0 = { url: a.gpu0.url, loaded: a.gpu0.loaded };
    _instCache.gpu1 = { url: a.gpu1.url, loaded: a.gpu1.loaded };
    renderInstanceCard("ollama-gpu0", "GPU 0", a.gpu0);
    renderInstanceCard("ollama-gpu1", "GPU 1", a.gpu1);
  } catch (_) {}
}

async function refresh() {
  try {
    const status = await api("/api/merllm/status");
    renderOverview(status);
    updateBadge(status);
    document.getElementById("refresh-indicator").textContent =
      "updated " + new Date().toLocaleTimeString();
  } catch (err) {
    document.getElementById("refresh-indicator").textContent = "error: " + err.message;
  }

  try {
    const metrics = await api("/api/merllm/metrics/current");
    renderMetrics(metrics);
  } catch (_) {}

  // Refresh fans tab if it is active (avoid background polling when not viewed)
  if (document.getElementById("tab-fans").classList.contains("active")) {
    loadFans().catch(() => {});
  }

  // Fault history is always refreshed — it lives on Overview
  loadFaultHistory().catch(() => {});

  // Refresh loaded models periodically (SSE only carries active state)
  try {
    const a = await api("/api/merllm/activity");
    _instCache.gpu0 = { url: a.gpu0.url, loaded: a.gpu0.loaded };
    _instCache.gpu1 = { url: a.gpu1.url, loaded: a.gpu1.loaded };
    // Only re-render if SSE is not driving updates (avoid fighting the stream)
    if (!_activityEventSrc) {
      renderInstanceCard("ollama-gpu0", "GPU 0", a.gpu0);
      renderInstanceCard("ollama-gpu1", "GPU 1", a.gpu1);
    }
  } catch (_) {}
}

// ── Overview ──────────────────────────────────────────────────────────────────

function updateBadge(status) {
  const el = document.getElementById("routing-badge");
  if (!el) return;
  const gpus = status.gpus || {};
  const allHealthy = Object.values(gpus).every(g => g.health === "healthy");
  if (allHealthy) {
    el.textContent = "round robin";
    el.className = "badge-routing badge-ok";
  } else {
    const faulted = Object.values(gpus).some(g => g.health === "faulted");
    el.textContent = faulted ? "faulted" : "degraded";
    el.className = faulted ? "badge-routing badge-fault" : "badge-routing badge-degraded";
  }
}

function renderOverview(s) {
  // Warnings
  const warn = document.getElementById("warnings");
  if (s.warnings && s.warnings.length) {
    warn.style.display = "block";
    warn.innerHTML = "<strong>Warnings</strong><ul>" +
      s.warnings.map(w => `<li>${esc(w)}</li>`).join("") + "</ul>";
  } else {
    warn.style.display = "none";
  }

  // GPU Router card
  set("ov-routing", s.routing || "round_robin");
  set("ov-default-model", s.default_model || "—");
  const q = s.queue || {};
  const qTotal = (q.total || 0) + (q.in_flight || 0);
  set("ov-queue", s.queue ? qTotal : "—");

  // Pending model change
  const pendingRow = document.getElementById("ov-pending-model-row");
  if (s.pending_default_model) {
    pendingRow.style.display = "";
    set("ov-pending-model", s.pending_default_model);
  } else {
    pendingRow.style.display = "none";
  }

  // Per-GPU status cards inside router card
  const routerCards = document.getElementById("gpu-router-cards");
  if (s.gpus && typeof s.gpus === "object") {
    routerCards.innerHTML = Object.entries(s.gpus).map(([label, g]) => {
      const healthClass = g.health === "healthy" ? "dot-ok"
        : g.health === "faulted" ? "dot-fail" : "dot-warn";
      const idleFmt = g.idle_seconds >= 60
        ? Math.floor(g.idle_seconds / 60) + "m " + Math.round(g.idle_seconds % 60) + "s"
        : Math.round(g.idle_seconds) + "s";
      const resetBtn = g.health !== "healthy"
        ? `<button class="btn" style="padding:2px 8px;font-size:11px;margin-left:8px" onclick="resetGpu('${esc(label)}')">Reset</button>`
        : "";
      return `<div class="gpu-router-gpu" style="margin-top:10px;padding:8px;border:1px solid var(--border);border-radius:6px">
        <div class="stat-row">
          <span class="stat-label"><span class="dot ${healthClass}"></span> ${esc(label.toUpperCase().replace("GPU", "GPU "))}</span>
          <span class="stat-value">${esc(g.health)}${resetBtn}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Model</span>
          <span class="stat-value">${esc(g.model)}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Idle</span>
          <span class="stat-value">${idleFmt}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">In flight</span>
          <span class="stat-value">${g.in_flight ? "yes" : "no"}</span>
        </div>
      </div>`;
    }).join("");
  }

  // Ollama health dots (just up/down; detail comes from refreshActivity)
  if (s.ollama) {
    for (const [key, v] of Object.entries(s.ollama)) {
      const el = document.getElementById(`ollama-${key}`);
      if (el && el.dataset.activityReady !== "1") {
        el.innerHTML =
          `<div class="instance-header">
             <span class="dot ${v.ok ? "dot-ok" : "dot-fail"}"></span>
             <span class="instance-label">${esc(key.replace("gpu", "GPU "))}</span>
             <span class="muted instance-url">${esc(v.url)}</span>
           </div>`;
      }
    }
  }

  // Batch counts
  if (s.batch_counts) {
    const bc = document.getElementById("batch-counts");
    bc.innerHTML = Object.entries(s.batch_counts).map(([k, v]) =>
      `<div class="stat-row"><span class="stat-label">${esc(k)}</span><span class="stat-value">${v}</span></div>`
    ).join("");
  }

  // GPU hardware metrics
  const gpuEl = document.getElementById("gpu-status");
  if (s.gpu_metrics && s.gpu_metrics.length) {
    gpuEl.innerHTML = s.gpu_metrics.map((g, i) => gpuCard(i, g)).join("");
  } else {
    gpuEl.innerHTML = '<span class="muted">No GPU data</span>';
  }
}

function gpuCard(i, g) {
  const util = g.utilization_pct ?? 0;
  const memPct = g.mem_total_mb ? (g.mem_used_mb / g.mem_total_mb * 100) : 0;
  return `
    <div style="margin-bottom:10px">
      <div class="stat-row">
        <span class="stat-label">GPU ${i} — ${esc(g.name || "unknown")}</span>
        <span class="stat-value">${g.temp_c != null ? g.temp_c + "°C" : "—"}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Utilization</span>
        <span class="bar-wrap">
          ${bar(util)}<span style="font-size:11px;width:32px;text-align:right">${Math.round(util)}%</span>
        </span>
      </div>
      <div class="stat-row">
        <span class="stat-label">VRAM ${g.mem_used_mb != null ? Math.round(g.mem_used_mb) + "/" + Math.round(g.mem_total_mb) + " MB" : "—"}</span>
        <span class="bar-wrap">
          ${bar(memPct)}<span style="font-size:11px;width:32px;text-align:right">${Math.round(memPct)}%</span>
        </span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Power</span>
        <span class="stat-value">${g.power_w != null ? g.power_w + " W" : "—"}</span>
      </div>
    </div>`;
}

function renderInstanceCard(elId, label, inst) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.dataset.activityReady = "1";

  const reachable = inst.loaded !== null && inst.loaded !== undefined;
  const dot = reachable ? "dot-ok" : "dot-fail";

  // Loaded models section
  let loadedHtml = "";
  if (!reachable) {
    loadedHtml = `<span class="muted instance-idle">unreachable</span>`;
  } else if (inst.loaded && inst.loaded.length) {
    loadedHtml = inst.loaded.map(m =>
      `<div class="instance-model">
         <span class="instance-model-name">${esc(m.name)}</span>
         <span class="muted">${m.size_vram_mb ? m.size_vram_mb.toLocaleString() + " MB VRAM" : ""}</span>
       </div>`
    ).join("");
  } else {
    loadedHtml = `<span class="muted instance-idle">no model loaded</span>`;
  }

  // Active request section
  let activeHtml = "";
  const act = inst.active;
  if (act) {
    const endpoint   = act.endpoint.replace("/api/", "");
    const elapsed    = act.elapsed_sec;
    const elapsedFmt = elapsed >= 60
      ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
      : `${elapsed}s`;
    const isText     = endpoint !== "embeddings";
    const chunksLabel = isText
      ? `<span class="instance-chunks">${act.chunks} tok</span>`
      : "";
    const textPreview = isText && act.text
      ? `<div class="instance-text-preview">${esc(act.text)}</div>`
      : "";
    activeHtml = `
      <div class="instance-active">
        <span class="instance-active-dot"></span>
        <span class="instance-active-model">${esc(act.model)}</span>
        <span class="instance-active-ep muted">${esc(endpoint)}</span>
        <span class="instance-elapsed muted">${elapsedFmt}</span>
        ${chunksLabel}
      </div>
      ${textPreview}`;
  } else if (reachable) {
    activeHtml = `<span class="muted instance-idle">idle</span>`;
  }

  el.innerHTML = `
    <div class="instance-header">
      <span class="dot ${dot}"></span>
      <span class="instance-label">${esc(label)}</span>
      <span class="muted instance-url">${esc(inst.url)}</span>
    </div>
    <div class="instance-loaded">${loadedHtml}</div>
    ${activeHtml ? `<div class="instance-active-wrap">${activeHtml}</div>` : ""}`;
}

// ── GPU control ──────────────────────────────────────────────────────────────

async function resetGpu(gpu) {
  if (!confirm(`Reset ${gpu.toUpperCase().replace("GPU", "GPU ")} to healthy and reload default model?`)) return;
  try {
    await post(`/api/merllm/gpu/${gpu}/reset`, {});
    setTimeout(refresh, 500);
  } catch (err) {
    alert("GPU reset failed: " + err.message);
  }
}

// ── Batch jobs ────────────────────────────────────────────────────────────────

async function loadBatchJobs() {
  try {
    const jobs = await api("/api/batch/status");
    const tbody = document.getElementById("batch-tbody");
    if (!jobs || !jobs.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted">No jobs.</td></tr>';
      return;
    }
    tbody.innerHTML = jobs.map(j => `
      <tr>
        <td class="mono" style="font-size:11px">${j.id.substring(0, 8)}&hellip;</td>
        <td>${esc(j.source_app)}</td>
        <td>${esc(j.model)}</td>
        <td><span class="status-pill pill-${j.status}">${j.status}</span></td>
        <td class="muted" style="font-size:11px">${fmtTs(j.submitted_at)}</td>
        <td>
          ${j.status === "queued" ? `<button class="btn" style="padding:2px 8px;font-size:11px" onclick="cancelJob('${j.id}')">Cancel</button>` : ""}
          ${j.status === "failed" ? `<button class="btn" style="padding:2px 8px;font-size:11px" onclick="requeueJob('${j.id}')">Requeue</button>` : ""}
          ${j.status === "completed" ? `<button class="btn" style="padding:2px 8px;font-size:11px" onclick="viewResult('${j.id}')">Result</button>` : ""}
        </td>
      </tr>`).join("");
  } catch (err) {
    console.error("Batch load error:", err);
  }
}

async function submitBatchJob() {
  const prompt = document.getElementById("b-prompt").value.trim();
  if (!prompt) { alert("Prompt is required."); return; }
  const body = {
    source_app: document.getElementById("b-source").value || "merllm-ui",
    prompt,
  };
  const model = document.getElementById("b-model").value.trim();
  if (model) body.model = model;
  try {
    const r = await post("/api/batch/submit", body);
    document.getElementById("b-result").textContent = "Submitted: " + r.id;
    document.getElementById("b-prompt").value = "";
    loadBatchJobs();
  } catch (err) {
    document.getElementById("b-result").textContent = "Error: " + err.message;
  }
}

async function cancelJob(id) {
  try { await post(`/api/batch/${id}/cancel`, {}); loadBatchJobs(); }
  catch (err) { alert("Cancel failed: " + err.message); }
}

async function requeueJob(id) {
  try { await post(`/api/batch/${id}/requeue`, {}); loadBatchJobs(); }
  catch (err) { alert("Requeue failed: " + err.message); }
}

async function viewResult(id) {
  try {
    const r = await api(`/api/batch/results/${id}`);
    alert(r.result);
  } catch (err) { alert("Error: " + err.message); }
}

async function batchRetryFailed() {
  try {
    const r = await post("/api/batch/retry-failed", {});
    loadBatchJobs();
    if (r.requeued === 0) alert("No failed jobs to retry.");
  } catch (err) { alert("Retry failed: " + err.message); }
}

async function batchDrain() {
  if (!confirm("Cancel all queued jobs?")) return;
  try {
    const r = await post("/api/batch/drain", {});
    loadBatchJobs();
    alert(`Drained ${r.cancelled} queued job(s).`);
  } catch (err) { alert("Drain failed: " + err.message); }
}

async function batchClearCompleted() {
  if (!confirm("Delete all completed and cancelled jobs?")) return;
  try {
    const r = await del("/api/batch/completed");
    loadBatchJobs();
    alert(`Deleted ${r.deleted} job(s).`);
  } catch (err) { alert("Clear failed: " + err.message); }
}

// ── Metrics ───────────────────────────────────────────────────────────────────

function renderMetrics(m) {
  if (!m || !Object.keys(m).length) return;

  // CPU — sort numerically so core10 doesn't come before core2
  const cpuKeys = Object.keys(m)
    .filter(k => k.startsWith("cpu.core"))
    .sort((a, b) => parseInt(a.replace("cpu.core", "")) - parseInt(b.replace("cpu.core", "")));
  const cpuEl = document.getElementById("cpu-metrics");
  if (cpuKeys.length) {
    const avg = cpuKeys.reduce((s, k) => s + (m[k]?.value ?? 0), 0) / cpuKeys.length;
    const cells = cpuKeys.map((k, i) => {
      const pct = m[k]?.value ?? 0;
      const col = pct > 90 ? "var(--red)" : pct > 70 ? "var(--yellow)" : pct > 30 ? "var(--accent)" : "var(--green)";
      return `<div class="cpu-cell" title="core${i}: ${pct.toFixed(1)}%" style="background:${col};opacity:${0.2 + pct / 125}"></div>`;
    }).join("");
    cpuEl.innerHTML = `
      <div class="stat-row" style="margin-bottom:8px"><span class="stat-label">Avg utilization</span>
        <span class="bar-wrap">${bar(avg)}<span style="width:36px;font-size:11px;text-align:right">${avg.toFixed(1)}%</span></span>
      </div>
      <div class="stat-row" style="margin-bottom:8px"><span class="stat-label">${cpuKeys.length} cores</span>
        <span class="muted" style="font-size:11px">&#9632; &lt;30% &nbsp; &#9632; 30–70% &nbsp; &#9632; 70–90% &nbsp; &#9632; &gt;90%</span>
      </div>
      <div class="cpu-grid">${cells}</div>`;
  }

  // Memory
  const ramT = m["ram.total"]?.value;
  const ramA = m["ram.available"]?.value;
  const ramU = ramT && ramA ? ramT - ramA : null;
  const ramPct = ramT && ramU ? ramU / ramT * 100 : null;
  const swapU = m["swap.used"]?.value;
  const swapT = m["swap.total"]?.value;
  const memEl = document.getElementById("mem-metrics");
  memEl.innerHTML = `
    <div class="stat-row"><span class="stat-label">RAM used</span>
      <span class="stat-value">${ramU != null ? (ramU / 1024**3).toFixed(1) + " / " + (ramT / 1024**3).toFixed(1) + " GB" : "—"}</span></div>
    <div class="stat-row"><span class="stat-label">RAM %</span>
      <span class="bar-wrap">${ramPct != null ? bar(ramPct) : "—"}
        <span style="width:36px;font-size:11px;text-align:right">${ramPct != null ? ramPct.toFixed(1) + "%" : ""}</span></span></div>
    <div class="stat-row"><span class="stat-label">Swap</span>
      <span class="stat-value">${swapU != null ? (swapU / 1024**3).toFixed(2) + " / " + (swapT / 1024**3).toFixed(1) + " GB" : "—"}</span></div>`;

  // Disk
  const diskEl = document.getElementById("disk-metrics");
  const diskKeys = Object.keys(m).filter(k => k.startsWith("disk.") && k.endsWith(".pct"));
  diskEl.innerHTML = diskKeys.length
    ? diskKeys.map(k => {
        const mount = k.replace("disk.", "").replace(".pct", "");
        const pct = m[k]?.value ?? 0;
        return `<div class="stat-row"><span class="stat-label">${esc(mount)}</span>
          <span class="bar-wrap">${bar(pct)}<span style="width:36px;font-size:11px;text-align:right">${pct.toFixed(1)}%</span></span></div>`;
      }).join("")
    : '<span class="muted">No disk data</span>';

  // GPU panels (reuse gpu-status data already rendered in overview but also show here)
  for (const idx of [0, 1]) {
    const elId = `gpu${idx}-metrics`;
    const el = document.getElementById(elId);
    if (!el) continue;
    const util = m[`gpu${idx}.util_pct`]?.value;
    const memU = m[`gpu${idx}.mem_used_mb`]?.value;
    const memT = m[`gpu${idx}.mem_total_mb`]?.value;
    const temp = m[`gpu${idx}.temp_c`]?.value;
    const pwr  = m[`gpu${idx}.power_w`]?.value;
    const memPct = memT && memU ? memU / memT * 100 : null;
    el.innerHTML = `
      <div class="stat-row"><span class="stat-label">Utilization</span>
        <span class="bar-wrap">${util != null ? bar(util) : "—"}
          <span style="width:36px;font-size:11px;text-align:right">${util != null ? util.toFixed(1) + "%" : ""}</span></span></div>
      <div class="stat-row"><span class="stat-label">VRAM</span>
        <span class="stat-value">${memU != null ? Math.round(memU) + " / " + Math.round(memT) + " MB" : "—"}</span></div>
      <div class="stat-row"><span class="stat-label">VRAM %</span>
        <span class="bar-wrap">${memPct != null ? bar(memPct) : "—"}
          <span style="width:36px;font-size:11px;text-align:right">${memPct != null ? memPct.toFixed(1) + "%" : ""}</span></span></div>
      <div class="stat-row"><span class="stat-label">Temp</span>
        <span class="stat-value">${temp != null ? temp + " °C" : "—"}</span></div>
      <div class="stat-row"><span class="stat-label">Power</span>
        <span class="stat-value">${pwr != null ? pwr + " W" : "—"}</span></div>`;
  }

  // Network
  const rxKey = Object.keys(m).find(k => k.endsWith(".rx_bps"));
  const txKey = Object.keys(m).find(k => k.endsWith(".tx_bps"));
  const netEl = document.getElementById("net-metrics");
  netEl.innerHTML = `
    <div class="stat-row"><span class="stat-label">RX</span>
      <span class="stat-value">${rxKey ? fmtBytes(m[rxKey]?.value) + "/s" : "—"}</span></div>
    <div class="stat-row"><span class="stat-label">TX</span>
      <span class="stat-value">${txKey ? fmtBytes(m[txKey]?.value) + "/s" : "—"}</span></div>`;

  // System card in overview
  const sysEl = document.getElementById("system-metrics");
  sysEl.innerHTML = `
    <div class="stat-row"><span class="stat-label">RAM used</span>
      <span class="stat-value">${ramU != null ? (ramU / 1024**3).toFixed(1) + " GB" : "—"}</span></div>
    <div class="stat-row"><span class="stat-label">CPU avg</span>
      <span class="stat-value">${cpuKeys.length ? ((cpuKeys.reduce((s,k) => s + (m[k]?.value ?? 0), 0) / cpuKeys.length).toFixed(1) + "%") : "—"}</span></div>
    <div class="stat-row"><span class="stat-label">RX</span>
      <span class="stat-value">${rxKey ? fmtBytes(m[rxKey]?.value) + "/s" : "—"}</span></div>
    <div class="stat-row"><span class="stat-label">TX</span>
      <span class="stat-value">${txKey ? fmtBytes(m[txKey]?.value) + "/s" : "—"}</span></div>`;
}

// ── Metrics Charts ────────────────────────────────────────────────────────────

const _charts = {};
let _metricsRange = "1h";

function _chartScaleDefaults() {
  return {
    x: {
      ticks: { color: "#8b949e", font: { size: 10 }, maxTicksLimit: 8, maxRotation: 0 },
      grid: { color: "#21262d" },
    },
    y: {
      ticks: { color: "#8b949e", font: { size: 10 } },
      grid: { color: "#21262d" },
    },
  };
}

function _chartOptions(extraScales) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: "#8b949e", font: { size: 11 }, boxWidth: 10, padding: 8 },
      },
      tooltip: {
        backgroundColor: "#161b22",
        borderColor: "#30363d",
        borderWidth: 1,
        titleColor: "#e6edf3",
        bodyColor: "#8b949e",
      },
    },
    scales: { ..._chartScaleDefaults(), ...(extraScales || {}) },
  };
}

function _mkDs(label, data, color, yAxisID) {
  const ds = {
    label,
    data,
    borderColor: color,
    backgroundColor: color + "18",
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  };
  if (yAxisID) ds.yAxisID = yAxisID;
  return ds;
}

function _fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function _setChart(id, labels, datasets, extraScales) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  if (_charts[id]) {
    _charts[id].data.labels = labels;
    _charts[id].data.datasets = datasets;
    _charts[id].update("none");
  } else {
    _charts[id] = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: _chartOptions(extraScales),
    });
  }
}

async function _hist(metric) {
  try {
    return await api(`/api/merllm/metrics/history?metric=${encodeURIComponent(metric)}&range=${_metricsRange}`);
  } catch (_) {
    return [];
  }
}

async function loadMetricsCharts() {
  const [cpu, ramUsed, diskPct, diskArchivePct, g0util, g0mem, g0temp, g1util, g1mem, g1temp, tx, rx] =
    await Promise.all([
      _hist("cpu.total"),
      _hist("ram.used"),
      _hist("disk.root.pct"),
      _hist("disk.archive.pct"),
      _hist("gpu0.util_pct"),
      _hist("gpu0.mem_used_mb"),
      _hist("gpu0.temp_c"),
      _hist("gpu1.util_pct"),
      _hist("gpu1.mem_used_mb"),
      _hist("gpu1.temp_c"),
      _hist("net.tx_bps"),
      _hist("net.rx_bps"),
    ]);

  if (cpu.length) {
    _setChart("chart-cpu",
      cpu.map(p => _fmtTs(p.ts)),
      [_mkDs("CPU %", cpu.map(p => p.value.toFixed(1)), "#58a6ff")]
    );
  }

  if (ramUsed.length) {
    _setChart("chart-mem",
      ramUsed.map(p => _fmtTs(p.ts)),
      [_mkDs("RAM GB", ramUsed.map(p => (p.value / 1073741824).toFixed(2)), "#3fb950")]
    );
  }

  if (diskPct.length || diskArchivePct.length) {
    const base = diskPct.length ? diskPct : diskArchivePct;
    const datasets = [];
    if (diskPct.length) datasets.push(_mkDs("OS Disk %", diskPct.map(p => p.value.toFixed(1)), "#d29922"));
    if (diskArchivePct.length) datasets.push(_mkDs("Archive %", diskArchivePct.map(p => p.value.toFixed(1)), "#58a6ff"));
    _setChart("chart-disk",
      base.map(p => _fmtTs(p.ts)),
      datasets
    );
  }

  // GPU charts: util+temp on left y, VRAM MB on right y
  for (const [idx, util, mem, temp] of [
    [0, g0util, g0mem, g0temp],
    [1, g1util, g1mem, g1temp],
  ]) {
    const base = util.length ? util : mem.length ? mem : temp;
    if (!base.length) continue;
    _setChart(`chart-gpu${idx}`,
      base.map(p => _fmtTs(p.ts)),
      [
        _mkDs("Util %",  util.map(p => p.value.toFixed(1)), "#58a6ff", "y"),
        _mkDs("Temp °C", temp.map(p => p.value.toFixed(1)), "#f85149", "y"),
        _mkDs("VRAM MB", mem.map(p  => Math.round(p.value)), "#a78bfa", "yMem"),
      ],
      {
        y:    { type: "linear", position: "left",  ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: "#21262d" } },
        yMem: { type: "linear", position: "right", ticks: { color: "#a78bfa", font: { size: 10 } }, grid: { drawOnChartArea: false } },
      }
    );
  }

  const netBase = tx.length ? tx : rx;
  if (netBase.length) {
    _setChart("chart-net",
      netBase.map(p => _fmtTs(p.ts)),
      [
        _mkDs("TX MB/s", tx.map(p => (p.value / 1048576).toFixed(3)), "#58a6ff"),
        _mkDs("RX MB/s", rx.map(p => (p.value / 1048576).toFixed(3)), "#3fb950"),
      ]
    );
  }
}

// ── Logs ──────────────────────────────────────────────────────────────────────

async function fetchLogs() {
  const service = document.getElementById("log-service").value;
  const lines = document.getElementById("log-lines").value || 100;
  const el = document.getElementById("log-output");
  el.textContent = "Fetching…";
  try {
    const r = await api(`/api/merllm/logs/${service}?lines=${lines}`);
    el.textContent = r.lines.join("\n") || "(no output)";
    el.scrollTop = el.scrollHeight;
  } catch (err) {
    el.textContent = "Error: " + err.message;
  }
}

// ── SSH Terminal ──────────────────────────────────────────────────────────────

let _term = null;
let _termWs = null;

function connectSSH() {
  if (_term) { _term.dispose(); _term = null; }
  if (_termWs) { _termWs.close(); _termWs = null; }

  const container = document.getElementById("terminal-container");
  container.innerHTML = "";

  _term = new Terminal({ theme: { background: "#000" }, cursorBlink: true });
  const fitAddon = new FitAddon.FitAddon();
  _term.loadAddon(fitAddon);
  _term.open(container);
  fitAddon.fit();

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  _termWs = new WebSocket(`${proto}//${location.host}/ws/ssh`);
  _termWs.binaryType = "arraybuffer";

  _termWs.onopen = () => _term.writeln("\r\nConnected.\r\n");
  _termWs.onmessage = e => _term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data));
  _termWs.onclose = () => _term.writeln("\r\n\r\nDisconnected.");
  _termWs.onerror = () => _term.writeln("\r\nWebSocket error.");

  _term.onData(data => {
    if (_termWs && _termWs.readyState === WebSocket.OPEN) _termWs.send(data);
  });

  window.addEventListener("resize", () => { try { fitAddon.fit(); } catch (_) {} });
}

function disconnectSSH() {
  if (_termWs) { _termWs.close(); _termWs = null; }
  if (_term) { _term.dispose(); _term = null; }
  document.getElementById("terminal-container").innerHTML = "";
}

// ── VNC ───────────────────────────────────────────────────────────────────────

let _vncWs = null;
let _vncSocket = null;

async function connectVNC() {
  const container = document.getElementById("vnc-container");
  container.innerHTML = '<span class="muted">Loading noVNC…</span>';
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/vnc`;

  let RFB;
  try {
    const mod = await import('/novnc/core/rfb.js');
    RFB = mod.default;
  } catch (err) {
    container.innerHTML = `<span class="muted">noVNC load error: ${esc(String(err))}</span>`;
    return;
  }

  container.innerHTML = '<canvas id="vnc-canvas"></canvas>';
  try {
    const rfb = new RFB(container, url);
    rfb.scaleViewport = true;
    rfb.addEventListener("disconnect", () => {
      container.innerHTML = '<span class="muted">VNC disconnected.</span>';
    });
  } catch (err) {
    container.innerHTML = `<span class="muted">VNC error: ${esc(err.message)}</span>`;
  }
}

function disconnectVNC() {
  document.getElementById("vnc-container").innerHTML =
    '<span class="muted">VNC not connected.</span>';
}

// ── Settings ──────────────────────────────────────────────────────────────────

const SETTINGS_FIELDS = [
  { key: "ollama_0_url",             label: "Ollama GPU 0 URL",           type: "text" },
  { key: "ollama_1_url",             label: "Ollama GPU 1 URL",           type: "text" },
  { key: "default_model",            label: "Default model",              type: "text",
    hint: "Changing this will reload both GPUs when they go idle." },
  { key: "reclaim_timeout",          label: "Reclaim timeout (sec)",      type: "number",
    hint: "Idle seconds before a non-default model GPU is reclaimed." },
  { key: "health_backoff_base",      label: "Health probe base (sec)",    type: "number" },
  { key: "health_backoff_cap",       label: "Health probe cap (sec)",     type: "number" },
  { key: "health_fault_timeout",     label: "Fault timeout (sec)",        type: "number",
    hint: "Continuous failure seconds before declaring a GPU faulted." },
  { key: "metrics_interval_sec",     label: "Metrics interval (sec)",     type: "number" },
  { key: "notification_webhook_url", label: "Notification webhook URL",   type: "text",
    hint: "POST to this URL on job completion. Supports Slack, ntfy.sh, Gotify, etc." },
];

async function loadSettings() {
  try {
    const s = await api("/api/merllm/settings");
    _lastSavedDefaultModel = s.default_model || "";
    const form = document.getElementById("settings-form");
    form.innerHTML = SETTINGS_FIELDS.map(f => {
      const val = s[f.key] != null ? s[f.key] : "";
      if (f.type === "select") {
        const opts = f.options.map(o =>
          `<option value="${o}"${o === val ? " selected" : ""}>${o}</option>`
        ).join("");
        return `<div class="form-group">
          <label>${esc(f.label)}</label>
          <select id="sf-${f.key}">${opts}</select>
        </div>`;
      }
      const hint = f.hint ? `<div class="form-hint">${esc(f.hint)}</div>` : '';
      return `<div class="form-group">
        <label>${esc(f.label)}</label>
        <input id="sf-${f.key}" type="${f.type}" value="${esc(String(val))}" />
        ${hint}
      </div>`;
    }).join("");
  } catch (err) {
    document.getElementById("settings-msg").textContent = "Load failed: " + err.message;
  }
}

async function saveSettings() {
  const body = {};
  SETTINGS_FIELDS.forEach(f => {
    const el = document.getElementById("sf-" + f.key);
    if (!el) return;
    body[f.key] = f.type === "number" ? Number(el.value) : el.value;
  });

  // If default_model changed, require confirmation
  if (body.default_model && body.default_model !== _lastSavedDefaultModel) {
    if (!confirm(`Change default model to "${body.default_model}"?\n\nBoth GPUs will reload when they go idle.`)) {
      return;
    }
    body.confirm_model_change = true;
  }

  try {
    await post("/api/merllm/settings", body);
    _lastSavedDefaultModel = body.default_model || _lastSavedDefaultModel;
    document.getElementById("settings-msg").textContent = "Saved.";
    setTimeout(() => { document.getElementById("settings-msg").textContent = ""; }, 3000);
  } catch (err) {
    document.getElementById("settings-msg").textContent = "Save failed: " + err.message;
  }
}

let _lastSavedDefaultModel = "";

// ── My Day Panel ──────────────────────────────────────────────────────────────

async function refreshMyDay() {
  try {
    const data = await api("/api/merllm/myday");
    renderMyDay(data);
  } catch (err) {
    document.getElementById("myday-cards").innerHTML =
      `<div class="myday-card myday-err">Failed to load My Day summary: ${esc(err.message)}</div>`;
  }
}

function renderMyDay(data) {
  const cards = [];

  // Parsival card
  if (!data.parsival.ok) {
    cards.push(`<div class="myday-card myday-offline">
      <div class="myday-app">Parsival</div>
      <div class="myday-offline-msg">offline</div>
    </div>`);
  } else {
    const p = data.parsival;
    const hasSituations = p.active_situations > 0;
    const hasOverdue    = p.overdue_followups > 0;
    const cls = (hasSituations || hasOverdue) ? "myday-card myday-attention" : "myday-card myday-ok";
    cards.push(`<a class="myday-link" href="/page/index.html" target="_blank" rel="noopener">
      <div class="${cls}">
        <div class="myday-app">Parsival</div>
        <div class="myday-rows">
          <div class="myday-row">
            <span class="myday-label">Active situations</span>
            <span class="myday-val${hasSituations ? ' myday-highlight' : ''}">${p.active_situations}</span>
          </div>
          <div class="myday-row">
            <span class="myday-label">New / investigating</span>
            <span class="myday-val">${p.new_investigating}</span>
          </div>
          <div class="myday-row">
            <span class="myday-label">Overdue follow-ups</span>
            <span class="myday-val${hasOverdue ? ' myday-highlight' : ''}">${p.overdue_followups}</span>
          </div>
          ${p.cold_start ? '<div class="myday-hint">Learning attention patterns…</div>' : ''}
        </div>
      </div>
    </a>`);
  }

  // LanceLLMot card
  if (!data.lancellmot.ok) {
    cards.push(`<div class="myday-card myday-offline">
      <div class="myday-app">LanceLLMot</div>
      <div class="myday-offline-msg">offline</div>
    </div>`);
  } else {
    const l = data.lancellmot;
    const hasPending = l.total_pending > 0;
    const cls = hasPending ? "myday-card myday-attention" : "myday-card myday-ok";
    cards.push(`<a class="myday-link" href="/web/index.html" target="_blank" rel="noopener">
      <div class="${cls}">
        <div class="myday-app">LanceLLMot</div>
        <div class="myday-rows">
          <div class="myday-row">
            <span class="myday-label">Acquisition pending</span>
            <span class="myday-val${l.acquisition_pending > 0 ? ' myday-highlight' : ''}">${l.acquisition_pending}</span>
          </div>
          <div class="myday-row">
            <span class="myday-label">Escalation pending</span>
            <span class="myday-val${l.escalation_pending > 0 ? ' myday-highlight' : ''}">${l.escalation_pending}</span>
          </div>
        </div>
      </div>
    </a>`);
  }

  // merLLM card
  const m = data.merllm;
  const hasJobs = m.queued_jobs > 0 || m.failed_jobs > 0;
  const mCls = hasJobs ? "myday-card myday-attention" : "myday-card myday-ok";
  cards.push(`<div class="${mCls}">
    <div class="myday-app">merLLM</div>
    <div class="myday-rows">
      <div class="myday-row">
        <span class="myday-label">Queued batch jobs</span>
        <span class="myday-val${m.queued_jobs > 0 ? ' myday-highlight' : ''}">${m.queued_jobs}</span>
      </div>
      <div class="myday-row">
        <span class="myday-label">Completed (unreviewed)</span>
        <span class="myday-val">${m.completed_jobs}</span>
      </div>
      <div class="myday-row">
        <span class="myday-label">Failed jobs</span>
        <span class="myday-val${m.failed_jobs > 0 ? ' myday-highlight' : ''}">${m.failed_jobs}</span>
      </div>
    </div>
  </div>`);

  document.getElementById("myday-cards").innerHTML = cards.join("");
}

// ── Fan Controller ────────────────────────────────────────────────────────────

const FAN_SETTINGS_FIELDS = [
  { key: "FAN_SPEED",                 label: "Min fan speed %",        min: 0,  max: 100 },
  { key: "FAN_SPEED_MAX",             label: "Max fan speed %",        min: 0,  max: 100 },
  { key: "FAN_RAMP_STEP",             label: "Ramp step % per cycle",  min: 1,  max: 50  },
  { key: "TEMP_RAMP_RANGE",           label: "Ramp range (°C)",        min: 1,  max: 40  },
  { key: "CPU_TEMPERATURE_THRESHOLD", label: "CPU threshold (°C)",     min: 30, max: 100 },
  { key: "GPU_TEMPERATURE_THRESHOLD", label: "GPU threshold (°C)",     min: 40, max: 100 },
  { key: "CHECK_INTERVAL",            label: "Check interval (sec)",   min: 1,  max: 60  },
];

async function loadFans() {
  const [fanStatus, fanSettings] = await Promise.all([
    api("/api/merllm/fans/status").catch(() => null),
    api("/api/merllm/fans/settings").catch(() => ({})),
  ]);
  renderFansStatus(fanStatus);
  renderFansControl(fanStatus, fanSettings);
}

async function loadFaultHistory() {
  const el = document.getElementById("fans-fault-content");
  try {
    const faults = await api("/api/merllm/fans/faults?limit=50");
    renderFaultHistory(el, faults);
  } catch (err) {
    el.innerHTML = `<span class="muted">Could not load fault history: ${esc(err.message)}</span>`;
  }
}

function renderFaultHistory(el, faults) {
  if (!faults || faults.length === 0) {
    el.innerHTML = `<span class="muted">No fault events recorded.</span>`;
    return;
  }

  const TYPE_STYLE = {
    gpu_fault_onset:   { color: "var(--red)",    icon: "&#9888;" },
    gpu_fault_cleared: { color: "var(--green)",  icon: "&#10003;" },
    idrac_read_fault:  { color: "var(--red)",    icon: "&#9888;" },
    default:           { color: "var(--yellow)", icon: "&#9432;" },
  };

  const rows = faults.map(f => {
    const style = TYPE_STYLE[f.event_type] || TYPE_STYLE.default;
    const ts    = f.ts ? new Date(f.ts * 1000).toLocaleString() : "—";
    const speed = f.fan_speed_applied != null
      ? `<span class="muted" style="font-size:11px">&nbsp;(fallback ${f.fan_speed_applied}%)</span>`
      : "";
    return `<tr>
      <td style="white-space:nowrap;color:var(--muted);font-size:11px">${esc(ts)}</td>
      <td style="color:${style.color};font-size:12px">${style.icon}&nbsp;${esc(f.event_type)}</td>
      <td style="font-size:12px">${esc(f.message)}${speed}</td>
    </tr>`;
  }).join("");

  el.innerHTML = `
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>Time</th><th>Type</th><th>Message</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="muted" style="font-size:11px;margin-top:8px">Showing last ${faults.length} event${faults.length !== 1 ? "s" : ""}. Stored in merLLM DB.</div>`;
}

function renderFansStatus(s) {
  const el = document.getElementById("fans-status-content");
  if (!s || s.error) {
    el.innerHTML = `<span class="muted">Fan controller unavailable${s && s.error ? ": " + esc(s.error) : ""}</span>`;
    return;
  }

  const t   = s.temperatures || {};
  const f   = s.fan          || {};
  const th  = s.thresholds   || {};
  const cpuTh  = th.cpu  || 50;
  const gpuTh  = th.gpu  || 80;
  const rampC  = th.ramp_range_c || 15;

  function tempRow(label, tempC, threshold) {
    const pct = Math.min(100, tempC / threshold * 100);
    return `<div class="stat-row">
      <span class="stat-label">${label}</span>
      <span class="bar-wrap">${bar(pct)}<span style="width:40px;font-size:11px;text-align:right">${tempC}°C</span></span>
    </div>`;
  }

  const cpuRows = (t.cpu || []).map((c, i) => tempRow(`CPU ${i + 1}`, c, cpuTh)).join("");
  const gpuRows = (t.gpu || []).map((g, i) => tempRow(`GPU ${i + 1}`, g, gpuTh)).join("");

  const fanPct    = f.current_speed_pct ?? 0;
  const targetPct = f.target_speed_pct  ?? 0;
  const safetyBadge = f.safety_override
    ? `<span style="color:var(--red);font-weight:600">&#9888; SAFETY — Dell default</span>`
    : `<span style="color:var(--green)">Manual ramp</span>`;

  el.innerHTML = `
    <div style="margin-bottom:14px">
      <div class="stat-row">
        <span class="stat-label">Inlet</span>
        <span class="stat-value">${t.inlet != null ? t.inlet + "°C" : "—"}</span>
      </div>
      ${cpuRows}
      ${gpuRows}
      <div class="stat-row">
        <span class="stat-label">Exhaust</span>
        <span class="stat-value">${t.exhaust != null ? t.exhaust + "°C" : "—"}</span>
      </div>
    </div>
    <hr style="border-color:var(--border);margin:12px 0">
    <div class="stat-row">
      <span class="stat-label">Fan speed</span>
      <span class="bar-wrap">${bar(fanPct)}<span style="width:36px;font-size:11px;text-align:right">${fanPct}%</span></span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Target</span>
      <span class="stat-value">${targetPct}%</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Profile</span>
      <span class="muted" style="font-size:11px">${esc(f.profile || "—")}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Control</span>
      ${safetyBadge}
    </div>
    <div class="muted" style="font-size:11px;margin-top:10px">
      Ramp: ${th.fan_speed_min_pct ?? "?"}% → ${th.fan_speed_max_pct ?? "?"}%
      over ${rampC}°C below CPU ${cpuTh}°C / GPU ${gpuTh}°C threshold
    </div>
    <div class="muted" style="font-size:10px;margin-top:4px">${esc(s.timestamp || "")}</div>`;
}

function renderFansControl(status, settings) {
  const el = document.getElementById("fans-control-content");
  const th = (status || {}).thresholds || {};

  // Show active override values; fall back to values reported by the controller
  const current = {
    FAN_SPEED:                 settings.FAN_SPEED                 ?? th.fan_speed_min_pct  ?? "",
    FAN_SPEED_MAX:             settings.FAN_SPEED_MAX             ?? th.fan_speed_max_pct  ?? "",
    FAN_RAMP_STEP:             settings.FAN_RAMP_STEP             ?? th.fan_ramp_step_pct  ?? "",
    TEMP_RAMP_RANGE:           settings.TEMP_RAMP_RANGE           ?? th.ramp_range_c       ?? "",
    CPU_TEMPERATURE_THRESHOLD: settings.CPU_TEMPERATURE_THRESHOLD ?? th.cpu                ?? "",
    GPU_TEMPERATURE_THRESHOLD: settings.GPU_TEMPERATURE_THRESHOLD ?? th.gpu                ?? "",
    CHECK_INTERVAL:            settings.CHECK_INTERVAL            ?? th.check_interval_sec ?? "",
  };

  const hasOverrides = Object.keys(settings).length > 0;
  const overrideBanner = hasOverrides
    ? `<div style="margin-bottom:10px;padding:6px 10px;background:rgba(251,191,36,0.1);border:1px solid var(--yellow);border-radius:6px;font-size:11px;color:var(--yellow)">
         Overrides active — values differ from container defaults.
       </div>`
    : "";

  const fields = FAN_SETTINGS_FIELDS.map(f => `
    <div class="form-group">
      <label>${esc(f.label)}</label>
      <input id="fan-${f.key}" type="number" min="${f.min}" max="${f.max}" value="${esc(String(current[f.key]))}" />
    </div>`).join("");

  el.innerHTML = `
    ${overrideBanner}
    ${fields}
    <div class="settings-actions" style="margin-top:12px">
      <button class="btn primary" onclick="saveFanSettings()">Save</button>
      <button class="btn" onclick="resetFanSettings()">Reset to Defaults</button>
      <span id="fan-settings-msg" class="muted" style="font-size:12px;align-self:center"></span>
    </div>
    <div style="margin-top:16px;padding:10px 12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;font-size:11px;color:var(--muted);line-height:1.8">
      <strong style="color:var(--text)">How the ramp works</strong><br>
      Below (threshold &minus; range): fans hold at min %<br>
      Within range of threshold: linear ramp → max %<br>
      At or above threshold: Dell default control (safety)
    </div>`;
}

async function saveFanSettings() {
  const body = {};
  FAN_SETTINGS_FIELDS.forEach(f => {
    const el = document.getElementById("fan-" + f.key);
    if (el && el.value !== "") body[f.key] = Number(el.value);
  });
  const msg = document.getElementById("fan-settings-msg");
  try {
    await post("/api/merllm/fans/settings", body);
    msg.textContent = "Saved.";
    setTimeout(() => { msg.textContent = ""; }, 3000);
    loadFans().catch(() => {});
  } catch (err) {
    msg.textContent = "Error: " + err.message;
  }
}

async function resetFanSettings() {
  const msg = document.getElementById("fan-settings-msg");
  try {
    await api("/api/merllm/fans/settings", { method: "DELETE" });
    msg.textContent = "Reset to defaults.";
    setTimeout(() => { msg.textContent = ""; }, 3000);
    loadFans().catch(() => {});
  } catch (err) {
    msg.textContent = "Error: " + err.message;
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? "—";
}

function bar(pct) {
  const cls = pct > 90 ? "crit" : pct > 70 ? "warn" : "";
  return `<div class="bar-bg"><div class="bar-fill ${cls}" style="width:${Math.min(100, pct).toFixed(1)}%"></div></div>`;
}

function relTime(ts) {
  if (!ts) return "never";
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 60) return diff + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  return Math.floor(diff / 3600) + "h ago";
}

function fmtTs(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

function fmtBytes(bytes) {
  if (bytes == null) return "—";
  if (bytes < 1024) return bytes.toFixed(0) + " B";
  if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + " KB";
  if (bytes < 1024 ** 3) return (bytes / 1024 ** 2).toFixed(1) + " MB";
  return (bytes / 1024 ** 3).toFixed(2) + " GB";
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadSettings();
loadBatchJobs();
loadFans().catch(() => {});
startPolling();
