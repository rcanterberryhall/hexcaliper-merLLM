/* app.js — merLLM dashboard */

// ── Navigation ────────────────────────────────────────────────────────────────

document.querySelectorAll("nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
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

// ── Polling ───────────────────────────────────────────────────────────────────

let _pollTimer = null;

function startPolling() {
  refresh();
  _pollTimer = setInterval(refresh, 10000);
}

async function refresh() {
  try {
    const status = await api("/api/merllm/status");
    renderOverview(status);
    updateBadge(status.mode);
    document.getElementById("refresh-indicator").textContent =
      "updated " + new Date().toLocaleTimeString();
  } catch (err) {
    document.getElementById("refresh-indicator").textContent = "error: " + err.message;
  }

  try {
    const metrics = await api("/api/merllm/metrics/current");
    renderMetrics(metrics);
  } catch (_) {}
}

// ── Overview ──────────────────────────────────────────────────────────────────

function updateBadge(mode) {
  const el = document.getElementById("mode-badge");
  el.textContent = mode;
  el.className = "badge-" + mode;
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

  // Mode card
  set("ov-mode", s.mode);
  set("ov-override", s.override || "none");
  const ago = s.last_interactive_at
    ? relTime(s.last_interactive_at)
    : "never";
  set("ov-activity", ago);
  set("ov-timeout", (s.inactivity_timeout_min || "—") + " min");
  set("ov-queue", s.queue ? s.queue.total : "—");

  // Ollama
  const ol = document.getElementById("ollama-status");
  if (s.ollama) {
    ol.innerHTML = Object.entries(s.ollama).map(([k, v]) =>
      `<div class="ollama-row">
         <span class="dot ${v.ok ? "dot-ok" : "dot-fail"}"></span>
         <span>${esc(k)}</span>
         <span class="muted" style="font-size:11px;margin-left:auto">${esc(v.url)}</span>
       </div>`
    ).join("");
  }

  // Batch counts
  if (s.batch_counts) {
    const bc = document.getElementById("batch-counts");
    bc.innerHTML = Object.entries(s.batch_counts).map(([k, v]) =>
      `<div class="stat-row"><span class="stat-label">${esc(k)}</span><span class="stat-value">${v}</span></div>`
    ).join("");
  }

  // GPU summary
  const gpuEl = document.getElementById("gpu-status");
  if (s.gpus && s.gpus.length) {
    gpuEl.innerHTML = s.gpus.map((g, i) => gpuCard(i, g)).join("");
  } else {
    gpuEl.innerHTML = '<span class="muted">No GPU data</span>';
  }

  // Transitions
  const trEl = document.getElementById("transitions-list");
  if (s.last_transition && s.last_transition.length) {
    trEl.innerHTML = s.last_transition.map(t =>
      `<div class="transition-item">
         <span class="ts">${fmtTs(t.ts)}</span>
         <span>${esc(t.from_mode)} &rarr; ${esc(t.to_mode)}</span>
         <span class="muted">${esc(t.trigger || "")}</span>
       </div>`
    ).join("");
  } else {
    trEl.innerHTML = '<span class="muted" style="font-size:12px">No transitions yet.</span>';
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

// ── Mode control ──────────────────────────────────────────────────────────────

async function setMode(mode) {
  try {
    const r = await post("/api/merllm/mode", { mode });
    updateBadge(r.mode);
    setTimeout(refresh, 1000);
  } catch (err) {
    alert("Mode change failed: " + err.message);
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

// ── Metrics ───────────────────────────────────────────────────────────────────

function renderMetrics(m) {
  if (!m || !Object.keys(m).length) return;

  // CPU
  const cpuKeys = Object.keys(m).filter(k => k.startsWith("cpu.core"));
  const cpuEl = document.getElementById("cpu-metrics");
  if (cpuKeys.length) {
    const avg = cpuKeys.reduce((s, k) => s + (m[k]?.value ?? 0), 0) / cpuKeys.length;
    cpuEl.innerHTML = `
      <div class="stat-row"><span class="stat-label">Avg utilization</span>
        <span class="bar-wrap">${bar(avg)}<span style="width:36px;font-size:11px;text-align:right">${avg.toFixed(1)}%</span></span>
      </div>` +
      cpuKeys.slice(0, 8).map(k =>
        `<div class="stat-row"><span class="stat-label muted">${esc(k)}</span>
           <span class="bar-wrap">${bar(m[k]?.value ?? 0)}</span></div>`
      ).join("");
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

function connectVNC() {
  const container = document.getElementById("vnc-container");
  container.innerHTML = '<canvas id="vnc-canvas"></canvas>';
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/vnc`;

  if (typeof RFB !== "undefined") {
    try {
      const rfb = new RFB(container, url);
      rfb.scaleViewport = true;
      rfb.addEventListener("disconnect", () => {
        container.innerHTML = '<span class="muted">VNC disconnected.</span>';
      });
    } catch (err) {
      container.innerHTML = `<span class="muted">VNC error: ${esc(err.message)}</span>`;
    }
  } else {
    container.innerHTML =
      '<span class="muted">noVNC library not loaded. Add noVNC\'s core/rfb.js to your setup.</span>';
  }
}

function disconnectVNC() {
  document.getElementById("vnc-container").innerHTML =
    '<span class="muted">VNC not connected.</span>';
}

// ── Settings ──────────────────────────────────────────────────────────────────

const SETTINGS_FIELDS = [
  { key: "ollama_0_url",           label: "Ollama GPU 0 URL",         type: "text" },
  { key: "ollama_1_url",           label: "Ollama GPU 1 URL",         type: "text" },
  { key: "day_model_gpu0",         label: "Day model GPU 0",          type: "text" },
  { key: "day_model_gpu1",         label: "Day model GPU 1",          type: "text" },
  { key: "night_model",            label: "Night model",              type: "text" },
  { key: "night_num_ctx",          label: "Night num_ctx",            type: "number" },
  { key: "inactivity_timeout_min", label: "Inactivity timeout (min)", type: "number" },
  { key: "base_day_end_local",     label: "Day end (HH:MM local)",    type: "text" },
  { key: "geoip_offset_override",  label: "UTC offset override",      type: "text" },
  { key: "ollama_manage_via",      label: "Manage Ollama via",        type: "select",
    options: ["none", "systemctl"] },
  { key: "drain_timeout_sec",      label: "Drain timeout (sec)",      type: "number" },
  { key: "metrics_interval_sec",   label: "Metrics interval (sec)",   type: "number" },
];

async function loadSettings() {
  try {
    const s = await api("/api/merllm/settings");
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
      return `<div class="form-group">
        <label>${esc(f.label)}</label>
        <input id="sf-${f.key}" type="${f.type}" value="${esc(String(val))}" />
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
  try {
    await post("/api/merllm/settings", body);
    document.getElementById("settings-msg").textContent = "Saved.";
    setTimeout(() => { document.getElementById("settings-msg").textContent = ""; }, 3000);
  } catch (err) {
    document.getElementById("settings-msg").textContent = "Save failed: " + err.message;
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
startPolling();
