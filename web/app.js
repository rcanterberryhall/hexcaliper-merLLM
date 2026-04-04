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

let _pollTimer     = null;
let _activityTimer = null;

function startPolling() {
  refresh();
  _pollTimer = setInterval(refresh, 10000);
  refreshActivity();
  _activityTimer = setInterval(refreshActivity, 2000);
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

  // Refresh fans tab if it is active (avoid background polling when not viewed)
  if (document.getElementById("tab-fans").classList.contains("active")) {
    loadFans().catch(() => {});
  }
}

async function refreshActivity() {
  // Only poll when Overview tab is visible
  if (!document.getElementById("tab-overview").classList.contains("active")) return;
  try {
    const a = await api("/api/merllm/activity");
    renderInstanceCard("ollama-gpu0", "GPU 0", a.gpu0);
    renderInstanceCard("ollama-gpu1", "GPU 1", a.gpu1);
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

  // Ollama health dots (just up/down; detail comes from refreshActivity)
  if (s.ollama) {
    for (const [key, v] of Object.entries(s.ollama)) {
      const el = document.getElementById(`ollama-${key}`);
      if (el && el.dataset.activityReady !== "1") {
        // Show minimal health dot until first activity poll arrives
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
    const endpoint  = act.endpoint.replace("/api/", "");
    const elapsed   = act.elapsed_sec;
    const elapsedFmt = elapsed >= 60
      ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
      : `${elapsed}s`;
    const chunksLabel = endpoint === "embeddings"
      ? ""
      : `<span class="instance-chunks">${act.chunks} chunks</span>`;
    activeHtml = `
      <div class="instance-active">
        <span class="instance-active-dot"></span>
        <span class="instance-active-model">${esc(act.model)}</span>
        <span class="instance-active-ep muted">${esc(endpoint)}</span>
        <span class="instance-elapsed muted">${elapsedFmt}</span>
        ${chunksLabel}
      </div>`;
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
