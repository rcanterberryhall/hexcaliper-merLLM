// merLLM dashboard client
const POLL_INTERVAL_MS = 3000;

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('queue-depth').textContent = data.queue_depth;
    const slotsEl = document.getElementById('slots');
    slotsEl.innerHTML = '';
    for (const s of data.slots) {
      const li = document.createElement('li');
      li.textContent = `Slot ${s.id}: ${s.state}`;
      slotsEl.appendChild(li);
    }
  } catch (err) {
    console.error('failed to load status', err);
  }
}

async function pollLoop() {
  await loadStatus();
  setTimeout(pollLoop, POLL_INTERVAL_MS);
}

pollLoop();
