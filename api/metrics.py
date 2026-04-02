"""
metrics.py — System and GPU metrics collection.

Collects CPU, RAM, disk, network, and per-GPU stats on a configurable interval
and stores them in SQLite. Exposes a current snapshot and history query.

GPU metrics use pynvml via the NVML library mounted in the container.
Gracefully degrades if pynvml is unavailable or NVML initialisation fails.
"""
import asyncio
import time
from typing import Optional

import psutil

import config
import db

_nvml_ok = False
_gpu_count = 0


def _init_nvml() -> None:
    global _nvml_ok, _gpu_count
    try:
        import pynvml
        pynvml.nvmlInit()
        _gpu_count = pynvml.nvmlDeviceGetCount()
        _nvml_ok = True
    except Exception:
        _nvml_ok = False


def _collect_gpu_points() -> list[tuple[str, float]]:
    if not _nvml_ok:
        return []
    points = []
    try:
        import pynvml
        for i in range(_gpu_count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1000  # mW → W
            points += [
                (f"gpu{i}.util",      float(util.gpu)),
                (f"gpu{i}.mem_used",  float(mem.used)),
                (f"gpu{i}.mem_total", float(mem.total)),
                (f"gpu{i}.temp",      float(temp)),
                (f"gpu{i}.power_w",   float(pwr)),
            ]
    except Exception:
        pass
    return points


def _net_bytes() -> tuple[float, float]:
    c = psutil.net_io_counters()
    return float(c.bytes_sent), float(c.bytes_recv)


_prev_net: Optional[tuple[float, float]] = None
_prev_net_ts: float = 0.0


def collect() -> list[tuple[str, float]]:
    """Collect one snapshot of all system and GPU metrics."""
    global _prev_net, _prev_net_ts

    now = time.time()
    points: list[tuple[str, float]] = []

    # CPU
    per_cpu = psutil.cpu_percent(percpu=True)
    points.append(("cpu.total", float(psutil.cpu_percent())))
    for i, pct in enumerate(per_cpu):
        points.append((f"cpu.core{i}", float(pct)))

    # RAM
    vm = psutil.virtual_memory()
    points += [
        ("ram.used",      float(vm.used)),
        ("ram.total",     float(vm.total)),
        ("ram.available", float(vm.available)),
        ("ram.cached",    float(getattr(vm, "cached", 0))),
    ]

    # Swap
    sw = psutil.swap_memory()
    points += [
        ("swap.used",  float(sw.used)),
        ("swap.total", float(sw.total)),
    ]

    # Disk (root)
    try:
        du = psutil.disk_usage("/")
        points += [
            ("disk.root.used",  float(du.used)),
            ("disk.root.total", float(du.total)),
        ]
    except Exception:
        pass

    # Disk I/O
    try:
        dio = psutil.disk_io_counters()
        if dio:
            points += [
                ("disk.read_bytes",  float(dio.read_bytes)),
                ("disk.write_bytes", float(dio.write_bytes)),
            ]
    except Exception:
        pass

    # Network throughput
    sent, recv = _net_bytes()
    if _prev_net is not None and now > _prev_net_ts:
        dt = now - _prev_net_ts
        points += [
            ("net.tx_bps", (sent - _prev_net[0]) / dt),
            ("net.rx_bps", (recv - _prev_net[1]) / dt),
        ]
    _prev_net = (sent, recv)
    _prev_net_ts = now

    # GPU
    points += _collect_gpu_points()

    return points


def gpu_snapshot() -> list[dict]:
    """Return a current GPU status list for the API."""
    if not _nvml_ok:
        return []
    result = []
    try:
        import pynvml
        for i in range(_gpu_count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
            result.append({
                "index":     i,
                "name":      name,
                "gpu_util":  util.gpu,
                "mem_used":  mem.used,
                "mem_total": mem.total,
                "temp":      temp,
                "power_w":   pwr,
            })
    except Exception:
        pass
    return result


async def collection_loop() -> None:
    """Background task: collect metrics every METRICS_INTERVAL_SEC seconds."""
    _init_nvml()
    while True:
        try:
            points = collect()
            db.insert_metrics(points)
            db.prune_old_metrics(config.METRICS_RETAIN_DAYS)
        except Exception as exc:
            print(f"[metrics] collection error: {exc}")
        await asyncio.sleep(config.METRICS_INTERVAL_SEC)
