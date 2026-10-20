import time

import numpy as np
import torch

# Untimed queries run before timing, so lazy CUDA init / allocator warm-up is not measured
WARMUP_QUERIES = 10


class QueryTimer:
    """Wall-clock timer for one query: perf_counter, with cuda.synchronize() before start and stop."""

    def __init__(self, device):
        self.cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        self.elapsed = None

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        return False


# Control periods for the deadline-miss rate (1 kHz and 100 Hz control)
DEADLINES_S = (1e-3, 1e-2)


def summarize(times):
    """Median and interquartile range (headline), plus mean, p95, max and deadline-miss rates,
    of per-query times in seconds."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return {"n": 0}
    q25, median, q75, p95 = np.percentile(times, [25, 50, 75, 95])
    return {
        "n": int(times.size),
        "median_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "iqr_s": float(q75 - q25),
        "mean_s": float(times.mean()),
        "p95_s": float(p95),
        "max_s": float(times.max()),
        "total_s": float(times.sum()),
        "deadline_miss_pct": {f"{d * 1e3:g}ms": float((times > d).mean() * 100) for d in DEADLINES_S},
    }


def describe(timing):
    """One console line: median, mean, p95 and max per query (ms), plus the IQR."""
    if not timing.get('n'):
        return "no timed queries"
    ms = lambda key: timing[key] * 1e3
    return (f"median {ms('median_s'):.2f} ms | mean {ms('mean_s'):.2f} ms | p95 {ms('p95_s'):.2f} ms | "
            f"max {ms('max_s'):.2f} ms (IQR {ms('iqr_s'):.2f} ms, n={timing['n']})")


class cpu_threads:
    """Pins torch to `n` CPU threads inside the block (only when timing on CPU) and restores the
    previous value afterwards, so CPU timings do not depend on how many cores happen to be idle."""

    def __init__(self, n, device='cpu'):
        self.n = int(n) if n and device_label(device) == 'cpu' else None

    def __enter__(self):
        self.previous = torch.get_num_threads()
        if self.n:
            torch.set_num_threads(self.n)
        return self

    def __exit__(self, *exc):
        torch.set_num_threads(self.previous)
        return False

    @property
    def threads(self):
        """Threads used for the timing (None on GPU)."""
        return self.n or None


def timing_settings(dataset_cfg):
    """The 'timing:' block of benchmark_config.yaml (defaults: NODE single paths on CPU, 1 thread)."""
    cfg = dataset_cfg.get('timing') or {}
    return {'single_path_device': cfg.get('single_path_device', 'cpu'),
            'cpu_threads': cfg.get('cpu_threads', 1)}


def device_label(device):
    return 'cuda' if str(device).startswith('cuda') else 'cpu'
