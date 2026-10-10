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


def summarize(times):
    """Median and interquartile range (plus mean) of per-query times in seconds."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return {"n": 0}
    q25, median, q75 = np.percentile(times, [25, 50, 75])
    return {
        "n": int(times.size),
        "median_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "iqr_s": float(q75 - q25),
        "mean_s": float(times.mean()),
        "total_s": float(times.sum()),
    }


def device_label(device):
    return 'cuda' if str(device).startswith('cuda') else 'cpu'
