"""CUDA-event timing for the dispatch/combine cycle."""
from __future__ import annotations

from typing import Callable, List

import torch


def time_fn(fn: Callable[[], None], warmups: int, iters: int) -> List[float]:
    """Time `fn` with CUDA events. Returns per-iter seconds (length == iters)."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]
