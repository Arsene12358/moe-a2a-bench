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


def time_phases(
    phases: List[tuple], warmups: int, iters: int
) -> dict:
    """Time a sequence of phases per iteration with CUDA events.

    phases: list of (name, fn); phases with name=None run untimed between
    timed ones (e.g. the identity glue). Returns {name: [seconds]*iters}.
    """
    for _ in range(warmups):
        for _, fn in phases:
            fn()
    torch.cuda.synchronize()

    named = [n for n, _ in phases if n]
    ev = {
        n: (
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
        )
        for n in named
    }
    for i in range(iters):
        for name, fn in phases:
            if name:
                ev[name][0][i].record()
                fn()
                ev[name][1][i].record()
            else:
                fn()
    torch.cuda.synchronize()
    return {
        n: [s.elapsed_time(e) / 1e3 for s, e in zip(ev[n][0], ev[n][1])]
        for n in named
    }
