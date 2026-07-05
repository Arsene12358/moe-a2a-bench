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


def _capture_graph(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_fn_cuda_graph(fn: Callable[[], None], warmups: int, iters: int) -> List[float]:
    """Like time_fn, but captures `fn` into a CUDA graph and times replays.

    Removes host-side cost (python, launches, wrapper logic) from the timed
    region: what remains is the device path — the kernels, including their
    fused receive-waits, which are genuine transport. Requires a
    capture-safe fn (no host synchronization inside): DeepEP low-latency
    (decode regime) qualifies — production serves it under CUDA graphs —
    while normal mode (prefill) does not.
    """
    for _ in range(warmups):
        fn()
    g = _capture_graph(fn)
    for _ in range(3):  # graph warmup
        g.replay()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]


def time_phases_cuda_graph(
    phases: List[tuple], warmups: int, iters: int
) -> dict:
    """Graph-mode variant of time_phases: one graph per named phase, timed
    per replay. Capture runs the phases' host code once (populating any
    shared state) without executing kernels; replays reuse the stable DeepEP
    buffers, so cross-phase tensor handles stay valid."""
    for _ in range(warmups):
        for _, fn in phases:
            fn()
    torch.cuda.synchronize()

    graphs = [(name, _capture_graph(fn)) for name, fn in phases if name]
    for _ in range(3):
        for _, g in graphs:
            g.replay()
    torch.cuda.synchronize()

    ev = {
        n: (
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
        )
        for n, _ in graphs
    }
    for i in range(iters):
        for name, g in graphs:
            ev[name][0][i].record()
            g.replay()
            ev[name][1][i].record()
    torch.cuda.synchronize()
    return {
        n: [s.elapsed_time(e) / 1e3 for s, e in zip(ev[n][0], ev[n][1])]
        for n, _ in ev.items()
    }


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
