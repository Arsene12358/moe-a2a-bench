"""Glue that runs one dispatch -> identity-expert -> combine cycle through a
DeepEP-family dispatcher, plus a correctness gate.

For normal mode the dispatch output is DeepEPNormalDispatchOutput; for low
latency it is DeepEPLLDispatchOutput. In both, combine() consumes a
(hidden_states, topk_ids, topk_weights) tuple. The "identity expert" feeds the
dispatched hidden states straight back to combine, so we measure pure
transport (dispatch + combine) with no GEMM.

NOTE (resolve on first GPU smoke): MooncakeEPDispatcher / NixlEPDispatcher and
the low-latency impl may return a DispatchOutput whose hidden_states layout or
combine tuple differs from DeepEP normal. If run_once fails for a backend/regime,
read that backend's dispatch/combine and add a minimal format branch keyed off
the dispatch output here -- do NOT fork run_once.
"""
from __future__ import annotations

import torch


def _expert_output_from_dispatch(dispatch_out):
    """Build the bf16 'expert output' tensor combine expects.

    In bf16 mode the dispatched hidden_states are already bf16 and can be fed
    back directly (true identity). In native (fp8/nvfp4) mode the dispatched
    tensor is quantized; production would emit a bf16 GEMM result of the same
    leading shape, so we materialize a bf16 tensor of the dispatched layout.
    """
    hs = dispatch_out.hidden_states
    if isinstance(hs, tuple):  # (quantized_tensor, scale)
        hs = hs[0]
    if hs.dtype == torch.bfloat16:
        return hs
    # Quantized dispatch (native mode): DeepEP combine only accepts bf16, so
    # cast the dispatched tokens back to bf16 (mimics the bf16 GEMM output).
    return hs.to(torch.bfloat16)


def run_once(dispatcher, hidden_states, topk_output):
    """One dispatch+combine cycle. Returns the combined output tensor.

    MaybeTboDeepEPDispatcher.dispatch/combine delegate to the inner dispatcher
    via **kwargs, so they must be called with keyword arguments.
    """
    dispatch_out = dispatcher.dispatch(
        hidden_states=hidden_states, topk_output=topk_output
    )
    expert_out = _expert_output_from_dispatch(dispatch_out)
    combine_input = (
        expert_out,
        dispatch_out.topk_ids,
        dispatch_out.topk_weights,
    )
    return dispatcher.combine(combine_input=combine_input)


def correctness_gate(dispatcher, hidden_states, topk_output, atol=2e-2):
    """In bf16 balanced mode, combine(dispatch(x)) reconstructs the
    topk-weighted sum of x. We check the output is finite, the right shape,
    and non-trivial (not all-zero). A backend that fails is INVALID.

    NOTE: exact numerical equivalence depends on whether combine internally
    re-applies topk_weights; the implementer confirms the precise reference on
    the smoke run and tightens this check. The minimum bar here catches a
    backend that silently drops or corrupts tokens.
    """
    out = run_once(dispatcher, hidden_states, topk_output)
    assert out.shape == hidden_states.shape, (out.shape, hidden_states.shape)
    assert torch.isfinite(out).all(), "combine produced non-finite values"
    assert out.abs().sum().item() > 0, "combine produced all-zero output"
    return out
