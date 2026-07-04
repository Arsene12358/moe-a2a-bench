"""Bring up torch.distributed + SGLang parallel state + MoE config, then
build the production dispatcher for the chosen backend."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from ep_a2a.config import BenchConfig


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int


def _maybe_patch_deepep_buffer():
    """Experiment knobs for DeepEP internode bring-up on GB200.

    The stock DeepEP Buffer uses legacy CUDA IPC (cudaIpcOpenMemHandle), which
    is node-local and fails cross-node with 'invalid resource handle'. On GB200
    with the IMEX fabric up, the cross-node path needs use_fabric=True
    (CU_MEM_HANDLE_TYPE_FABRIC). sglang sets neither, so we override here:
      EP_A2A_USE_FABRIC=1     -> use_fabric=True   (fabric shareable handles)
      EP_A2A_DISABLE_MNNVL=1  -> allow_mnnvl=False (force IB-RDMA internode)
    """
    use_fabric = os.environ.get("EP_A2A_USE_FABRIC") == "1"
    disable_mnnvl = os.environ.get("EP_A2A_DISABLE_MNNVL") == "1"
    if not (use_fabric or disable_mnnvl):
        return
    try:
        import deep_ep
    except ImportError:
        return
    _orig = deep_ep.Buffer.__init__

    def _patched(self, *args, **kwargs):
        if use_fabric:
            kwargs["use_fabric"] = True
        if disable_mnnvl:
            kwargs["allow_mnnvl"] = False
        return _orig(self, *args, **kwargs)

    deep_ep.Buffer.__init__ = _patched


def init_dist_env(cfg: BenchConfig) -> DistEnv:
    """Read torchrun/srun env, init NCCL + SGLang model parallel (TP==EP).

    moe_a2a_backend is passed so init_distributed_environment creates the
    extra coordination state some backends need (e.g. NIXL's global TCPStore).
    """
    _maybe_patch_deepep_buffer()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    init_method = f"tcp://{master_addr}:{master_port}"

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=init_method,
        backend="nccl",
        moe_a2a_backend=cfg.backend,
    )
    # ep_size == tp_size == world_size for all DeepEP-family backends.
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return DistEnv(rank=rank, world_size=world_size, local_rank=local_rank)


def _deepep_mode_for(regime: str) -> str:
    # prefill -> normal (bandwidth), decode -> low_latency.
    return "normal" if regime == "prefill" else "low_latency"


def build_dispatcher(cfg: BenchConfig, env: DistEnv):
    """Set MoE globals for the chosen backend/regime and build the dispatcher
    via the production factory.

    NOTE (resolve on first GPU smoke): ServerArgs runs validation in
    __post_init__ that may require extra fields or mutate ep_size/deepep_mode.
    If construction raises, inspect python/sglang/srt/server_args.py and supply
    the minimal fields (e.g. tp_size/ep_size = world_size) to build it without
    launching a server.
    """
    # With JIT DeepGEMM the dispatcher defaults to fp8 output; force bf16 for
    # bf16 mode so dispatch emits bf16 (combine requires bf16). native -> auto.
    # Field names vary across sglang versions (e.g. v0.5.11 has no
    # deepep_dispatcher_output_dtype); construct with what this version
    # supports and report anything dropped.
    import dataclasses

    dispatch_dtype = "bf16" if cfg.dtype_mode == "bf16" else "auto"
    wanted = {
        "model_path": "dummy",  # never loaded; ServerArgs requires the field
        "moe_a2a_backend": cfg.backend,
        "deepep_mode": _deepep_mode_for(cfg.regime),
        "deepep_dispatcher_output_dtype": dispatch_dtype,
        "moe_runner_backend": "auto",
    }
    supported = {f.name for f in dataclasses.fields(ServerArgs)}
    dropped = sorted(k for k in wanted if k not in supported)
    if dropped and env.rank == 0:
        print(f"[bootstrap] ServerArgs fields unsupported here, dropped: {dropped}")
    server_args = ServerArgs(**{k: v for k, v in wanted.items() if k in supported})
    # The DeepEP-family dispatchers read the process-global ServerArgs (e.g.
    # get_deepep_output_dtype -> get_global_server_args), so it must be set in
    # addition to the MoE config globals.
    set_global_server_args_for_scheduler(server_args)
    initialize_moe_config(server_args)
    # DeepEP _get_impl() reads this global during dispatch; normally set per
    # forward batch. prefill -> extend (True), decode -> not extend (False).
    set_is_extend_in_batch(cfg.regime == "prefill")

    assert cfg.num_experts % env.world_size == 0, (
        f"num_experts {cfg.num_experts} not divisible by world_size {env.world_size}"
    )
    num_local_experts = cfg.num_experts // env.world_size
    runner_config = MoeRunnerConfig(
        num_experts=cfg.num_experts,
        num_local_experts=num_local_experts,
        hidden_size=cfg.hidden,
        top_k=cfg.topk,
        params_dtype=torch.bfloat16,
    )
    dispatcher = create_moe_dispatcher(runner_config)
    return dispatcher, num_local_experts
