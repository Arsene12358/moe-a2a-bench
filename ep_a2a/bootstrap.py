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
from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs

from ep_a2a.config import BenchConfig


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int


def init_dist_env() -> DistEnv:
    """Read torchrun/srun env, init NCCL + SGLang model parallel (TP==EP)."""
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
    server_args = ServerArgs(
        model_path="dummy",  # never loaded; ServerArgs requires the field
        moe_a2a_backend=cfg.backend,
        deepep_mode=_deepep_mode_for(cfg.regime),
        moe_runner_backend="auto",
    )
    initialize_moe_config(server_args)

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
