from __future__ import annotations

import os
from typing import Any

import torch


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return distributed_world_size() > 1


def ensure_distributed_initialized() -> None:
    if not is_distributed():
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(distributed_local_rank())
        backend = "nccl"
    else:
        backend = "gloo"
    torch.distributed.init_process_group(backend=backend)


def distributed_barrier() -> None:
    if is_distributed():
        ensure_distributed_initialized()
        torch.distributed.barrier()


def broadcast_from_rank_zero(value: Any) -> Any:
    if not is_distributed():
        return value
    ensure_distributed_initialized()
    payload = [value]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def resolve_local_device_map(configured_device_map: str | dict[str, Any] | None):
    if not torch.cuda.is_available():
        return configured_device_map
    if is_distributed():
        return {"": distributed_local_rank()}
    if configured_device_map in (None, "auto"):
        return {"": 0}
    return configured_device_map
