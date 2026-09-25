"""Strict, metadata-complete checkpoints for decentralized policies."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .decentralized_policy import (
    HETEROMRTA_UPSTREAM_COMMIT,
    HetMRTAPolicy,
    HetMRTAPolicyConfig,
)
from .repro import environment_metadata, utc_now


CHECKPOINT_SCHEMA_VERSION = 6


def save_hetmrta_checkpoint(
    path: Path,
    model: HetMRTAPolicy,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    training_config: Dict[str, Any],
    data_seed: int,
    model_seed: int,
    step: int,
    validation_objective: float,
    validation_success_rate: float,
    started_at: str,
    project_dir: Optional[Path] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": model.method_name,
        "upstream": {
            "repository": "https://github.com/marmotlab/HeteroMRTA",
            "commit": HETEROMRTA_UPSTREAM_COMMIT,
            "license": "Apache-2.0",
        },
        "policy_config": model.config.to_dict(),
        "training_config": training_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
        "step": int(step),
        "validation_objective": float(validation_objective),
        "validation_success_rate": float(validation_success_rate),
        "started_at": started_at,
        "finished_at": utc_now(),
        "environment": environment_metadata(project_dir),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_hetmrta_checkpoint(
    path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[HetMRTAPolicy, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Hete checkpoint schema; current task column order requires retraining or the original checkpoint code")
    if payload.get("method") != HetMRTAPolicy.method_name:
        raise ValueError("checkpoint method is not HetMRTA-RL-MRS")
    upstream = payload.get("upstream", {})
    if upstream.get("commit") != HETEROMRTA_UPSTREAM_COMMIT:
        raise ValueError("checkpoint upstream version does not match the frozen baseline")
    config = HetMRTAPolicyConfig(**payload["policy_config"])
    model = HetMRTAPolicy(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
