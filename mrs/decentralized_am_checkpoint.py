"""Strict checkpoint helpers for the decentralized AM adaptation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .decentralized_am import DecentralizedAMConfig, DecentralizedAMPolicy
from .repro import environment_metadata, utc_now


CHECKPOINT_SCHEMA_VERSION = 7


def save_decentralized_am_checkpoint(
    path: Path,
    model: DecentralizedAMPolicy,
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
        "encoder": "standard GraphAttentionEncoder",
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


def load_decentralized_am_checkpoint(
    path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DecentralizedAMPolicy, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported decentralized AM checkpoint schema; current task column order requires retraining or the original checkpoint code")
    if payload.get("method") != DecentralizedAMPolicy.method_name:
        raise ValueError("checkpoint method is not D-AM")
    if payload.get("encoder") != "standard GraphAttentionEncoder":
        raise ValueError("checkpoint encoder is not the standard AM encoder")
    model = DecentralizedAMPolicy(DecentralizedAMConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
