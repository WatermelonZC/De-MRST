"""Strict, metadata-complete checkpoints for the formal MEDP policy."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .decentralized_am import DecentralizedAMConfig
from .formal_medp import FormalMEDPConfig, FormalMEDPPolicy
from .repro import environment_metadata, utc_now


FORMAL_MEDP_CHECKPOINT_SCHEMA_VERSION = 13
_CONFIG_FIELDS = tuple(
    field for field in FormalMEDPConfig.__dataclass_fields__ if field != "backbone"
)
_REQUIRED_PAYLOAD_FIELDS = {
    "schema_version",
    "method",
    "policy_config",
    "training_config",
    "model_state_dict",
    "optimizer_state_dict",
    "data_seed",
    "model_seed",
    "step",
    "validation_objective",
    "validation_success_rate",
    "started_at",
    "finished_at",
    "environment",
}


def save_formal_medp_checkpoint(
    path: Path,
    model: FormalMEDPPolicy,
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
    if not isinstance(model, FormalMEDPPolicy):
        raise TypeError("model must be FormalMEDPPolicy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FORMAL_MEDP_CHECKPOINT_SCHEMA_VERSION,
        "method": model.method_name,
        "policy_config": model.medp_config.to_dict(),
        "training_config": dict(training_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
        "step": int(step),
        "validation_objective": float(validation_objective),
        "validation_success_rate": float(validation_success_rate),
        "started_at": str(started_at),
        "finished_at": utc_now(),
        "environment": environment_metadata(project_dir),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_formal_medp_checkpoint(
    path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[FormalMEDPPolicy, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("MEDP checkpoint payload must be a dictionary")
    missing = _REQUIRED_PAYLOAD_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"MEDP checkpoint is missing fields: {sorted(missing)}")
    extra = set(payload).difference(_REQUIRED_PAYLOAD_FIELDS)
    if extra:
        raise ValueError(f"MEDP checkpoint has unexpected fields: {sorted(extra)}")
    if payload["schema_version"] != FORMAL_MEDP_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported De-MRST checkpoint schema; compact edge features and plain ReLU fusion require a matching checkpoint or the original checkpoint code")
    if payload["method"] != FormalMEDPPolicy.method_name:
        raise ValueError("checkpoint method is not De-MRST")
    raw_config = payload["policy_config"]
    if not isinstance(raw_config, dict) or not isinstance(
        raw_config.get("backbone"), dict
    ):
        raise TypeError("MEDP policy_config and backbone must be dictionaries")
    raw_config = dict(raw_config)
    stored_config = dict(raw_config)
    # Checkpoints produced before the ablation switch default to the full model.
    raw_config.setdefault("ablation", "full")
    raw_config.setdefault("query_fusion_contexts", 4)
    missing_config = {"backbone", *_CONFIG_FIELDS}.difference(raw_config)
    if missing_config:
        raise ValueError(f"MEDP config is missing fields: {sorted(missing_config)}")
    config = FormalMEDPConfig(
        backbone=DecentralizedAMConfig(**raw_config["backbone"]),
        **{field: raw_config[field] for field in _CONFIG_FIELDS},
    )
    config.validate()
    # New code must continue to accept exact metadata from the completed
    # four-context checkpoints, which predate query_fusion_contexts.
    stored_config.setdefault("ablation", "full")
    if stored_config != config.to_dict():
        raise ValueError("MEDP architecture metadata does not match the config")
    model = FormalMEDPPolicy(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
