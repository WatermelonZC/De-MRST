"""Standard-attention centralized model with the existing hierarchical decoder."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from nets.attention_model import AttentionModel
from nets.graph_encoder import GraphAttentionEncoder
from problems.centralized_mrs.problem import CentralizedMRSProblem
from .repro import environment_metadata, utc_now


@dataclass(frozen=True)
class CAmConfig:
    n_mbr: int
    n_dor: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    n_encode_layers: int = 1
    n_heads: int = 8
    normalization: str = 'batch'
    tanh_clipping: float = 10.0
    designated_driver: bool = True


def build_c_am(config: CAmConfig) -> AttentionModel:
    model = AttentionModel(
        config.embedding_dim,
        config.hidden_dim,
        CentralizedMRSProblem,
        n_encode_layers=config.n_encode_layers,
        n_heads=config.n_heads,
        normalization=config.normalization,
        tanh_clipping=config.tanh_clipping,
        mom_vehicle_size=config.n_mbr,
        sub_vehicle_size=config.n_dor,
        designated_driver=config.designated_driver,
    )
    if not isinstance(model.embedder, GraphAttentionEncoder):
        raise TypeError('C-AM did not instantiate GraphAttentionEncoder')
    return model


def save_c_am_checkpoint(
    path: Path,
    model: AttentionModel,
    config: CAmConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    validation_objective: Optional[float] = None,
    validation_success_rate: Optional[float] = None,
    training_config: Optional[Dict[str, Any]] = None,
    data_seed: Optional[int] = None,
    model_seed: Optional[int] = None,
    step: Optional[int] = None,
    started_at: Optional[str] = None,
    project_dir: Optional[Path] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        'schema_version': 2,
        'method': 'C-AM',
        'config': asdict(config),
        'model': model.state_dict(),
        'validation_objective': validation_objective,
        'validation_success_rate': validation_success_rate,
        'training_config': training_config,
        'data_seed': data_seed,
        'model_seed': model_seed,
        'step': step,
        'started_at': started_at,
        'finished_at': utc_now(),
        'environment': environment_metadata(project_dir) if project_dir else None,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_c_am_checkpoint(
    path: Path,
    device: torch.device,
    fleet_override: Optional[Tuple[int, int]] = None,
) -> Tuple[AttentionModel, CAmConfig, Dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get('method') != 'C-AM':
        raise ValueError('checkpoint is not a C-AM checkpoint')
    raw_config = dict(payload['config'])
    if raw_config.pop('cat_layer', False):
        raise ValueError('CAT checkpoints are not part of the C-AM baseline')
    config = CAmConfig(**raw_config)
    if fleet_override is not None:
        config = CAmConfig(
            n_mbr=int(fleet_override[0]),
            n_dor=int(fleet_override[1]),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_encode_layers=config.n_encode_layers,
            n_heads=config.n_heads,
            normalization=config.normalization,
            tanh_clipping=config.tanh_clipping,
            designated_driver=config.designated_driver,
        )
    model = build_c_am(config).to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return model, config, payload
