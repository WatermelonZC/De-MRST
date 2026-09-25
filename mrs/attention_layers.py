"""Shared attention primitives used by the current De-MRST network."""

import torch
from torch import nn
from torch.nn import functional as F

from .observations import AGENT_FEATURES

AGENT_RELATIVE_X_FEATURE_INDEX = AGENT_FEATURES.index("relative_x")
AGENT_RELATIVE_Y_FEATURE_INDEX = AGENT_FEATURES.index("relative_y")


def _masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Masked softmax whose result is exactly zero for an all-masked row."""
    mask = torch.broadcast_to(mask.to(device=scores.device, dtype=torch.bool), scores.shape)
    all_masked = mask.all(dim=dim, keepdim=True)
    masked_scores = scores.masked_fill(mask, -torch.inf)
    safe_scores = torch.where(all_masked, torch.zeros_like(masked_scores), masked_scores)
    probabilities = torch.softmax(safe_scores, dim=dim).masked_fill(mask, 0.0)
    return torch.where(all_masked, torch.zeros_like(probabilities), probabilities)


class RMSNorm(nn.Module):
    """RMSNorm implemented without depending on ``torch.nn.RMSNorm``."""

    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source_dtype = inputs.dtype
        normalized = inputs.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=source_dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(inputs)) * self.value(inputs))


class LayerScale(nn.Module):
    def __init__(self, dimension: int, initial_value: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dimension,), float(initial_value)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale


class EdgeTransition(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        edge_dim: int,
        hidden_dim: int,
        layer_scale_init: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.task_projection = nn.Linear(embedding_dim, edge_dim, bias=False)
        self.agent_projection = nn.Linear(embedding_dim, edge_dim, bias=False)
        self.normalization = RMSNorm(edge_dim, rms_norm_eps)
        self.feed_forward = SwiGLU(edge_dim, hidden_dim, edge_dim)
        self.layer_scale = LayerScale(edge_dim, layer_scale_init)

    def forward(
        self,
        edge_state: torch.Tensor,
        task_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> torch.Tensor:
        transition_input = (
            edge_state
            + self.task_projection(task_state).unsqueeze(2)
            + self.agent_projection(agent_state).unsqueeze(1)
        )
        return edge_state + self.layer_scale(
            self.feed_forward(self.normalization(transition_input))
        )
