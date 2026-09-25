"""AM encoder adapted to the shared decentralized task-selection interface."""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from nets.graph_encoder import GraphAttentionEncoder, MultiHeadAttention as AMMultiHeadAttention

from .decentralized_policy import SingleHeadPointer
from .observations import (
    AGENT_FEATURE_DIM,
    owner_one_hot_block, TASK_OBSERVATION_SCHEMA, task_feature_dim,
)


@dataclass(frozen=True)
class DecentralizedAMConfig:
    """AM backbone with the common current task and robot observations."""

    task_input_dim: Optional[int] = None
    agent_input_dim: int = AGENT_FEATURE_DIM
    embedding_dim: int = 128
    n_heads: int = 8
    encoder_layers: int = 1
    feed_forward_hidden: int = 512
    tanh_clipping: float = 10.0
    task_observation_schema: str = TASK_OBSERVATION_SCHEMA
    n_mbr: int = 4
    n_dor: int = 8

    def __post_init__(self):
        if self.task_input_dim is None:
            object.__setattr__(self, "task_input_dim", task_feature_dim(self.n_mbr, self.n_dor))

    def validate(self) -> None:
        self._validate_task_dim()
        if self.agent_input_dim != AGENT_FEATURE_DIM:
            raise ValueError(f"agent_input_dim must be {AGENT_FEATURE_DIM}")
        if self.embedding_dim <= 0 or self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be positive and divisible by n_heads")
        if self.encoder_layers <= 0 or self.feed_forward_hidden <= 0:
            raise ValueError("encoder layers and feed-forward size must be positive")
        if self.tanh_clipping <= 0:
            raise ValueError("tanh clipping must be positive")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def _validate_task_dim(self) -> None:
        if (
            self.task_input_dim != task_feature_dim(self.n_mbr, self.n_dor)
            or self.task_observation_schema != TASK_OBSERVATION_SCHEMA
        ):
            raise ValueError(
                "Task feature width or schema differs from the current AM/MEDP contract. "
                "Retrain or use the checkpoint's original source snapshot."
            )


class DecentralizedAMPolicy(nn.Module):
    """Standard AM graph encoder with agent-centric, legal task decoding.

    The centralized C-AM decoder selects a whole task/DOR/MBR plan.  That
    decoder is not valid under the decentralized information boundary.  This
    policy instead encodes the current task and robot observations with the
    same standard AM ``GraphAttentionEncoder``, then lets the currently
    scheduled robot select one legal task through an AM-style glimpse and
    pointer decoder.
    """

    method_name = "D-AM"
    requires_lookahead = False
    requires_owner_one_hot = True
    task_observation_schema = TASK_OBSERVATION_SCHEMA

    def __init__(self, config: DecentralizedAMConfig = None):
        if config is not None and not isinstance(config, DecentralizedAMConfig):
            raise TypeError("current AM requires DecentralizedAMConfig")
        super().__init__()
        self.config = config or DecentralizedAMConfig()
        self.config.validate()
        cfg = self.config
        self.task_embedding = nn.Linear(cfg.task_input_dim, cfg.embedding_dim)
        self.agent_embedding = nn.Linear(cfg.agent_input_dim, cfg.embedding_dim)
        self.global_token = nn.Parameter(torch.empty(1, 1, cfg.embedding_dim))
        bound = 1.0 / math.sqrt(cfg.embedding_dim)
        nn.init.uniform_(self.global_token, -bound, bound)
        self.encoder = GraphAttentionEncoder(
            n_heads=cfg.n_heads,
            embed_dim=cfg.embedding_dim,
            n_layers=cfg.encoder_layers,
            node_dim=None,
            normalization="batch",
            feed_forward_hidden=cfg.feed_forward_hidden,
        )
        self.context_projection = nn.Linear(3 * cfg.embedding_dim, cfg.embedding_dim)
        self.glimpse = AMMultiHeadAttention(
            cfg.n_heads,
            input_dim=cfg.embedding_dim,
            embed_dim=cfg.embedding_dim,
        )
        self.pointer = SingleHeadPointer(cfg.embedding_dim, cfg.tanh_clipping)

    def _validate_inputs(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> None:
        if task_inputs.ndim != 3 or task_inputs.size(-1) != self.config.task_input_dim:
            raise ValueError("invalid task input shape")
        if agent_inputs.ndim != 3 or agent_inputs.size(-1) != self.config.agent_input_dim:
            raise ValueError("invalid agent input shape")
        batch, task_count, _ = task_inputs.shape
        if agent_inputs.size(0) != batch or tuple(action_mask.shape) != (batch, task_count):
            raise ValueError("batch and mask shapes are inconsistent")
        if tuple(agent_index.shape) != (batch,):
            raise ValueError("agent_index must have shape [batch]")
        if not torch.is_floating_point(task_inputs) or not torch.is_floating_point(agent_inputs):
            raise TypeError("policy inputs must be floating-point tensors")
        if not 0 <= int(agent_index.min()) or int(agent_index.max()) >= agent_inputs.size(1):
            raise IndexError("agent index is out of range")
        if action_mask.all(dim=-1).any():
            raise ValueError("each decision needs at least one legal task")
        self._validate_owner_features(task_inputs, agent_inputs)

    def _embed_inputs(self, task_inputs, agent_inputs):
        return self.task_embedding(task_inputs), self.agent_embedding(agent_inputs)

    def forward(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(task_inputs, agent_inputs, action_mask, agent_index)
        batch_size, task_count, _ = task_inputs.shape
        task_embedding, agent_embedding = self._embed_inputs(task_inputs, agent_inputs)
        global_token = self.global_token.expand(batch_size, -1, -1)
        encoded_nodes, graph_context, encoded_global = self.encoder(
            torch.cat((task_embedding, agent_embedding, global_token), dim=1)
        )
        task_context = encoded_nodes[:, :task_count]
        agent_context = encoded_nodes[:, task_count:]
        batch = torch.arange(batch_size, device=task_inputs.device)
        current_agent = agent_context[batch, agent_index]
        decoder_query = self.context_projection(
            torch.cat((current_agent, graph_context, encoded_global.squeeze(1)), dim=-1)
        ).unsqueeze(1)
        glimpse = self.glimpse(
            decoder_query,
            task_context,
            mask=action_mask.unsqueeze(1),
        )
        return self.pointer(decoder_query + glimpse, task_context, action_mask)

    def _owner_vectors(self, task_inputs):
        owners = owner_one_hot_block(task_inputs)
        return owners[..., :self.config.n_mbr], owners[..., self.config.n_mbr:]

    def _validate_owner_features(self, task_inputs, agent_inputs):
        if agent_inputs.size(1) != self.config.n_mbr + self.config.n_dor:
            raise ValueError("robot count differs from the checkpoint's one-hot fleet")
        role_expected = torch.arange(agent_inputs.size(1), device=agent_inputs.device) < self.config.n_mbr
        self._runtime_assert(
            (agent_inputs[..., 0] == role_expected) & (agent_inputs[..., 1] == ~role_expected),
            ValueError, "robot rows must be CRs followed by WRs in role-local ID order",
        )
        for vector in self._owner_vectors(task_inputs):
            self._runtime_assert(
                ((vector == 0) | (vector == 1)).all(-1) & (vector.sum(-1) <= 1),
                ValueError, "owner vectors must be one-hot or all zero",
            )

    def _assigned_owner_indices(self, task_inputs):
        """Derive exact current assignment rows for MEDP masks, not new inputs."""
        cr, wr = self._owner_vectors(task_inputs)
        return torch.stack((
            torch.where(cr.sum(-1) > 0, cr.argmax(-1), -1),
            torch.where(wr.sum(-1) > 0, wr.argmax(-1) + self.config.n_mbr, -1),
        ), dim=-1)

    @staticmethod
    def _runtime_assert(condition, error_type, message):
        valid = condition.all()
        if valid.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise error_type(message)
