"""HetMRTA-style shared decentralized task-selection policy.

The architecture is adapted from marmotlab/HeteroMRTA commit
db51e29535e34bcaa8cb75f70be3fd28b0027988 (Apache-2.0). Problem-specific
inputs and transitions are implemented locally for the Marsupial system.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .observations import (
    AGENT_FEATURE_DIM,
    owner_one_hot_block, TASK_OBSERVATION_SCHEMA, task_feature_dim,
)


HETEROMRTA_UPSTREAM_COMMIT = "db51e29535e34bcaa8cb75f70be3fd28b0027988"


@dataclass(frozen=True)
class HetMRTAPolicyConfig:
    task_input_dim: Optional[int] = None
    agent_input_dim: int = AGENT_FEATURE_DIM
    embedding_dim: int = 128
    n_heads: int = 8
    task_encoder_layers: int = 1
    agent_encoder_layers: int = 1
    cross_decoder_layers: int = 2
    global_decoder_layers: int = 2
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
        if min(
            self.task_encoder_layers,
            self.agent_encoder_layers,
            self.cross_decoder_layers,
            self.global_decoder_layers,
        ) <= 0:
            raise ValueError("all encoder and decoder layer counts must be positive")
        if self.feed_forward_hidden <= 0 or self.tanh_clipping <= 0:
            raise ValueError("feed-forward size and tanh clipping must be positive")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def _validate_task_dim(self):
        if (self.task_input_dim != task_feature_dim(self.n_mbr, self.n_dor)
                or self.task_observation_schema != TASK_OBSERVATION_SCHEMA):
            raise ValueError(
                "Task feature width or schema differs from the current Hete contract. "
                "Retrain or use the checkpoint's original source snapshot."
            )


def _uniform_parameter_initialization(module: nn.Module) -> None:
    for parameter in module.parameters(recurse=False):
        if parameter.ndim == 0:
            continue
        bound = 1.0 / math.sqrt(parameter.size(-1))
        nn.init.uniform_(parameter, -bound, bound)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.w_query = nn.Parameter(torch.empty(n_heads, embedding_dim, self.head_dim))
        self.w_key = nn.Parameter(torch.empty(n_heads, embedding_dim, self.head_dim))
        self.w_value = nn.Parameter(torch.empty(n_heads, embedding_dim, self.head_dim))
        self.w_out = nn.Parameter(torch.empty(n_heads, self.head_dim, embedding_dim))
        _uniform_parameter_initialization(self)

    def forward(
        self,
        query: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = query if memory is None else memory
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("attention inputs must have shape [batch, sequence, embedding]")
        if query.size(0) != memory.size(0) or query.size(-1) != self.embedding_dim:
            raise ValueError("incompatible query and memory shapes")
        if memory.size(-1) != self.embedding_dim:
            raise ValueError("incompatible memory embedding dimension")

        q = torch.einsum("bqd,hdk->hbqk", query, self.w_query)
        k = torch.einsum("btd,hdk->hbtk", memory, self.w_key)
        v = torch.einsum("btd,hdv->hbtv", memory, self.w_value)
        compatibility = self.norm_factor * torch.einsum("hbqk,hbtk->hbqt", q, k)

        expanded_mask = None
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=query.device)
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).expand(-1, query.size(1), -1)
            expected = (query.size(0), query.size(1), memory.size(1))
            if tuple(mask.shape) != expected:
                raise ValueError(f"attention mask must have shape {expected}")
            if mask.all(dim=-1).any():
                raise ValueError("attention cannot mask every memory item for a query")
            expanded_mask = mask.unsqueeze(0)
            compatibility = compatibility.masked_fill(expanded_mask, -torch.inf)

        attention = torch.softmax(compatibility, dim=-1)
        if expanded_mask is not None:
            attention = attention.masked_fill(expanded_mask, 0.0)
        heads = torch.einsum("hbqt,hbtv->hbqv", attention, v)
        return torch.einsum("bqhv,hvd->bqd", heads.permute(1, 2, 0, 3), self.w_out)


class GatedFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, embedding_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(torch.sigmoid(self.gate(inputs)) * self.value(inputs))


class GatedFeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.normalization = nn.LayerNorm(embedding_dim)
        self.feed_forward = GatedFeedForward(embedding_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.normalization(inputs))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.normalization = nn.LayerNorm(embedding_dim)
        self.feed_forward = GatedFeedForwardLayer(embedding_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = inputs + self.attention(self.normalization(inputs), mask=mask)
        return hidden + self.feed_forward(hidden)


class CrossDecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        # Retained for state-dict compatibility with the released architecture;
        # its forward pass, like the upstream implementation, uses cross-attention only.
        self.unused_self_attention = MultiHeadAttention(embedding_dim, n_heads)
        self.cross_attention = MultiHeadAttention(embedding_dim, n_heads)
        self.query_normalization = nn.LayerNorm(embedding_dim)
        self.memory_normalization = nn.LayerNorm(embedding_dim)
        self.feed_forward = GatedFeedForwardLayer(embedding_dim, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = query + self.cross_attention(
            self.query_normalization(query),
            self.memory_normalization(memory),
            mask,
        )
        return hidden + self.feed_forward(hidden)


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)
        )

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs, mask)
        return inputs


class CrossDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossDecoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, memory, mask)
        return query


class SingleHeadPointer(nn.Module):
    def __init__(self, embedding_dim: int, tanh_clipping: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping
        self.norm_factor = 1.0 / math.sqrt(embedding_dim)
        self.w_query = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.w_key = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        _uniform_parameter_initialization(self)

    def logits(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.ndim != 2 or tuple(mask.shape) != (query.size(0), memory.size(1)):
            raise ValueError("pointer mask must have shape [batch, tasks]")
        if mask.all(dim=-1).any():
            raise ValueError("pointer requires at least one legal task per batch row")
        q = torch.matmul(query, self.w_query)
        k = torch.matmul(memory, self.w_key)
        logits = self.norm_factor * torch.matmul(q, k.transpose(1, 2))
        logits = self.tanh_clipping * torch.tanh(logits).squeeze(1)
        return logits.masked_fill(mask, -torch.inf)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(query, memory, mask)
        return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)


class HetMRTAPolicy(nn.Module):
    """One parameter-shared policy for all MBR and DOR decision makers."""

    method_name = "HetMRTA-RL-MRS"
    requires_owner_one_hot = True
    task_observation_schema = TASK_OBSERVATION_SCHEMA

    def __init__(self, config: HetMRTAPolicyConfig = None):
        if config is not None and not isinstance(config, HetMRTAPolicyConfig):
            raise TypeError("current Hete requires HetMRTAPolicyConfig")
        super().__init__()
        self.config = config or HetMRTAPolicyConfig()
        self.config.validate()
        cfg = self.config
        self.task_embedding = nn.Linear(cfg.task_input_dim, cfg.embedding_dim)
        self.agent_embedding = nn.Linear(cfg.agent_input_dim, cfg.embedding_dim)
        self.task_encoder = Encoder(
            cfg.embedding_dim,
            cfg.n_heads,
            cfg.feed_forward_hidden,
            cfg.task_encoder_layers,
        )
        self.agent_encoder = Encoder(
            cfg.embedding_dim,
            cfg.n_heads,
            cfg.feed_forward_hidden,
            cfg.agent_encoder_layers,
        )
        self.task_to_agent_decoder = CrossDecoder(
            cfg.embedding_dim,
            cfg.n_heads,
            cfg.feed_forward_hidden,
            cfg.cross_decoder_layers,
        )
        self.agent_to_task_decoder = CrossDecoder(
            cfg.embedding_dim,
            cfg.n_heads,
            cfg.feed_forward_hidden,
            cfg.cross_decoder_layers,
        )
        self.fusion = nn.Linear(cfg.embedding_dim * 3, cfg.embedding_dim)
        self.global_decoder = CrossDecoder(
            cfg.embedding_dim,
            cfg.n_heads,
            cfg.feed_forward_hidden,
            cfg.global_decoder_layers,
        )
        self.pointer = SingleHeadPointer(cfg.embedding_dim, cfg.tanh_clipping)

    def _encode_backbone_components(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(task_inputs, agent_inputs, action_mask, agent_index)
        task_embedding = self.task_embedding(task_inputs)
        agent_embedding = self.agent_embedding(agent_inputs)
        task_encoding = self.task_encoder(task_embedding)
        agent_encoding = self.agent_encoder(agent_embedding)
        task_agent_features = self.task_to_agent_decoder(task_encoding, agent_encoding)
        agent_task_features = self.agent_to_task_decoder(agent_encoding, task_encoding)

        batch = torch.arange(task_inputs.size(0), device=task_inputs.device)
        current_agent = agent_task_features[batch, agent_index].unsqueeze(1)
        aggregated_tasks = task_embedding.mean(dim=1, keepdim=True)
        aggregated_agents = agent_embedding.mean(dim=1, keepdim=True)
        current_state = self.fusion(
            torch.cat((current_agent, aggregated_tasks, aggregated_agents), dim=-1)
        )
        return task_agent_features, current_state, agent_encoding

    def encode_backbone(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_context, current_state, _ = self._encode_backbone_components(
            task_inputs, agent_inputs, action_mask, agent_index
        )
        return task_context, current_state

    def decode_tasks(
        self,
        task_context: torch.Tensor,
        current_state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.decode_task_logits(task_context, current_state, action_mask)
        return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)

    def decode_task_logits(
        self,
        task_context: torch.Tensor,
        current_state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoded_state = self.global_decoder(current_state, task_context, action_mask)
        return self.pointer.logits(decoded_state, task_context, action_mask)

    def encode_common(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible access to the common context and decoded state."""
        task_context, current_state = self.encode_backbone(
            task_inputs, agent_inputs, action_mask, agent_index
        )
        decoded_state = self.global_decoder(current_state, task_context, action_mask)
        return task_context, decoded_state

    def forward(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_context, current_state = self.encode_backbone(
            task_inputs,
            agent_inputs,
            action_mask,
            agent_index,
        )
        return self.decode_tasks(task_context, current_state, action_mask)

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
        cfg = self.config
        if agent_inputs.size(1) != cfg.n_mbr + cfg.n_dor:
            raise ValueError("robot count differs from Hete's one-hot fleet")
        is_cr = torch.arange(agent_inputs.size(1), device=agent_inputs.device) < cfg.n_mbr
        valid = ((agent_inputs[..., 0] == is_cr) & (agent_inputs[..., 1] == ~is_cr)).all()
        bits = owner_one_hot_block(task_inputs)
        for owners in (bits[..., :cfg.n_mbr], bits[..., cfg.n_mbr:]):
            valid = valid & (((owners == 0) | (owners == 1)).all() & (owners.sum(-1) <= 1).all())
        message = "Hete requires CR/WR-ordered robot rows and zero-or-one-hot owner vectors"
        if valid.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise ValueError(message)
