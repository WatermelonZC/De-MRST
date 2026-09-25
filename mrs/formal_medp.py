"""De-MRST policy with pair reasoning and an attention decoder."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .decentralized_am import DecentralizedAMConfig, DecentralizedAMPolicy
from .method_names import DE_MRST_NAME
from .attention_layers import (
    AGENT_RELATIVE_X_FEATURE_INDEX,
    AGENT_RELATIVE_Y_FEATURE_INDEX,
    EdgeTransition,
    RMSNorm,
    _masked_softmax,
)
from .observations import (
    TASK_BASE_FEATURES, TASK_BASE_FEATURE_DIM, task_allocation_masks,
)

# Geometry is in the fixed-width suffix after the fleet-dependent owner block.
TASK_SOURCE_X_FEATURE_INDEX = TASK_BASE_FEATURES.index("source_dx") - TASK_BASE_FEATURE_DIM
TASK_SOURCE_Y_FEATURE_INDEX = TASK_BASE_FEATURES.index("source_dy") - TASK_BASE_FEATURE_DIM
TASK_DESTINATION_X_FEATURE_INDEX = TASK_BASE_FEATURES.index("destination_dx") - TASK_BASE_FEATURE_DIM
TASK_DESTINATION_Y_FEATURE_INDEX = TASK_BASE_FEATURES.index("destination_dy") - TASK_BASE_FEATURE_DIM


MEDP_ARCHITECTURE = "dam_trunk_partner_pooling_relu_fusion_v11"
MEDP_PAPER_ARCHITECTURE = "dam_trunk_partner_pooling_paper_query_v12"
MEDP_EDGE_FEATURE_SCHEMA = "marsupial_edge_offsets_manhattan_owner_v1"
MEDP_EDGE_FEATURES = (
    "source_dx_from_robot", "source_dy_from_robot",
    "target_dx_from_robot", "target_dy_from_robot",
    "source_distance", "target_distance", "committed_partner",
)
MEDP_EDGE_FEATURE_DIM = len(MEDP_EDGE_FEATURES)
MEDP_RETAINED_COMPONENTS = (
    "exact_d_am_d128_h8_l1_ff512_trunk",
    "dense_task_agent_edge_state",
    "shared_task_and_robot_embeddings",
    "edge_ffn_transition",
    "task_partner_relu_mlp",
    "query_context_relu_mlp",
    "same_am_glimpse_applied_twice",
    "pair_aware_pointer",
    "exact_open_owner_binding",
    "role_local_one_hot_owner_task_features",
    "legal_empty_task_partner_competition",
)
MEDP_LITE_REDUCTIONS = (
    "edge_width_32",
    "glimpse_parameters_shared_across_two_applications",
    "fusion_mlp_hidden_width_32",
    "initial_query_projection_removed",
    "no_extra_node_mlps",
    "partner_pooling_rank_48",
)

MECHANISM_ABLATIONS = (
    "no_cross_task_competition",
    "no_exact_binding",
)

MEDP_ABLATIONS = (
    "full",
    *MECHANISM_ABLATIONS,
    "no_edge_context",
    "no_pair_decoder",
)


def _default_backbone() -> DecentralizedAMConfig:
    return DecentralizedAMConfig(
        embedding_dim=128,
        n_heads=8,
        encoder_layers=1,
        feed_forward_hidden=512,
    )


def _gate_parameter(initial_value: float) -> torch.Tensor:
    return torch.tensor(math.atanh(initial_value))


@dataclass(frozen=True)
class FormalMEDPConfig:
    """D-AM-compatible trunk plus shared lightweight pair modules."""

    backbone: DecentralizedAMConfig = _default_backbone()
    edge_dim: int = 32
    recurrent_steps: int = 2
    edge_feed_forward_hidden: int = 64
    conditioner_hidden: int = 32
    partner_pool_rank: int = 48
    decoder_glimpses: int = 1
    query_fusion_contexts: int = 3
    pair_gate_init: float = 0.01
    pair_logit_limit: float = 2.0
    forward_chunk_size: int = 2048  # Zero disables policy-forward chunking.
    ablation: str = "full"

    def validate(self) -> None:
        if not isinstance(self.backbone, DecentralizedAMConfig):
            raise TypeError("formal MEDP requires the one-hot owner AM backbone config")
        self.backbone.validate()
        if (
            self.backbone.embedding_dim != 128
            or self.backbone.n_heads != 8
            or self.backbone.encoder_layers != 1
            or self.backbone.feed_forward_hidden != 512
        ):
            raise ValueError("MEDP requires the exact D-AM d128/h8/l1/ff512 trunk")
        if self.edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        if self.recurrent_steps not in {1, 2} or self.decoder_glimpses not in {1, 2}:
            raise ValueError(
                "recurrent_steps and decoder_glimpses must each be one or two"
            )
        if self.query_fusion_contexts not in {3, 4}:
            raise ValueError("query_fusion_contexts must be three or four")
        if min(
            self.edge_feed_forward_hidden,
            self.conditioner_hidden,
            self.partner_pool_rank,
        ) <= 0:
            raise ValueError("adapter, FFN, conditioner, and pool sizes must be positive")
        if not 0.0 < self.pair_gate_init < 0.1:
            raise ValueError("pair_gate_init must be in (0, 0.1)")
        if self.pair_logit_limit <= 0.0:
            raise ValueError("pair_logit_limit must be positive")
        if type(self.forward_chunk_size) is not int or self.forward_chunk_size < 0:
            raise ValueError("forward_chunk_size must be a nonnegative integer; zero disables chunking")
        if self.ablation not in MEDP_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {MEDP_ABLATIONS}, got {self.ablation!r}"
            )

    def to_dict(self) -> Dict[str, object]:
        components = list(MEDP_RETAINED_COMPONENTS)
        reductions = list(MEDP_LITE_REDUCTIONS)
        if self.recurrent_steps == 1:
            components.remove("edge_ffn_transition")
            reductions.append("inactive_edge_transition_not_registered")
        if self.decoder_glimpses == 1:
            components[components.index("same_am_glimpse_applied_twice")] = "single_am_glimpse"
            reductions.remove("glimpse_parameters_shared_across_two_applications")
            reductions.append("second_glimpse_not_registered")
        if self.query_fusion_contexts == 3:
            components[components.index("query_context_relu_mlp")] = "paper_three_context_query_relu_mlp"
        metadata = {
            "backbone": self.backbone.to_dict(),
            "edge_dim": self.edge_dim,
            "bidirectional_cross_attention": False,
            "node_updates": False,
            "recurrent_steps": self.recurrent_steps,
            "edge_feed_forward_hidden": self.edge_feed_forward_hidden,
            "conditioner_hidden": self.conditioner_hidden,
            "fusion_type": "linear_relu_linear",
            "fusion_normalization": "none",
            "fusion_residual": False,
            "fusion_gates": False,
            "task_fusion_input_dim": 2 * self.backbone.embedding_dim,
            "query_fusion_input_dim": self.query_fusion_contexts * self.backbone.embedding_dim,
            "initial_query_projection_registered": False,
            "partner_pool_rank": self.partner_pool_rank,
            "decoder_glimpses": self.decoder_glimpses,
            "pair_gate_init": self.pair_gate_init,
            "pair_logit_limit": self.pair_logit_limit,
            "forward_chunk_size": self.forward_chunk_size,
            "ablation": self.ablation,
            "architecture": (
                MEDP_ARCHITECTURE if self.query_fusion_contexts == 4
                else MEDP_PAPER_ARCHITECTURE
            ),
            "edge_feature_schema": MEDP_EDGE_FEATURE_SCHEMA,
            "edge_feature_order": list(MEDP_EDGE_FEATURES),
            "edge_feature_dim": MEDP_EDGE_FEATURE_DIM,
            "edge_transition_registered": self.recurrent_steps > 1,
            "retained_components": components,
            "parameter_and_compute_reductions": reductions,
            "parameter_sharing": {
                "am_glimpse": (
                    "D-AM module applied twice with gated second residual"
                    if self.decoder_glimpses == 2 else "D-AM module applied once"
                ),
            },
            "information_boundary": "current_observation_only",
            "positional_encoding": False,
            "lookahead": False,
        }
        # Keep legacy metadata byte-for-byte compatible with existing schema-13
        # checkpoints, whose policy_config has no query_fusion_contexts key.
        if self.query_fusion_contexts == 3:
            metadata["query_fusion_contexts"] = 3
        return metadata


class FormalMEDPPolicy(DecentralizedAMPolicy):
    """D-AM policy augmented by parameter-shared pair reasoning."""

    method_name = DE_MRST_NAME
    requires_lookahead = False

    def __init__(self, config: FormalMEDPConfig = None):
        self.medp_config = config or FormalMEDPConfig()
        self.medp_config.validate()
        super().__init__(self.medp_config.backbone)
        # Do not retain the inherited, now-unused 3d-to-d decoder projection.
        del self.context_projection
        cfg = self.medp_config
        embedding_dim = self.config.embedding_dim

        self.task_edge_projection = nn.Linear(embedding_dim, cfg.edge_dim, bias=False)
        self.agent_edge_projection = nn.Linear(embedding_dim, cfg.edge_dim, bias=False)
        self.current_edge_projection = nn.Linear(embedding_dim, cfg.edge_dim, bias=False)
        self.geometry_edge_projection = nn.Linear(MEDP_EDGE_FEATURE_DIM, cfg.edge_dim)
        self.edge_input_norm = RMSNorm(cfg.edge_dim)
        self.edge_transition = (
            EdgeTransition(
                embedding_dim,
                cfg.edge_dim,
                cfg.edge_feed_forward_hidden,
                cfg.pair_gate_init,
                1e-6,
            ) if cfg.recurrent_steps > 1 else None
        )

        self.pool_task_projection = nn.Linear(
            embedding_dim, cfg.partner_pool_rank, bias=False
        )
        self.pool_agent_projection = nn.Linear(
            embedding_dim, cfg.partner_pool_rank, bias=False
        )
        self.pool_edge_score = nn.Linear(cfg.edge_dim, 1, bias=False)
        self.task_conditioner = nn.Sequential(
            nn.Linear(2 * embedding_dim, cfg.conditioner_hidden),
            nn.ReLU(),
            nn.Linear(cfg.conditioner_hidden, embedding_dim),
        )
        self.query_conditioner = nn.Sequential(
            nn.Linear(cfg.query_fusion_contexts * embedding_dim, cfg.conditioner_hidden),
            nn.ReLU(),
            nn.Linear(cfg.conditioner_hidden, embedding_dim),
        )
        self.glimpse_edge_score = nn.Parameter(
            torch.empty(self.config.n_heads, cfg.edge_dim)
        )
        nn.init.xavier_uniform_(self.glimpse_edge_score)
        self.glimpse_pair_gates = nn.Parameter(
            torch.full(
                (self.config.n_heads,),
                math.atanh(cfg.pair_gate_init),
            )
        )
        self.second_glimpse_gate = (
            nn.Parameter(_gate_parameter(cfg.pair_gate_init))
            if cfg.decoder_glimpses == 2 else None
        )
        self.pointer_edge_score = nn.Linear(cfg.edge_dim, 1, bias=False)
        self.pointer_pair_gate = nn.Parameter(_gate_parameter(cfg.pair_gate_init))
        self.pool_norm_factor = cfg.partner_pool_rank**-0.5

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
        batch_size, task_count, _ = task_inputs.shape
        if agent_inputs.size(0) != batch_size or tuple(action_mask.shape) != (
            batch_size,
            task_count,
        ):
            raise ValueError("batch and mask shapes are inconsistent")
        if tuple(agent_index.shape) != (batch_size,):
            raise ValueError("agent_index must have shape [batch]")
        if not torch.is_floating_point(task_inputs) or not torch.is_floating_point(
            agent_inputs
        ):
            raise TypeError("policy inputs must be floating-point tensors")
        if task_inputs.size(0) == 0 or task_inputs.size(1) == 0 or agent_inputs.size(1) == 0:
            raise ValueError("policy inputs must contain non-empty sets")
        if task_inputs.dtype != agent_inputs.dtype:
            raise TypeError("task and agent inputs must have the same dtype")
        if action_mask.dtype != torch.bool:
            raise TypeError("action_mask must use torch.bool")
        if agent_index.dtype != torch.long:
            raise TypeError("agent_index must use torch.long")
        if any(
            tensor.device != task_inputs.device
            for tensor in (agent_inputs, action_mask, agent_index)
        ):
            raise ValueError("all policy inputs must be on the same device")
        self._runtime_assert(
            (agent_index >= 0) & (agent_index < agent_inputs.size(1)),
            IndexError,
            "agent index is out of range",
        )
        self._runtime_assert(
            ~action_mask.all(dim=-1),
            ValueError,
            "each decision needs at least one legal task",
        )
        self._runtime_assert(
            torch.isfinite(task_inputs).all() & torch.isfinite(agent_inputs).all(),
            ValueError,
            "task and agent inputs must be finite",
        )

    def _partner_structure(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, task_count, _ = task_inputs.shape
        agent_count = agent_inputs.size(1)
        batch = torch.arange(batch_size, device=task_inputs.device)
        is_mbr = agent_inputs[..., 0] >= 0.5
        current_is_mbr = is_mbr[batch, agent_index]
        mbr_count = is_mbr.sum(dim=1, dtype=torch.long)
        dor_count = agent_count - mbr_count
        self._runtime_assert(
            (mbr_count >= 1) & (dor_count >= 1),
            ValueError,
            "MEDP needs both robot roles",
        )
        complementary = torch.where(current_is_mbr[:, None], ~is_mbr, is_mbr)
        candidate_mask = ~complementary[:, None].expand(-1, task_count, -1)
        task_owner_indices = self._assigned_owner_indices(task_inputs)
        owner_index = torch.where(
            current_is_mbr[:, None],
            task_owner_indices[..., 1],
            task_owner_indices[..., 0],
        )
        empty, opened = task_allocation_masks(task_inputs)
        legal_open = ~action_mask & opened
        self._runtime_assert(
            ~legal_open | (owner_index >= 0), ValueError,
            "a legal open task must identify its committed complementary owner",
        )
        fixed_partner = (
            opened
            & (owner_index >= 0)
        )
        partner_index = owner_index.clamp_min(0)
        partner_is_mbr = is_mbr.gather(1, partner_index)
        self._runtime_assert(
            ~(fixed_partner & (partner_is_mbr != (~current_is_mbr[:, None]))),
            ValueError,
            "open-task owner does not identify a complementary robot",
        )
        owner_mask = ~F.one_hot(partner_index, num_classes=agent_count).to(torch.bool)
        task_to_agent_mask = torch.where(
            fixed_partner[..., None], owner_mask, candidate_mask
        )
        if self.medp_config.ablation == "no_exact_binding":
            # Keep the committed-owner relation in fixed_partner for geometry
            # and diagnostics. Only the hard attention restriction is removed.
            task_to_agent_mask = candidate_mask
        legal_empty = (
            ~action_mask
            & empty
        )
        return task_to_agent_mask, partner_index, fixed_partner, legal_empty

    def _geometry_features(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        partner_index: torch.Tensor,
        fixed_partner: torch.Tensor,
    ) -> torch.Tensor:
        """Relative offsets, normalized travel distances and committed owner."""
        source = task_inputs[
            ..., (TASK_SOURCE_X_FEATURE_INDEX, TASK_SOURCE_Y_FEATURE_INDEX)
        ].unsqueeze(2)
        destination = task_inputs[
            ..., (TASK_DESTINATION_X_FEATURE_INDEX, TASK_DESTINATION_Y_FEATURE_INDEX)
        ].unsqueeze(2)
        agent_position = agent_inputs[
            ..., (AGENT_RELATIVE_X_FEATURE_INDEX, AGENT_RELATIVE_Y_FEATURE_INDEX)
        ].unsqueeze(1)
        source_offset = source - agent_position
        destination_offset = destination - agent_position
        # The simulator uses Manhattan distance; offsets already share the
        # observation's spatial normalization.
        source_distance = source_offset.abs().sum(dim=-1, keepdim=True)
        destination_distance = destination_offset.abs().sum(dim=-1, keepdim=True)
        owner_match = F.one_hot(
            partner_index, num_classes=agent_inputs.size(1)
        ).to(task_inputs.dtype)
        owner_match = (
            owner_match * fixed_partner[..., None].to(task_inputs.dtype)
        ).unsqueeze(-1)
        return torch.cat(
            (
                source_offset,
                destination_offset,
                source_distance,
                destination_distance,
                owner_match,
            ),
            dim=-1,
        )

    def _encode_components(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        batch_size, task_count, _ = task_inputs.shape
        task_embedding, agent_embedding = self._embed_inputs(
            task_inputs, agent_inputs,
        )
        global_token = self.global_token.expand(batch_size, -1, -1)
        encoded_nodes, graph_context, encoded_global = self.encoder(
            torch.cat((task_embedding, agent_embedding, global_token), dim=1)
        )
        task_state = encoded_nodes[:, :task_count]
        agent_state = encoded_nodes[:, task_count:]
        batch = torch.arange(batch_size, device=task_inputs.device)
        current_agent = agent_state[batch, agent_index]
        query_context = torch.cat((current_agent, graph_context, encoded_global.squeeze(1)), dim=-1)
        task_mask, partner_index, fixed_partner, legal_empty = self._partner_structure(
            task_inputs, agent_inputs, action_mask, agent_index
        )
        geometry = self._geometry_features(
            task_inputs, agent_inputs, partner_index, fixed_partner
        )
        edge_state = self.edge_input_norm(
            self.task_edge_projection(task_state).unsqueeze(2)
            + self.agent_edge_projection(agent_state).unsqueeze(1)
            + self.current_edge_projection(current_agent)[:, None, None]
            + self.geometry_edge_projection(geometry)
        )
        for _ in range(self.medp_config.recurrent_steps - 1):
            assert self.edge_transition is not None
            edge_state = self.edge_transition(edge_state, task_state, agent_state)
        return (
            task_state,
            agent_state,
            edge_state,
            query_context,
            partner_index,
            fixed_partner,
            task_mask,
            legal_empty,
        )

    def _partner_pooling(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
        task_state: torch.Tensor,
        agent_state: torch.Tensor,
        edge_state: torch.Tensor,
        partner_index: torch.Tensor,
        fixed_partner: torch.Tensor,
        task_mask: Optional[torch.Tensor] = None,
        legal_empty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The edge-context ablation removes every learned use of e_ij while
        # retaining the node compatibility, masks, exact binding and pooling
        # rules.  Keeping the registered modules preserves parameter shapes.
        pooling_edge_state = (
            torch.zeros_like(edge_state)
            if self.medp_config.ablation == "no_edge_context"
            else edge_state
        )
        compatibility = self.pool_norm_factor * torch.matmul(
            self.pool_task_projection(task_state),
            self.pool_agent_projection(agent_state).transpose(1, 2),
        )
        compatibility = compatibility + self.pool_edge_score(
            pooling_edge_state
        ).squeeze(-1)
        if task_mask is None or legal_empty is None:
            batch = torch.arange(task_inputs.size(0), device=task_inputs.device)
            is_cr = agent_inputs[..., 0] >= 0.5
            complementary = is_cr != is_cr[batch, agent_index, None]
            owner_mask = ~F.one_hot(partner_index, agent_inputs.size(1)).bool()
            task_mask = torch.where(
                fixed_partner[..., None], owner_mask, ~complementary[:, None, :]
            )
            if self.medp_config.ablation == "no_exact_binding":
                task_mask = (~complementary[:, None, :]).expand(
                    -1, task_inputs.size(1), -1
                )
            legal_empty = ~action_mask & task_allocation_masks(task_inputs)[0]
        poolable_tasks = legal_empty
        if self.medp_config.ablation == "no_cross_task_competition":
            # Remove only the partner-to-task factor, not learned partner scores.
            pooled_weight = _masked_softmax(compatibility, task_mask, dim=-1)
            pooled_weight = torch.where(
                poolable_tasks[..., None], pooled_weight, torch.zeros_like(pooled_weight)
            )
        else:
            task_to_partner = _masked_softmax(compatibility, task_mask, dim=-1)
            competition_mask = task_mask | (~poolable_tasks)[..., None]
            partner_to_task = _masked_softmax(compatibility, competition_mask, dim=-2)
            pooled_weight = task_to_partner * partner_to_task
            pooled_weight = pooled_weight / pooled_weight.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(pooled_weight.dtype).eps)
            pooled_weight = torch.where(
                poolable_tasks[..., None], pooled_weight, torch.zeros_like(pooled_weight)
            )
        if self.medp_config.ablation == "no_exact_binding":
            # Open tasks get ordinary learned partner attention. They remain
            # excluded from the cross-task competition among legal empty tasks.
            open_task_weight = task_to_partner
        else:
            open_task_weight = F.one_hot(
                partner_index, num_classes=agent_state.size(1)
            ).to(dtype=pooled_weight.dtype)
        partner_weight = torch.where(
            fixed_partner[..., None], open_task_weight, pooled_weight
        )
        partner_message = torch.matmul(partner_weight, agent_state)
        pair_edge = torch.einsum(
            "bta,btae->bte", partner_weight, pooling_edge_state
        )
        pair_evidence = torch.sum(partner_weight * compatibility, dim=-1).tanh()
        pair_evidence = pair_evidence.masked_fill(action_mask, 0.0)
        return partner_message, pair_edge, pair_evidence, partner_weight

    def _pair_biased_glimpse(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        action_mask: torch.Tensor,
        pair_edge: torch.Tensor,
        pair_evidence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, task_count, embedding_dim = memory.shape
        query_count = query.size(1)
        glimpse = self.glimpse
        query_heads = torch.matmul(
            query.reshape(-1, embedding_dim), glimpse.W_query
        ).view(glimpse.n_heads, batch_size, query_count, -1)
        memory_flat = memory.reshape(-1, embedding_dim)
        key_heads = torch.matmul(memory_flat, glimpse.W_key).view(
            glimpse.n_heads, batch_size, task_count, -1
        )
        value_heads = torch.matmul(memory_flat, glimpse.W_val).view(
            glimpse.n_heads, batch_size, task_count, -1
        )
        compatibility = glimpse.norm_factor * torch.matmul(
            query_heads, key_heads.transpose(2, 3)
        )
        edge_bias = torch.einsum("bte,he->hbt", pair_edge, self.glimpse_edge_score)
        pair_bias = torch.tanh(self.glimpse_pair_gates)[:, None, None, None] * (
            pair_evidence[None, :, None, :] + torch.tanh(edge_bias)[:, :, None, :]
        )
        compatibility = compatibility + self.medp_config.pair_logit_limit * pair_bias
        expanded_mask = action_mask.view(
            1, batch_size, query_count, task_count
        ).expand_as(compatibility)
        attention = _masked_softmax(compatibility, expanded_mask, dim=-1)
        heads = torch.matmul(attention, value_heads)
        return torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(
                -1, glimpse.n_heads * glimpse.val_dim
            ),
            glimpse.W_out.view(-1, glimpse.embed_dim),
        ).view(batch_size, query_count, glimpse.embed_dim)

    def _pointer_logits(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        action_mask: torch.Tensor,
        pair_edge: torch.Tensor,
        pair_evidence: torch.Tensor,
    ) -> torch.Tensor:
        projected_query = torch.matmul(query, self.pointer.w_query)
        projected_key = torch.matmul(memory, self.pointer.w_key)
        compatibility = self.pointer.norm_factor * torch.matmul(
            projected_query, projected_key.transpose(1, 2)
        )
        pair_residual = pair_evidence + torch.tanh(
            self.pointer_edge_score(pair_edge).squeeze(-1)
        )
        compatibility = compatibility + (
            self.medp_config.pair_logit_limit
            * torch.tanh(self.pointer_pair_gate)
            * pair_residual[:, None, :]
        )
        logits = self.pointer.tanh_clipping * torch.tanh(compatibility).squeeze(1)
        return logits.masked_fill(action_mask, -torch.inf)

    def _forward_impl(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            task_state,
            agent_state,
            edge_state,
            query_context,
            partner_index,
            fixed_partner,
            task_mask,
            legal_empty,
        ) = self._encode_components(
            task_inputs, agent_inputs, action_mask, agent_index
        )
        partner_message, pair_edge, pair_evidence, _ = self._partner_pooling(
            task_inputs,
            agent_inputs,
            action_mask,
            agent_index,
            task_state,
            agent_state,
            edge_state,
            partner_index,
            fixed_partner,
            task_mask,
            legal_empty,
        )
        task_state = self.task_conditioner(torch.cat((task_state, partner_message), dim=-1))
        if self.medp_config.query_fusion_contexts == 3:
            query_input = query_context
        else:
            legal = (~action_mask).to(dtype=task_state.dtype)
            legal_count = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
            partner_summary = torch.sum(
                partner_message * legal[..., None], dim=1
            ) / legal_count
            query_input = torch.cat((query_context, partner_summary), dim=-1)
        query = self.query_conditioner(query_input).unsqueeze(1)
        if self.medp_config.ablation == "no_pair_decoder":
            first_glimpse = self.glimpse(
                query,
                task_state,
                mask=action_mask.unsqueeze(1),
            )
        else:
            first_glimpse = self._pair_biased_glimpse(
                query, task_state, action_mask, pair_edge, pair_evidence
            )
        decoder_state = query + first_glimpse
        if self.medp_config.decoder_glimpses == 2:
            if self.medp_config.ablation == "no_pair_decoder":
                second_glimpse = self.glimpse(
                    decoder_state,
                    task_state,
                    mask=action_mask.unsqueeze(1),
                )
            else:
                second_glimpse = self._pair_biased_glimpse(
                    decoder_state, task_state, action_mask, pair_edge, pair_evidence
                )
            decoder_state = decoder_state + torch.tanh(
                self.second_glimpse_gate
            ) * second_glimpse
        if self.medp_config.ablation == "no_pair_decoder":
            logits = self.pointer.logits(decoder_state, task_state, action_mask)
        else:
            logits = self._pointer_logits(
                decoder_state, task_state, action_mask, pair_edge, pair_evidence
            )
        return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)

    def forward(
        self,
        task_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(task_inputs, agent_inputs, action_mask, agent_index)
        self._validate_owner_features(task_inputs, agent_inputs)
        chunk_size = self.medp_config.forward_chunk_size
        if chunk_size == 0 or task_inputs.size(0) <= chunk_size:
            return self._forward_impl(
                task_inputs, agent_inputs, action_mask, agent_index
            )
        probability_chunks = []
        log_probability_chunks = []
        for start in range(0, task_inputs.size(0), chunk_size):
            end = min(start + chunk_size, task_inputs.size(0))
            probabilities, log_probabilities = self._forward_impl(
                task_inputs[start:end],
                agent_inputs[start:end],
                action_mask[start:end],
                agent_index[start:end],
            )
            probability_chunks.append(probabilities)
            log_probability_chunks.append(log_probabilities)
        return torch.cat(probability_chunks), torch.cat(log_probability_chunks)
