"""Shared raw observations for decentralized HetMRTA and MEDP policies.

This module deliberately excludes any MBR-DOR pairing lookahead. All policies
receive the same communicated information and encoding: Hete, D-AM and formal
MEDP use remaining roles, one-hot assignments, location/duration attributes and
actual completion. Completion becomes true only after the WR finishes processing.
The redundant feasible-assignment flag is absent from current policy inputs.
The simulator's four-state execution lifecycle is unchanged. Owners are communicated assignment
records, not future actions.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from .event_env import DecentralizedMRSEnv, Role, TaskStatus


AGENT_FEATURE_DIM = 8

# Base attributes, with robot identity bits inserted after remaining roles.
TASK_BASE_FEATURES = (
    "remaining_mbr", "remaining_dor",
    "source_dx", "source_dy", "destination_dx", "destination_dy",
    "pickup_time", "handling_time", "completed",
)
TASK_BASE_FEATURE_DIM = len(TASK_BASE_FEATURES)
TASK_OBSERVATION_SCHEMA = "marsupial_compact9_owner_onehot_completion_dai_order_v4"
TASK_OWNER_START_INDEX = 2
# Completion is the last column at every fleet size. Protocols record its
# nonnegative position using task_feature_names(...).index("completed").
TASK_COMPLETION_FEATURE_INDEX = -1
TASK_STATUS_ENCODING = "task_completed_binary"
TASK_REQUIREMENT_INDICES = tuple(
    TASK_BASE_FEATURES.index(name) for name in ("remaining_mbr", "remaining_dor")
)


def task_allocation_masks(task_inputs):
    """Empty/open allocation state from the two missing-role entries."""
    needs_cr = task_inputs[..., TASK_REQUIREMENT_INDICES[0]] >= 0.5
    needs_wr = task_inputs[..., TASK_REQUIREMENT_INDICES[1]] >= 0.5
    return needs_cr & needs_wr, needs_cr ^ needs_wr


def task_feature_dim(n_mbr: int, n_dor: int) -> int:
    if any(type(count) is not int or count < 1 for count in (n_mbr, n_dor)):
        raise ValueError("one-hot owner fleet counts must be positive integers")
    return TASK_BASE_FEATURE_DIM + n_mbr + n_dor


def task_feature_names(n_mbr: int, n_dor: int) -> Tuple[str, ...]:
    """Exact ordered columns recorded in the immutable observation contract."""
    task_feature_dim(n_mbr, n_dor)
    return (*TASK_BASE_FEATURES[:TASK_OWNER_START_INDEX],
            *(f"cr_owner_{j}" for j in range(n_mbr)),
            *(f"wr_owner_{j}" for j in range(n_dor)),
            *TASK_BASE_FEATURES[TASK_OWNER_START_INDEX:])


def owner_one_hot_block(task_features):
    """Identity bits between the two requirement entries and physical suffix."""
    suffix_width = TASK_BASE_FEATURE_DIM - TASK_OWNER_START_INDEX
    return task_features[..., TASK_OWNER_START_INDEX:-suffix_width]


def insert_owner_one_hot(task_features, owner_indices, n_mbr, n_dor):
    """Insert CR then WR identity bits after requirements; unassigned is zero.

    owner_indices are simulator-side integer rows, not extra policy inputs.
    Both observation builders use this conversion to guarantee column parity.
    """
    task_feature_dim(n_mbr, n_dor)
    cr = owner_indices[..., 0]
    wr = owner_indices[..., 1] - n_mbr
    cr_bits = cr[..., None] == torch.arange(n_mbr, device=cr.device)
    wr_bits = wr[..., None] == torch.arange(n_dor, device=wr.device)
    split = TASK_OWNER_START_INDEX
    return torch.cat((task_features[..., :split], cr_bits.to(task_features.dtype),
                      wr_bits.to(task_features.dtype), task_features[..., split:]), dim=-1)


AGENT_FEATURES: Tuple[str, ...] = (
    "is_mbr",
    "is_dor",
    "remaining_travel_time",
    "remaining_execution_time",
    "current_waiting_time",
    "relative_x",
    "relative_y",
    "assignment_state",
)


def validate_protocol_observation_schema(protocol, method=None) -> None:
    """Validate the single current observation contract before training."""
    boundary = protocol.get("information_boundary", {})
    boundary = boundary.get("method_observations", {}).get(method, {}) if method else boundary
    n_mbr, n_dor = boundary.get("n_mbr"), boundary.get("n_dor")
    if (
        type(n_mbr) is not int or type(n_dor) is not int
        or min(n_mbr, n_dor) < 1
        or boundary.get("task_observation_schema") != TASK_OBSERVATION_SCHEMA
        or boundary.get("task_feature_dim") != task_feature_dim(n_mbr, n_dor)
        or boundary.get("owner_encoding") != "role_local_one_hot_zero_unassigned"
        or boundary.get("base_task_feature_dim") != TASK_BASE_FEATURE_DIM
        or boundary.get("task_status_encoding") != TASK_STATUS_ENCODING
        or boundary.get("task_feature_order") != list(task_feature_names(n_mbr, n_dor))
    ):
        raise ValueError(
            "Common one-hot observations require a newly frozen protocol with "
            f"information_boundary.method_observations.{method} recording "
            f"task_observation_schema={TASK_OBSERVATION_SCHEMA!r}, "
            "n_mbr, n_dor, task_feature_dim=9+n_mbr+n_dor, base_task_feature_dim, "
            "task_status_encoding='task_completed_binary', exact task_feature_order and "
            "owner_encoding='role_local_one_hot_zero_unassigned'. Do not overwrite a legacy protocol."
        )
    for setting in protocol.get("decentralized_training", {}).get("training_settings", []):
        if (setting.get("n_mbr"), setting.get("n_dor")) != (n_mbr, n_dor):
            raise ValueError("one-hot owner schema fleet differs from the training setting")


def validate_protocol_ablation(protocol, method, ablation="full"):
    """Do not silently run a new intervention under an old full-model protocol."""
    if method not in {"medp_formal", "medp_1r"} and ablation != "full":
        raise ValueError("MEDP ablations only apply to formal MEDP methods")
    architectures = protocol.get("ablation_architectures")
    if architectures is not None and method not in architectures:
        raise ValueError("method is not authorized by this ablation protocol")
    if method not in {"medp_formal", "medp_1r"}:
        return
    allowed = protocol.get("decentralized_training", {}).get("ablation_variants", ["full"])
    if ablation not in allowed:
        raise ValueError(f"ablation {ablation!r} is not authorized by the frozen protocol")
    variants = protocol.get("ablation_architectures", {}).get(method)
    if variants is not None and ablation not in variants:
        raise ValueError("frozen protocol is missing the selected ablation architecture")


def validate_model_observation_protocol(protocol, method, model, *, runtime_overrides=()):
    """Reject architecture drift in the new one-hot study before training."""
    config = getattr(model, "medp_config", model.config)
    ablation = getattr(config, "ablation", "full")
    variants = protocol.get("ablation_architectures", {}).get(method)
    if variants is not None:
        validate_protocol_ablation(protocol, method, ablation)
        expected = variants[ablation]
    else:
        expected = protocol.get("method_architectures", {}).get(method)
    if expected is None:
        if getattr(model, "requires_owner_one_hot", False):
            raise ValueError("one-hot protocol must freeze the method architecture")
        return
    if set(runtime_overrides) - {"forward_chunk_size"}:
        raise ValueError("only forward_chunk_size may be overridden by a development runtime probe")
    expected_config = {k: v for k, v in expected.get("policy_config", {}).items() if k not in runtime_overrides}
    actual_config = {k: v for k, v in config.to_dict().items() if k not in runtime_overrides}
    if expected_config != actual_config:
        raise ValueError("model configuration differs from the frozen method architecture")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if expected.get("parameter_count") != count:
        raise ValueError("parameter count differs from the frozen method architecture")


@dataclass(frozen=True)
class ObservationScales:
    location: float
    time: float

    @classmethod
    def for_environment(
        cls,
        env: DecentralizedMRSEnv,
        location: float = 100.0,
    ) -> "ObservationScales":
        if location <= 0:
            raise ValueError("location scale must be positive")
        released_tasks = [
            task for task, released in zip(env.tasks, env.task_released) if released
        ]
        time_scale = max(
            [1.0]
            + [
                task.pickup_time
                + task.handling_time
                + env.dock_time
                + env.detach_time
                + 2.0 * location / env.speed
                for task in released_tasks
            ]
        )
        return cls(location=float(location), time=time_scale)


@dataclass(frozen=True)
class PolicyObservation:
    task_features: torch.Tensor
    agent_features: torch.Tensor
    action_mask: torch.Tensor
    agent_index: int
    role: Role
    robot_id: int
    scales: ObservationScales
    task_observation_schema: str = TASK_OBSERVATION_SCHEMA

    @property
    def has_legal_action(self) -> bool:
        return bool((~self.action_mask).any().item())

    def batched(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.task_features.unsqueeze(0).to(device),
            self.agent_features.unsqueeze(0).to(device),
            self.action_mask.unsqueeze(0).to(device),
            torch.tensor([self.agent_index], dtype=torch.long, device=device),
        )


def _robot_view(env: DecentralizedMRSEnv, role: Role, robot_id: int):
    if role == Role.MBR:
        return (
            env.mbr_state[robot_id],
            env.mbr_busy[robot_id],
            env.mbr_waiting[robot_id],
            env.mbr_travel_until[robot_id],
        )
    return (
        env.dor_state[robot_id],
        env.dor_busy[robot_id],
        env.dor_waiting[robot_id],
        env.dor_travel_until[robot_id],
    )


def build_policy_observation(
    env: DecentralizedMRSEnv,
    role: Role,
    robot_id: int,
    enforce_open_limit: bool,
    scales: ObservationScales = None,
) -> PolicyObservation:
    """Build the common current task and robot state without lookahead."""
    env._validate_robot(role, robot_id)
    scales = scales or ObservationScales.for_environment(env)
    deciding_state, _, _, _ = _robot_view(env, role, robot_id)
    deciding_position = deciding_state.position

    task_rows = []
    for task_id, (task, state) in enumerate(zip(env.tasks, env.task_state)):
        if not env.task_released[task_id]:
            task_rows.append([0.0] * TASK_BASE_FEATURE_DIM)
            continue
        active = state.status in (TaskStatus.EMPTY, TaskStatus.OPEN)
        remaining_mbr = float(active and state.mbr_id is None)
        remaining_dor = float(active and state.dor_id is None)
        source_dx = (task.source[0] - deciding_position[0]) / scales.location
        source_dy = (task.source[1] - deciding_position[1]) / scales.location
        destination_dx = (task.destination[0] - deciding_position[0]) / scales.location
        destination_dy = (task.destination[1] - deciding_position[1]) / scales.location
        attributes = [
            source_dx, source_dy, destination_dx, destination_dy,
            task.pickup_time / scales.time, task.handling_time / scales.time,
        ]
        completed = float(state.status == TaskStatus.COMPLETED)
        task_rows.append([remaining_mbr, remaining_dor, *attributes, completed])

    agent_rows = []
    for global_index in range(env.n_mbr + env.n_dor):
        other_role, other_id = env.agent_from_global_index(global_index)
        state, busy, waiting, travel_until = _robot_view(env, other_role, other_id)
        remaining_travel = max(0.0, travel_until - env.time) if busy else 0.0
        remaining_execution = (
            max(0.0, state.available_time - max(env.time, travel_until)) if busy else 0.0
        )
        waiting_time = max(0.0, env.time - state.available_time) if waiting else 0.0
        agent_rows.append(
            [
                float(other_role == Role.MBR),
                float(other_role == Role.DOR),
                remaining_travel / scales.time,
                remaining_execution / scales.time,
                waiting_time / scales.time,
                (state.position[0] - deciding_position[0]) / scales.location,
                (state.position[1] - deciding_position[1]) / scales.location,
                1.0 if busy else (0.0 if waiting else -1.0),
            ]
        )

    legal = env.role_mask(role, robot_id, enforce_open_limit=enforce_open_limit)
    task_features = torch.tensor(task_rows, dtype=torch.float32)
    task_owner_indices = torch.tensor([
        [
            -1 if state.mbr_id is None else state.mbr_id,
            -1 if state.dor_id is None else env.n_mbr + state.dor_id,
        ]
        for state in env.task_state
    ], dtype=torch.long)
    task_features = insert_owner_one_hot(
        task_features, task_owner_indices, env.n_mbr, env.n_dor
    )
    observation = PolicyObservation(
        task_features=task_features,
        agent_features=torch.tensor(agent_rows, dtype=torch.float32),
        action_mask=torch.tensor([not value for value in legal], dtype=torch.bool),
        agent_index=env.global_agent_index(role, robot_id),
        role=role,
        robot_id=robot_id,
        scales=scales,
        task_observation_schema=TASK_OBSERVATION_SCHEMA,
    )
    task_dim = task_feature_dim(env.n_mbr, env.n_dor)
    if observation.task_features.shape != (len(env.tasks), task_dim):
        raise AssertionError("unexpected task observation shape")
    if observation.agent_features.shape != (env.n_mbr + env.n_dor, AGENT_FEATURE_DIM):
        raise AssertionError("unexpected agent observation shape")
    return observation
