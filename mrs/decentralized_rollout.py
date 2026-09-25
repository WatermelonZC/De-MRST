"""Event rollouts, constrained flashforward, and repeat-average REINFORCE.

The current learned policies consume the same communicated task and robot state.
"""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .core import (
    PlanMetrics,
    TaskSpec,
    deadlock_penalty,
    normalize_objective_mode,
    protocol_episode_timeout,
    protocol_failure_penalty,
)
from .event_env import DecentralizedMRSEnv, Role
from .observations import (
    TASK_OBSERVATION_SCHEMA, ObservationScales, PolicyObservation, build_policy_observation,
)
from .scalar_execution import (
    AssignmentClaim,
    ProtocolRound,
    ScalarExecutionMetrics,
    ScalarExecutionProtocol,
)


AgentKey = Tuple[Role, int]
FixedPartnerMap = Mapping[int, Tuple[int, int]]


def _normalize_fixed_partner_map(
    fixed_partner_map: Optional[FixedPartnerMap],
    task_count: int,
    n_mbr: int,
    n_dor: int,
) -> Optional[Mapping[int, Tuple[int, int]]]:
    if fixed_partner_map is None:
        return None
    expected = set(range(task_count))
    actual = {int(task_id) for task_id in fixed_partner_map}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"fixed partner map must cover every task exactly once; "
            f"missing={missing}, extra={extra}"
        )
    normalized = {}
    for raw_task_id, pair in fixed_partner_map.items():
        task_id = int(raw_task_id)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"fixed partner for task {task_id} must be (mbr_id, dor_id)"
            )
        mbr_id, dor_id = (int(pair[0]), int(pair[1]))
        if not 0 <= mbr_id < n_mbr or not 0 <= dor_id < n_dor:
            raise ValueError(
                f"fixed partner for task {task_id} is outside the fleet: "
                f"(mbr={mbr_id}, dor={dor_id})"
            )
        normalized[task_id] = (mbr_id, dor_id)
    return normalized


@dataclass(frozen=True)
class DecisionRecord:
    task_features: torch.Tensor
    agent_features: torch.Tensor
    action_mask: torch.Tensor
    agent_index: int
    action: int
    role: Role
    robot_id: int
    observation_time: float
    origin_time: float
    state_version: Optional[int] = None
    token_sequence: Optional[int] = None
    task_observation_schema: str = TASK_OBSERVATION_SCHEMA

    @property
    def used_flashforward(self) -> bool:
        return self.observation_time > self.origin_time + 1e-12


@dataclass(frozen=True)
class EpisodeOutcome:
    cost: float
    reward: float
    success: bool
    failure_reason: str
    metrics: Optional[PlanMetrics]
    plan: Tuple[Tuple[int, int, int], ...]
    records: Tuple[DecisionRecord, ...]
    decision_time_s: float
    deferred_attempts: int
    flashforward_decisions: int
    execution_metrics: Optional[ScalarExecutionMetrics] = None
    protocol_rounds: Tuple[ProtocolRound, ...] = ()


@dataclass(frozen=True)
class WeightedDecision:
    record: DecisionRecord
    cost_advantage: float
    source_group: Optional[int] = None


@dataclass(frozen=True)
class RepeatAverageBatch:
    outcomes: Tuple[EpisodeOutcome, ...]
    mean_cost: float
    decisions: Tuple[WeightedDecision, ...]


@dataclass(frozen=True)
class ReinforceLoss:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    decision_count: int


def _policy_device(policy: nn.Module) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _select_action(
    policy: nn.Module,
    observation: PolicyObservation,
    sample: bool,
    generator: torch.Generator,
) -> Tuple[int, float]:
    device = _policy_device(policy)
    task_inputs, agent_inputs, action_mask, agent_index = observation.batched(device)
    started = perf_counter()
    with torch.no_grad():
        probabilities, log_probabilities = policy(
            task_inputs,
            agent_inputs,
            action_mask,
            agent_index,
        )
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("policy produced non-finite probabilities")
        if sample:
            action = int(torch.multinomial(probabilities[0], 1, generator=generator).item())
        else:
            action = int(probabilities[0].argmax().item())
        if observation.action_mask[action]:
            raise AssertionError("policy selected a masked task")
        if not torch.isfinite(log_probabilities[0, action]):
            raise FloatingPointError("selected action has a non-finite log probability")
    return action, perf_counter() - started


def run_decentralized_episode(
    policy: nn.Module,
    tasks: Sequence[TaskSpec],
    n_mbr: int,
    n_dor: int,
    speed: float,
    *,
    sample: bool,
    training_cfm: bool,
    env_seed: int,
    action_seed: int,
    max_open_tasks: Optional[int] = None,
    time_limit: Optional[float] = None,
    dock_time: float = 8.0,
    detach_time: float = 8.0,
    fixed_partner_map: Optional[FixedPartnerMap] = None,
    mbr_speed: Optional[float] = None,
    dor_speed: Optional[float] = None,
    mbr_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    dor_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    objective_mode: str = "v4",
    task_release_times: Optional[Sequence[float]] = None,
    travel_time_factors: Optional[Sequence[float]] = None,
    operation_time_factors: Optional[Sequence[float]] = None,
    execution_time_factors: Optional[Sequence[float]] = None,
) -> EpisodeOutcome:
    """Execute one episode under sequential decentralized decisions.

    During CFM, a blocked robot keeps its original availability state while
    later event decisions are observed. Its eventual action therefore takes
    effect from the original decision time, exactly as the training-only
    flashforward maneuver specifies.
    """
    if not tasks:
        raise ValueError("at least one task is required")
    objective_mode = normalize_objective_mode(objective_mode)
    fixed_partner_map = _normalize_fixed_partner_map(
        fixed_partner_map, len(tasks), n_mbr, n_dor
    )
    env = DecentralizedMRSEnv(
        tasks,
        n_mbr,
        n_dor,
        speed,
        max_open_tasks=max_open_tasks,
        seed=env_seed,
        time_limit=(
            time_limit
            if time_limit is not None
            else (
                deadlock_penalty(tasks)
                if objective_mode == "v4"
                else protocol_episode_timeout(len(tasks))
            )
        ),
        dock_time=dock_time,
        detach_time=detach_time,
        mbr_speed=mbr_speed,
        dor_speed=dor_speed,
        mbr_initial_positions=mbr_initial_positions,
        dor_initial_positions=dor_initial_positions,
        objective_mode=objective_mode,
        task_release_times=task_release_times,
        travel_time_factors=travel_time_factors,
        operation_time_factors=operation_time_factors,
        execution_time_factors=execution_time_factors,
    )
    scales = ObservationScales.for_environment(
        env,
    )
    generator = _make_generator(_policy_device(policy), action_seed)
    was_training = policy.training
    policy.eval()

    protocol = None if training_cfm else ScalarExecutionProtocol(env)
    ready: List[AgentKey] = list(
        protocol.initial_decisions() if protocol is not None else env.initial_decisions()
    )
    deferred: List[AgentKey] = []
    origins = {agent: env.time for agent in ready}
    records: List[DecisionRecord] = []
    decision_time = 0.0
    deferred_attempts = 0
    failure_reason = ""
    max_decisions = 2 * len(tasks)
    event_steps = 0

    def try_agent(agent: AgentKey, allow_defer: bool) -> bool:
        nonlocal decision_time, deferred_attempts
        role, robot_id = agent
        if not env.is_available(role, robot_id):
            return True
        context = (
            protocol.begin_round(agent, (agent,))
            if protocol is not None
            else None
        )
        strategy_started = perf_counter() if protocol is not None else 0.0
        observation = build_policy_observation(
            env,
            role,
            robot_id,
            enforce_open_limit=training_cfm,
            scales=scales,
        )
        if fixed_partner_map is not None:
            allowed = torch.tensor(
                [
                    (
                        fixed_partner_map[task_id][0] == robot_id
                        if role == Role.MBR
                        else fixed_partner_map[task_id][1] == robot_id
                    )
                    for task_id in range(len(tasks))
                ],
                dtype=torch.bool,
            )
            observation = replace(
                observation,
                action_mask=observation.action_mask | ~allowed,
            )
        if not observation.has_legal_action:
            if protocol is not None:
                protocol.add_strategy_time(perf_counter() - strategy_started)
                protocol.record_noop(context)
            if allow_defer:
                deferred_attempts += 1
            return False
        action, elapsed = _select_action(policy, observation,  sample, generator)
        decision_time += elapsed
        if protocol is not None:
            protocol.add_strategy_time(perf_counter() - strategy_started)
            protocol.commit(
                context,
                (AssignmentClaim(agent, action),),
                enforce_open_limit=training_cfm,
            )
        else:
            env.assign(
                role,
                robot_id,
                action,
                enforce_open_limit=training_cfm,
            )
        records.append(
            DecisionRecord(
                task_features=observation.task_features.clone(),
                agent_features=observation.agent_features.clone(),
                action_mask=observation.action_mask.clone(),
                agent_index=observation.agent_index,
                action=action,
                role=role,
                robot_id=robot_id,
                observation_time=env.time,
                origin_time=origins.get(agent, env.time),
                state_version=(
                    context.snapshot.state_version if context is not None else None
                ),
                token_sequence=(context.token.sequence if context is not None else None),
                task_observation_schema=observation.task_observation_schema,
            )
        )
        return True

    try:
        while not env.terminal:
            event_steps += 1
            if event_steps > 4 * len(tasks) + n_mbr + n_dor + 4:
                failure_reason = "event_limit"
                break

            newly_blocked: List[AgentKey] = []
            for agent in ready:
                origins.setdefault(agent, env.time)
                if not try_agent(agent, allow_defer=training_cfm) and training_cfm:
                    newly_blocked.append(agent)
            ready = []

            if training_cfm:
                candidates = deferred + newly_blocked
                deferred = []
                while candidates:
                    still_blocked: List[AgentKey] = []
                    progress = False
                    seen = set()
                    for agent in candidates:
                        if agent in seen:
                            continue
                        seen.add(agent)
                        if not env.is_available(*agent):
                            continue
                        if try_agent(agent, allow_defer=True):
                            progress = True
                        else:
                            still_blocked.append(agent)
                    if not progress:
                        deferred = still_blocked
                        break
                    candidates = still_blocked

            if len(records) > max_decisions:
                failure_reason = "decision_limit"
                break
            if env.done:
                break
            if not env.pending_events:
                failure_reason = "deadlock"
                break

            ready = list(
                protocol.advance() if protocol is not None else env.advance()
            )
            if env.timed_out:
                failure_reason = "timeout"
                break
            for agent in ready:
                origins[agent] = env.time

        if env.done:
            metrics = env.metrics()
            cost = metrics.objective
            success = True
            failure_reason = ""
        else:
            metrics = None
            failure_penalty = (
                deadlock_penalty(tasks)
                if objective_mode == "v4"
                else protocol_failure_penalty(len(tasks))
            )
            cost = env.partial_objective + failure_penalty
            success = False
            if not failure_reason:
                failure_reason = "timeout" if env.timed_out else "deadlock"
    finally:
        policy.train(was_training)

    flashforward_decisions = sum(record.used_flashforward for record in records)
    return EpisodeOutcome(
        cost=float(cost),
        reward=-float(cost),
        success=success,
        failure_reason=failure_reason,
        metrics=metrics,
        plan=tuple(env.plan),
        records=tuple(records),
        decision_time_s=decision_time,
        deferred_attempts=deferred_attempts,
        flashforward_decisions=flashforward_decisions,
        execution_metrics=(protocol.metrics if protocol is not None else None),
        protocol_rounds=(protocol.rounds if protocol is not None else ()),
    )


def collect_repeat_average_batch(
    policy: nn.Module,
    tasks: Sequence[TaskSpec],
    n_mbr: int,
    n_dor: int,
    speed: float,
    *,
    pomo_size: int = 10,
    seed: int = 0,
    max_open_tasks: Optional[int] = None,
    time_limit: Optional[float] = None,
    dock_time: float = 8.0,
    detach_time: float = 8.0,
    mbr_speed: Optional[float] = None,
    dor_speed: Optional[float] = None,
    mbr_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    dor_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    objective_mode: str = "v4",
) -> RepeatAverageBatch:
    if pomo_size < 2:
        raise ValueError("pomo_size must be at least two for a repeat-average baseline")
    outcomes = tuple(
        run_decentralized_episode(
            policy,
            tasks,
            n_mbr,
            n_dor,
            speed,
            sample=True,
            training_cfm=True,
            env_seed=seed + 104729 * repeat,
            action_seed=seed + 130363 * repeat,
            max_open_tasks=max_open_tasks,
            time_limit=time_limit,
            dock_time=dock_time,
            detach_time=detach_time,
            mbr_speed=mbr_speed,
            dor_speed=dor_speed,
            mbr_initial_positions=mbr_initial_positions,
            dor_initial_positions=dor_initial_positions,
            objective_mode=objective_mode,
        )
        for repeat in range(pomo_size)
    )
    mean_cost = sum(outcome.cost for outcome in outcomes) / pomo_size
    decisions = tuple(
        WeightedDecision(record, outcome.cost - mean_cost, seed)
        for outcome in outcomes
        for record in outcome.records
    )
    if not decisions:
        raise RuntimeError("repeat-average rollout produced no trainable decisions")
    return RepeatAverageBatch(outcomes, mean_cost, decisions)


def reinforce_loss(
    policy: nn.Module,
    decisions: Sequence[WeightedDecision],
    device: Optional[torch.device] = None,
) -> ReinforceLoss:
    """Recompute policy probabilities and return the action-level REINFORCE loss."""
    if not decisions:
        raise ValueError("at least one weighted decision is required")
    device = device or _policy_device(policy)
    task_shapes = {tuple(item.record.task_features.shape) for item in decisions}
    agent_shapes = {tuple(item.record.agent_features.shape) for item in decisions}
    if len(task_shapes) != 1 or len(agent_shapes) != 1:
        raise ValueError("a reinforce batch must use fixed task and fleet sizes")

    task_inputs = torch.stack([item.record.task_features for item in decisions]).to(device)
    agent_inputs = torch.stack([item.record.agent_features for item in decisions]).to(device)
    action_mask = torch.stack([item.record.action_mask for item in decisions]).to(device)
    agent_index = torch.tensor(
        [item.record.agent_index for item in decisions],
        dtype=torch.long,
        device=device,
    )
    actions = torch.tensor(
        [item.record.action for item in decisions],
        dtype=torch.long,
        device=device,
    )
    advantages = torch.tensor(
        [item.cost_advantage for item in decisions],
        dtype=task_inputs.dtype,
        device=device,
    )
    probabilities, log_probabilities = policy(
        task_inputs,
        agent_inputs,
        action_mask,
        agent_index,
    )
    selected_log_probabilities = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
    policy_loss = (selected_log_probabilities * advantages.detach()).mean()
    safe_log_probabilities = log_probabilities.masked_fill(action_mask, 0.0)
    entropy_terms = probabilities * safe_log_probabilities
    entropy = -entropy_terms.sum(dim=-1).mean()
    if not torch.isfinite(policy_loss) or not torch.isfinite(entropy):
        raise FloatingPointError("non-finite REINFORCE loss")
    return ReinforceLoss(policy_loss, entropy, len(decisions))
