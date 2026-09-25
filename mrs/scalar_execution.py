"""Shared scalar execution protocol for decentralized evaluation.

The physical world remains :class:`DecentralizedMRSEnv`.  This module adds the
deployment-facing semantics around it: versioned snapshots, a movable decision
token, conflict-free claim commits, and event advancement.  Decision strategies
remain method-specific and never share a winner-selection rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .core import RobotState, TaskExecution, TaskSpec
from .event_env import DecentralizedMRSEnv, Role


AgentKey = Tuple[Role, int]
SCALAR_EXECUTION_CONTRACT_VERSION = "scalar_token_atomic_v1"


@dataclass(frozen=True)
class TaskStateSnapshot:
    status: int
    mbr_id: Optional[int]
    dor_id: Optional[int]


@dataclass(frozen=True)
class AgentStateSnapshot:
    role: Role
    robot_id: int
    state: RobotState
    busy: bool
    waiting: bool
    current_task: Optional[int]
    travel_until: float


@dataclass(frozen=True)
class MRSStateSnapshot:
    state_version: int
    time: float
    tasks: Tuple[TaskSpec, ...]
    task_states: Tuple[TaskStateSnapshot, ...]
    task_released: Tuple[bool, ...]
    task_release_times: Tuple[float, ...]
    travel_time_factors: Tuple[float, ...]
    operation_time_factors: Tuple[float, ...]
    execution_time_factors: Tuple[float, ...]
    agent_states: Tuple[AgentStateSnapshot, ...]
    pending_events: Tuple[Tuple[float, int, int, int], ...]
    plan: Tuple[Tuple[int, int, int], ...]
    executions: Tuple[TaskExecution, ...]
    n_mbr: int
    n_dor: int
    max_open_tasks: int
    speed: float
    mbr_speed: float
    dor_speed: float
    mbr_initial_positions: Tuple[Tuple[float, float], ...]
    dor_initial_positions: Tuple[Tuple[float, float], ...]
    dock_time: float
    detach_time: float
    time_limit: Optional[float]
    objective_mode: str
    timed_out: bool
    random_state: tuple
    digest: str


@dataclass(frozen=True)
class DecisionToken:
    sequence: int
    state_version: int
    holder: AgentKey


@dataclass(frozen=True)
class DecisionContext:
    token: DecisionToken
    snapshot: MRSStateSnapshot
    participants: Tuple[AgentKey, ...]


@dataclass(frozen=True)
class AssignmentClaim:
    agent: AgentKey
    task_id: int


@dataclass(frozen=True)
class CommitReceipt:
    commit_id: int
    previous_version: int
    current_version: int
    claims: Tuple[AssignmentClaim, ...]


@dataclass(frozen=True)
class ProtocolRound:
    token: DecisionToken
    snapshot_digest: str
    participants: Tuple[AgentKey, ...]
    claims: Tuple[AssignmentClaim, ...]
    receipt: Optional[CommitReceipt]


@dataclass(frozen=True)
class ScalarExecutionMetrics:
    round_count: int
    claim_count: int
    commit_count: int
    message_count: int
    strategy_compute_time_s: float
    coordination_time_s: float
    commit_time_s: float
    simulator_advance_time_s: float


def _agent_snapshots(env: DecentralizedMRSEnv) -> Tuple[AgentStateSnapshot, ...]:
    snapshots = []
    for robot_id in range(env.n_mbr):
        snapshots.append(AgentStateSnapshot(
            Role.MBR,
            robot_id,
            env.mbr_state[robot_id],
            bool(env.mbr_busy[robot_id]),
            bool(env.mbr_waiting[robot_id]),
            env.mbr_current_task[robot_id],
            float(env.mbr_travel_until[robot_id]),
        ))
    for robot_id in range(env.n_dor):
        snapshots.append(AgentStateSnapshot(
            Role.DOR,
            robot_id,
            env.dor_state[robot_id],
            bool(env.dor_busy[robot_id]),
            bool(env.dor_waiting[robot_id]),
            env.dor_current_task[robot_id],
            float(env.dor_travel_until[robot_id]),
        ))
    return tuple(snapshots)


def capture_state(env: DecentralizedMRSEnv) -> MRSStateSnapshot:
    """Capture an immutable, hash-addressed copy of the current world state."""
    task_states = tuple(
        TaskStateSnapshot(int(state.status), state.mbr_id, state.dor_id)
        for state in env.task_state
    )
    task_released = tuple(bool(value) for value in env.task_released)
    agent_states = _agent_snapshots(env)
    pending_events = tuple(env.pending_events)
    plan = tuple(env.plan)
    executions = tuple(env.executions)
    random_state = env.random.getstate()
    payload = (
        int(env.state_version),
        float(env.time),
        tuple(env.tasks),
        task_states,
        task_released,
        env.task_release_times,
        env.travel_time_factors,
        env.operation_time_factors,
        env.execution_time_factors,
        agent_states,
        pending_events,
        plan,
        executions,
        int(env.n_mbr),
        int(env.n_dor),
        int(env.max_open_tasks),
        float(env.speed),
        float(env.mbr_speed),
        float(env.dor_speed),
        tuple(env.mbr_initial_positions),
        tuple(env.dor_initial_positions),
        float(env.dock_time),
        float(env.detach_time),
        None if env.time_limit is None else float(env.time_limit),
        str(env.objective_mode),
        bool(env.timed_out),
        random_state,
    )
    digest = sha256(repr(payload).encode("ascii")).hexdigest()
    return MRSStateSnapshot(
        state_version=int(env.state_version),
        time=float(env.time),
        tasks=tuple(env.tasks),
        task_states=task_states,
        task_released=task_released,
        task_release_times=env.task_release_times,
        travel_time_factors=env.travel_time_factors,
        operation_time_factors=env.operation_time_factors,
        execution_time_factors=env.execution_time_factors,
        agent_states=agent_states,
        pending_events=pending_events,
        plan=plan,
        executions=executions,
        n_mbr=int(env.n_mbr),
        n_dor=int(env.n_dor),
        max_open_tasks=int(env.max_open_tasks),
        speed=float(env.speed),
        mbr_speed=float(env.mbr_speed),
        dor_speed=float(env.dor_speed),
        mbr_initial_positions=tuple(env.mbr_initial_positions),
        dor_initial_positions=tuple(env.dor_initial_positions),
        dock_time=float(env.dock_time),
        detach_time=float(env.detach_time),
        time_limit=None if env.time_limit is None else float(env.time_limit),
        objective_mode=str(env.objective_mode),
        timed_out=bool(env.timed_out),
        random_state=random_state,
        digest=digest,
    )


class ScalarExecutionProtocol:
    """Versioned token and commit layer around one scalar environment."""

    def __init__(self, env: DecentralizedMRSEnv):
        self.env = env
        self._token_sequence = 0
        self._commit_id = 0
        self._rounds = []
        self._claim_count = 0
        self._message_count = 0
        self._strategy_compute_time_s = 0.0
        self._coordination_time_s = 0.0
        self._commit_time_s = 0.0
        self._simulator_advance_time_s = 0.0

    @property
    def state_version(self) -> int:
        return int(self.env.state_version)

    @property
    def rounds(self) -> Tuple[ProtocolRound, ...]:
        return tuple(self._rounds)

    @property
    def metrics(self) -> ScalarExecutionMetrics:
        return ScalarExecutionMetrics(
            round_count=len(self._rounds),
            claim_count=self._claim_count,
            commit_count=self._commit_id,
            message_count=self._message_count,
            strategy_compute_time_s=self._strategy_compute_time_s,
            coordination_time_s=self._coordination_time_s,
            commit_time_s=self._commit_time_s,
            simulator_advance_time_s=self._simulator_advance_time_s,
        )

    def initial_decisions(self) -> Tuple[AgentKey, ...]:
        return tuple(self.env.initial_decisions())

    def begin_round(
        self,
        holder: AgentKey,
        participants: Sequence[AgentKey],
    ) -> DecisionContext:
        participants = tuple(participants)
        if not participants:
            raise ValueError("a decision round requires at least one participant")
        if holder not in participants:
            raise ValueError("token holder must be a round participant")
        started = perf_counter()
        snapshot = capture_state(self.env)
        token = DecisionToken(
            sequence=self._token_sequence,
            state_version=snapshot.state_version,
            holder=holder,
        )
        self._token_sequence += 1
        self._coordination_time_s += perf_counter() - started
        # One logical bundle transfers the token and its versioned state view.
        self._message_count += 1
        return DecisionContext(token, snapshot, participants)

    def _verify_context(self, context: DecisionContext) -> None:
        current = capture_state(self.env)
        if current.state_version != context.snapshot.state_version:
            raise RuntimeError("stale decision context")
        if current.digest != context.snapshot.digest:
            raise RuntimeError("environment changed outside the execution protocol")

    def record_noop(self, context: DecisionContext) -> None:
        started = perf_counter()
        self._verify_context(context)
        self._coordination_time_s += perf_counter() - started
        self._rounds.append(ProtocolRound(
            context.token,
            context.snapshot.digest,
            context.participants,
            tuple(),
            None,
        ))

    def commit(
        self,
        context: DecisionContext,
        claims: Sequence[AssignmentClaim],
        *,
        enforce_open_limit: bool,
    ) -> CommitReceipt:
        claims = tuple(claims)
        if not claims:
            raise ValueError("cannot commit an empty claim batch")
        started = perf_counter()
        self._verify_context(context)
        previous_version = self.state_version
        self.env.commit_assignments(
            tuple((claim.agent[0], claim.agent[1], claim.task_id) for claim in claims),
            enforce_open_limit=enforce_open_limit,
            expected_state_version=previous_version,
        )
        self._commit_time_s += perf_counter() - started
        self._commit_id += 1
        self._claim_count += len(claims)
        # The receipt/commit result is one logical broadcast bundle.
        self._message_count += 1
        receipt = CommitReceipt(
            self._commit_id,
            previous_version,
            self.state_version,
            claims,
        )
        self._rounds.append(ProtocolRound(
            context.token,
            context.snapshot.digest,
            context.participants,
            claims,
            receipt,
        ))
        return receipt

    def advance(self) -> Tuple[AgentKey, ...]:
        started = perf_counter()
        ready = tuple(self.env.advance())
        self._simulator_advance_time_s += perf_counter() - started
        return ready

    def add_strategy_time(self, elapsed_s: float) -> None:
        self._strategy_compute_time_s += max(0.0, float(elapsed_s))

    def add_coordination_time(self, elapsed_s: float) -> None:
        self._coordination_time_s += max(0.0, float(elapsed_s))

    def add_messages(self, count: int) -> None:
        if count < 0:
            raise ValueError("message count cannot be negative")
        self._message_count += int(count)
