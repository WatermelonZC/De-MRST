"""Discrete-event decentralized environment for one-MBR/one-DOR coalitions."""

from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import List, Optional, Sequence, Tuple

from .core import (
    DEFAULT_DOR_SPEED,
    DEFAULT_DETACH_TIME,
    DEFAULT_DOCK_TIME,
    DEFAULT_MBR_SPEED,
    PlanMetrics,
    RobotState,
    TaskExecution,
    TaskSpec,
    deadlock_penalty,
    distance_tardiness_objective,
    makespan_distance_objective,
    marsupial_transition,
    engineering_objective,
    metrics_from_executions,
    normalize_objective_mode,
    objective_value,
    protocol_failure_penalty,
    validate_role_speed_order,
)


class TaskStatus(IntEnum):
    EMPTY = 0
    OPEN = 1
    IN_PROGRESS = 2
    COMPLETED = 3


class Role(IntEnum):
    MBR = 0
    DOR = 1


@dataclass
class TaskState:
    status: TaskStatus = TaskStatus.EMPTY
    mbr_id: Optional[int] = None
    dor_id: Optional[int] = None


class DecentralizedMRSEnv:
    """Sequential decisions at each event time with immediate state updates."""

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        n_mbr: int,
        n_dor: int,
        speed: float,
        max_open_tasks: Optional[int] = None,
        seed: int = 0,
        time_limit: Optional[float] = None,
        dock_time: float = DEFAULT_DOCK_TIME,
        detach_time: float = DEFAULT_DETACH_TIME,
        *,
        mbr_speed: Optional[float] = DEFAULT_MBR_SPEED,
        dor_speed: Optional[float] = DEFAULT_DOR_SPEED,
        mbr_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
        dor_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
        objective_mode: str = "v4",
        task_release_times: Optional[Sequence[float]] = None,
        travel_time_factors: Optional[Sequence[float]] = None,
        operation_time_factors: Optional[Sequence[float]] = None,
        execution_time_factors: Optional[Sequence[float]] = None,
    ):
        if not tasks:
            raise ValueError("at least one task is required")
        if n_mbr <= 0 or n_dor <= 0:
            raise ValueError("both robot roles require at least one robot")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self.mbr_speed = float(DEFAULT_MBR_SPEED if mbr_speed is None else mbr_speed)
        self.dor_speed = float(DEFAULT_DOR_SPEED if dor_speed is None else dor_speed)
        validate_role_speed_order(self.mbr_speed, self.dor_speed)
        self.tasks = tuple(tasks)
        self.task_release_times = self._resolve_task_vector(
            task_release_times, len(self.tasks), 0.0, "task release times", allow_zero=True
        )
        self.travel_time_factors = self._resolve_task_vector(
            travel_time_factors, len(self.tasks), 1.0, "travel time factors"
        )
        self.operation_time_factors = self._resolve_task_vector(
            operation_time_factors, len(self.tasks), 1.0, "operation time factors"
        )
        self.execution_time_factors = self._resolve_task_vector(
            execution_time_factors, len(self.tasks), 1.0, "execution time factors"
        )
        self.n_mbr = n_mbr
        self.n_dor = n_dor
        self.mbr_initial_positions = self._resolve_initial_positions(
            mbr_initial_positions, n_mbr, "MBR"
        )
        self.dor_initial_positions = self._resolve_initial_positions(
            dor_initial_positions, n_dor, "DOR"
        )
        self.max_open_tasks = (
            max_open_tasks
            if max_open_tasks is not None
            else min(5, max(1, n_mbr + n_dor - 1))
        )
        if self.max_open_tasks <= 0:
            raise ValueError("max_open_tasks must be positive")
        if dock_time < 0 or detach_time < 0:
            raise ValueError("dock and detach times must be non-negative")
        self.time_limit = time_limit
        self.dock_time = float(dock_time)
        self.detach_time = float(detach_time)
        self.objective_mode = normalize_objective_mode(objective_mode)
        self.random = Random(seed)
        self.reset()

    @staticmethod
    def _resolve_task_vector(values, count, default, name, allow_zero=False):
        if values is None:
            return tuple(float(default) for _ in range(count))
        resolved = tuple(float(value) for value in values)
        if len(resolved) != count:
            raise ValueError(f"{name} must match task count")
        invalid = (
            any(value < 0 for value in resolved)
            if allow_zero
            else any(value <= 0 for value in resolved)
        )
        if invalid:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {relation}")
        return resolved

    @staticmethod
    def _resolve_initial_positions(
        positions: Optional[Sequence[Tuple[float, float]]],
        count: int,
        name: str,
    ) -> Tuple[Tuple[float, float], ...]:
        if positions is None:
            return tuple((0.0, 0.0) for _ in range(count))
        values = tuple((float(item[0]), float(item[1])) for item in positions)
        if len(values) != count:
            raise ValueError(f"{name} initial positions must match fleet size")
        return values

    def reset(self) -> None:
        self.state_version = 0
        self.time = 0.0
        self.task_state = [TaskState() for _ in self.tasks]
        self.task_released = [time <= 0.0 for time in self.task_release_times]
        self.mbr_state = [
            RobotState(position=self.mbr_initial_positions[robot_id], speed=self.mbr_speed)
            for robot_id in range(self.n_mbr)
        ]
        self.dor_state = [
            RobotState(position=self.dor_initial_positions[robot_id], speed=self.dor_speed)
            for robot_id in range(self.n_dor)
        ]
        self.mbr_busy = [False] * self.n_mbr
        self.dor_busy = [False] * self.n_dor
        self.mbr_waiting = [False] * self.n_mbr
        self.dor_waiting = [False] * self.n_dor
        self.mbr_current_task: List[Optional[int]] = [None] * self.n_mbr
        self.dor_current_task: List[Optional[int]] = [None] * self.n_dor
        self.mbr_travel_until = [0.0] * self.n_mbr
        self.dor_travel_until = [0.0] * self.n_dor
        self.pending_events: List[Tuple[float, int, int, int]] = [
            (time, 2, -1, task_id)
            for task_id, time in enumerate(self.task_release_times)
            if time > 0.0
        ]
        self.plan: List[Tuple[int, int, int]] = []
        self.executions: List[TaskExecution] = []
        self.timed_out = False

    def initial_decisions(self) -> Tuple[Tuple[Role, int], ...]:
        decisions = [
            *((Role.MBR, robot_id) for robot_id in range(self.n_mbr)),
            *((Role.DOR, robot_id) for robot_id in range(self.n_dor)),
        ]
        self.random.shuffle(decisions)
        return tuple(decisions)

    def role_mask(
        self,
        role: Role,
        robot_id: int,
        enforce_open_limit: bool = True,
    ) -> Tuple[bool, ...]:
        self._validate_robot(role, robot_id)
        if role == Role.MBR and (self.mbr_busy[robot_id] or self.mbr_waiting[robot_id]):
            return tuple(False for _ in self.tasks)
        if role == Role.DOR and (self.dor_busy[robot_id] or self.dor_waiting[robot_id]):
            return tuple(False for _ in self.tasks)
        open_count = sum(state.status == TaskStatus.OPEN for state in self.task_state)
        legal = []
        for task_id, state in enumerate(self.task_state):
            if not self.task_released[task_id]:
                legal.append(False)
                continue
            if state.status == TaskStatus.EMPTY:
                legal.append(not enforce_open_limit or open_count < self.max_open_tasks)
            elif state.status == TaskStatus.OPEN:
                legal.append((role == Role.MBR and state.mbr_id is None) or (role == Role.DOR and state.dor_id is None))
            else:
                legal.append(False)
        return tuple(legal)

    def assign(
        self,
        role: Role,
        robot_id: int,
        task_id: int,
        enforce_open_limit: bool = True,
    ) -> None:
        self.commit_assignments(
            ((role, robot_id, task_id),),
            enforce_open_limit=enforce_open_limit,
        )

    def commit_assignments(
        self,
        assignments: Sequence[Tuple[Role, int, int]],
        enforce_open_limit: bool = True,
        expected_state_version: Optional[int] = None,
    ) -> None:
        """Atomically validate and commit one decision-round assignment batch."""
        if (
            expected_state_version is not None
            and expected_state_version != self.state_version
        ):
            raise RuntimeError(
                "stale environment state version: "
                f"expected {expected_state_version}, current {self.state_version}"
            )

        claims = tuple(assignments)
        if not claims:
            return

        shadow = deepcopy(self)
        for role, robot_id, task_id in claims:
            shadow._apply_assignment(
                role,
                robot_id,
                task_id,
                enforce_open_limit=enforce_open_limit,
            )

        for role, robot_id, task_id in claims:
            self._apply_assignment(
                role,
                robot_id,
                task_id,
                enforce_open_limit=enforce_open_limit,
            )
        self.state_version += 1

    def _apply_assignment(
        self,
        role: Role,
        robot_id: int,
        task_id: int,
        enforce_open_limit: bool,
    ) -> None:
        mask = self.role_mask(role, robot_id, enforce_open_limit=enforce_open_limit)
        if not 0 <= task_id < len(mask) or not mask[task_id]:
            raise ValueError("illegal decentralized action")
        state = self.task_state[task_id]
        state.status = TaskStatus.OPEN
        if role == Role.MBR:
            state.mbr_id = robot_id
            self.mbr_waiting[robot_id] = True
            self.mbr_current_task[robot_id] = task_id
        else:
            state.dor_id = robot_id
            self.dor_waiting[robot_id] = True
            self.dor_current_task[robot_id] = task_id
        if state.mbr_id is not None and state.dor_id is not None:
            self._start_task(task_id, state.mbr_id, state.dor_id)

    def _start_task(self, task_id: int, mbr_id: int, dor_id: int) -> None:
        # Preserve externally edited position/time probes while enforcing the
        # role-level physical speeds of this environment.
        mbr_state = self.mbr_state[mbr_id]
        dor_state = self.dor_state[dor_id]
        mbr_before = RobotState(
            mbr_state.position, mbr_state.available_time, self.mbr_speed
        )
        dor_before = RobotState(
            dor_state.position, dor_state.available_time, self.dor_speed
        )
        result = marsupial_transition(
            mbr_before,
            dor_before,
            self.tasks[task_id],
            None,
            self.dock_time,
            self.detach_time,
            travel_time_factor=self.travel_time_factors[task_id],
            operation_time_factor=self.operation_time_factors[task_id],
            execution_time_factor=self.execution_time_factors[task_id],
        )
        destination = self.tasks[task_id].destination
        self.mbr_state[mbr_id] = RobotState(
            destination, result.mbr_release_time, self.mbr_speed
        )
        self.dor_state[dor_id] = RobotState(
            destination, result.dor_release_time, self.dor_speed
        )
        self.mbr_travel_until[mbr_id] = result.destination_time
        self.dor_travel_until[dor_id] = result.destination_time
        self.mbr_waiting[mbr_id] = self.dor_waiting[dor_id] = False
        self.mbr_busy[mbr_id] = self.dor_busy[dor_id] = True
        self.task_state[task_id].status = TaskStatus.IN_PROGRESS
        self.pending_events.extend([(result.mbr_release_time, 0, mbr_id, task_id), (result.dor_release_time, 1, dor_id, task_id)])
        self.plan.append((task_id, mbr_id, dor_id))
        self.executions.append(TaskExecution(task_id, mbr_id, dor_id, mbr_before, dor_before, result))

    def advance(self) -> Tuple[Tuple[Role, int], ...]:
        if not self.pending_events:
            return tuple()
        next_time = min(event[0] for event in self.pending_events)
        if self.time_limit is not None and next_time > self.time_limit:
            self.time = self.time_limit
            self.pending_events = []
            self.timed_out = True
            self.state_version += 1
            return tuple()
        self.time = next_time
        ready, future = [], []
        for event in self.pending_events:
            (ready if abs(event[0] - next_time) <= 1e-12 else future).append(event)
        self.pending_events = future
        decisions = []
        released_task = False
        for _, role_value, robot_id, task_id in ready:
            if role_value == 0:
                self.mbr_busy[robot_id] = False
                self.mbr_current_task[robot_id] = None
                self.mbr_travel_until[robot_id] = 0.0
                decisions.append((Role.MBR, robot_id))
            elif role_value == 1:
                self.dor_busy[robot_id] = False
                self.dor_current_task[robot_id] = None
                self.dor_travel_until[robot_id] = 0.0
                self.task_state[task_id].status = TaskStatus.COMPLETED
                decisions.append((Role.DOR, robot_id))
            else:
                self.task_released[task_id] = True
                released_task = True
        if released_task:
            decisions.extend(
                (role, robot_id)
                for role, count in ((Role.MBR, self.n_mbr), (Role.DOR, self.n_dor))
                for robot_id in range(count)
                if self.is_available(role, robot_id)
                and (role, robot_id) not in decisions
            )
        self.random.shuffle(decisions)
        self.state_version += 1
        return tuple(decisions)

    def metrics(self) -> PlanMetrics:
        if not self.done:
            raise RuntimeError("episode is not complete")
        return metrics_from_executions(
            self.tasks,
            self.executions,
            self.speed,
            self.mbr_speed,
            self.objective_mode,
        )

    def global_agent_index(self, role: Role, robot_id: int) -> int:
        self._validate_robot(role, robot_id)
        return robot_id if role == Role.MBR else self.n_mbr + robot_id

    def agent_from_global_index(self, index: int) -> Tuple[Role, int]:
        if not 0 <= index < self.n_mbr + self.n_dor:
            raise IndexError("invalid global agent index")
        if index < self.n_mbr:
            return Role.MBR, index
        return Role.DOR, index - self.n_mbr

    def is_available(self, role: Role, robot_id: int) -> bool:
        self._validate_robot(role, robot_id)
        if role == Role.MBR:
            return not self.mbr_busy[robot_id] and not self.mbr_waiting[robot_id]
        return not self.dor_busy[robot_id] and not self.dor_waiting[robot_id]

    def robot_speed(self, role: Role, robot_id: int = 0) -> float:
        """Return the role-level speed used by every robot of that role."""
        self._validate_robot(role, robot_id)
        return self.mbr_speed if role == Role.MBR else self.dor_speed

    def _validate_robot(self, role: Role, robot_id: int) -> None:
        count = self.n_mbr if role == Role.MBR else self.n_dor
        if not 0 <= robot_id < count:
            raise IndexError(f"invalid {role.name} robot id {robot_id}")

    @property
    def partial_objective(self) -> float:
        if self.objective_mode == "makespan":
            return float(self.time)
        total_tardiness = sum(
            max(0.0, execution.result.task_completion_time - self.tasks[execution.task_id].due_time)
            for execution in self.executions
        )
        system_distance = sum(execution.result.system_distance for execution in self.executions)
        if self.objective_mode == "engineering":
            return engineering_objective(
                float(self.time),
                float(system_distance),
                self.mbr_speed,
            )
        if self.objective_mode == "distance_tardiness":
            return distance_tardiness_objective(
                float(total_tardiness),
                float(system_distance),
                self.mbr_speed,
            )
        if self.objective_mode == "makespan_distance":
            return makespan_distance_objective(
                float(self.time),
                float(system_distance),
                self.mbr_speed,
            )
        return objective_value(total_tardiness, system_distance, len(self.tasks), self.speed)

    @property
    def episode_cost(self) -> float:
        return self.partial_objective + self.terminal_penalty

    @property
    def deadlocked(self) -> bool:
        if self.done or self.timed_out or self.pending_events:
            return False
        for robot_id in range(self.n_mbr):
            if any(self.role_mask(Role.MBR, robot_id, enforce_open_limit=False)):
                return False
        for robot_id in range(self.n_dor):
            if any(self.role_mask(Role.DOR, robot_id, enforce_open_limit=False)):
                return False
        return True

    @property
    def failed(self) -> bool:
        return self.deadlocked or self.timed_out

    @property
    def terminal(self) -> bool:
        return self.done or self.failed

    @property
    def terminal_penalty(self) -> float:
        if not self.failed:
            return 0.0
        if self.objective_mode == "v4":
            return deadlock_penalty(self.tasks)
        return protocol_failure_penalty(len(self.tasks))

    @property
    def done(self) -> bool:
        return all(state.status == TaskStatus.COMPLETED for state in self.task_state)
