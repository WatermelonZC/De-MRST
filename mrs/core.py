"""Single source of truth for Marsupial execution timing and cost.

The functions in this module are deliberately policy-agnostic. Centralized
solvers and decentralized environments must all use these primitives so that
execution semantics cannot silently diverge.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


Position = Tuple[float, float]
DEFAULT_DOCK_TIME = 8.0
DEFAULT_DETACH_TIME = 8.0
DEFAULT_ROBOT_SPEED = 1.8
DEFAULT_MBR_SPEED = 1.2
DEFAULT_DOR_SPEED = 2.4
OBJECTIVE_TARDINESS_WEIGHT = 0.95
OBJECTIVE_DISTANCE_WEIGHT = 0.05
OBJECTIVE_MAKESPAN_WEIGHT = 0.9
OBJECTIVE_TRAVEL_TIME_WEIGHT = 0.1
OBJECTIVE_DISTANCE_TIME_WEIGHT = 0.1
OBJECTIVE_TARDINESS_TIME_WEIGHT = 0.9
OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT = 0.7
OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT = 0.3
OBJECTIVE_MODES = (
    "v4",
    "makespan",
    "engineering",
    "distance_tardiness",
    "makespan_distance",
)


def manhattan_distance(a: Position, b: Position) -> float:
    return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))


@dataclass(frozen=True)
class TaskSpec:
    source: Position
    destination: Position
    due_time: float
    pickup_time: float
    handling_time: float


@dataclass(frozen=True)
class RobotState:
    position: Position = (0.0, 0.0)
    available_time: float = 0.0
    speed: float = DEFAULT_ROBOT_SPEED


@dataclass(frozen=True)
class TransitionResult:
    dock_time: float
    destination_time: float
    mbr_release_time: float
    dor_release_time: float
    task_completion_time: float
    system_distance: float
    mbr_distance: float
    dor_distance: float
    mbr_wait: float
    dor_wait: float


@dataclass(frozen=True)
class TaskExecution:
    task_id: int
    mbr_id: int
    dor_id: int
    mbr_before: RobotState
    dor_before: RobotState
    result: TransitionResult


@dataclass(frozen=True)
class PlanMetrics:
    objective: float
    total_tardiness: float
    average_tardiness: float
    on_time_rate: float
    system_distance: float
    mbr_distance: float
    dor_distance: float
    average_mbr_wait: float
    average_dor_wait: float
    makespan: float
    success: bool
    completion_times: Tuple[float, ...]
    executions: Tuple[TaskExecution, ...]


def validate_role_speed_order(
    mbr_speed: float,
    dor_speed: float,
    *,
    require_strict: bool = False,
) -> None:
    """Validate the mother/child speed convention.

    Legacy scalar callers may still use one shared speed for both roles, so
    equality is accepted by default. Generated experiment instances request
    the strict physical convention that the child DOR is faster.
    """
    if mbr_speed <= 0 or dor_speed <= 0:
        raise ValueError("robot speeds must be positive")
    if dor_speed < mbr_speed or (require_strict and dor_speed <= mbr_speed):
        relation = "greater than" if require_strict else "at least"
        raise ValueError(
            f"DOR (child) speed must be {relation} MBR (mother) speed; "
            f"got mbr_speed={mbr_speed}, dor_speed={dor_speed}"
        )


def normalize_objective_mode(mode: str) -> str:
    """Validate and normalize the shared objective-mode name."""
    mode = str(mode).lower()
    if mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"unsupported objective mode: {mode}; expected one of {OBJECTIVE_MODES}"
        )
    return mode


def marsupial_transition(
    mbr: RobotState,
    dor: RobotState,
    task: TaskSpec,
    speed: float = None,
    dock_time: float = DEFAULT_DOCK_TIME,
    detach_time: float = DEFAULT_DETACH_TIME,
    *,
    travel_time_factor: float = 1.0,
    operation_time_factor: float = 1.0,
    execution_time_factor: float = 1.0,
) -> TransitionResult:
    """Evaluate one MBR-DOR execution without mutating either robot."""
    if travel_time_factor <= 0 or operation_time_factor <= 0 or execution_time_factor <= 0:
        raise ValueError("uncertainty factors must be positive")
    if speed is None:
        mbr_speed = float(mbr.speed)
        dor_speed = float(dor.speed)
    else:
        mbr_speed = dor_speed = float(speed)
    validate_role_speed_order(mbr_speed, dor_speed)
    # After rendezvous the MBR is the carrier, so the coupled transport uses
    # the MBR speed.  In the frozen v4 convention the DOR stays at its current
    # rendezvous location; its role speed is retained in state/observations but
    # does not create an independent pre-docking motion segment.
    combined_speed = mbr_speed

    rendezvous_distance = manhattan_distance(mbr.position, dor.position)
    mbr_arrival = (
        mbr.available_time
        + execution_time_factor * travel_time_factor * rendezvous_distance / mbr_speed
    )
    synchronized = max(dor.available_time, mbr_arrival)
    dock_finish = synchronized + execution_time_factor * dock_time

    rendezvous_to_source = manhattan_distance(dor.position, task.source)
    source_to_destination = manhattan_distance(task.source, task.destination)
    destination_arrival = (
        dock_finish
        + execution_time_factor * travel_time_factor * rendezvous_to_source / combined_speed
        + execution_time_factor * task.pickup_time
        + execution_time_factor * travel_time_factor * source_to_destination / combined_speed
    )
    mbr_release = destination_arrival + execution_time_factor * detach_time
    dor_release = mbr_release + execution_time_factor * operation_time_factor * task.handling_time

    return TransitionResult(
        dock_time=dock_finish,
        destination_time=destination_arrival,
        mbr_release_time=mbr_release,
        dor_release_time=dor_release,
        task_completion_time=dor_release,
        system_distance=rendezvous_distance + rendezvous_to_source + source_to_destination,
        mbr_distance=rendezvous_distance + rendezvous_to_source + source_to_destination,
        dor_distance=rendezvous_to_source + source_to_destination,
        mbr_wait=max(0.0, dor.available_time - mbr_arrival),
        dor_wait=max(0.0, mbr_arrival - dor.available_time),
    )


def objective_value(total_tardiness: float, system_distance: float, n_tasks: int, speed: float) -> float:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if speed <= 0:
        raise ValueError("speed must be positive")
    return (
        OBJECTIVE_TARDINESS_WEIGHT * total_tardiness / n_tasks
        + OBJECTIVE_DISTANCE_WEIGHT * system_distance / (n_tasks * speed)
    )


def engineering_objective(
    makespan: float,
    system_distance: float,
    mbr_speed: float,
) -> float:
    """Return the engineering cost in seconds.

    ``system_distance`` is the Marsupial carrier path.  In the current
    execution protocol the MBR carries the DOR after docking, so converting
    the path to vehicle work uses the physical MBR speed rather than the
    scalar reporting speed.
    """
    if makespan < 0 or system_distance < 0 or mbr_speed <= 0:
        raise ValueError("engineering objective inputs must be non-negative")
    return (
        OBJECTIVE_MAKESPAN_WEIGHT * float(makespan)
        + OBJECTIVE_TRAVEL_TIME_WEIGHT * float(system_distance) / float(mbr_speed)
    )


def distance_tardiness_objective(
    total_tardiness: float,
    system_distance: float,
    mbr_speed: float,
) -> float:
    """Return a physical-time objective combining travel and tardiness.

    Distance is converted to seconds with the carrier speed before applying
    the 0.1 travel / 0.9 tardiness weights.  Both terms are totals, so this mode is
    intentionally distinct from the per-task normalized frozen v4 objective.
    """
    if total_tardiness < 0 or system_distance < 0 or mbr_speed <= 0:
        raise ValueError("distance-tardiness inputs must be non-negative")
    return (
        OBJECTIVE_DISTANCE_TIME_WEIGHT * float(system_distance) / float(mbr_speed)
        + OBJECTIVE_TARDINESS_TIME_WEIGHT * float(total_tardiness)
    )


def makespan_distance_objective(
    makespan: float,
    system_distance: float,
    mbr_speed: float,
) -> float:
    """Return the time-consistent makespan/carrier-distance objective.

    ``system_distance`` is the carrier path after the MBR-DOR rendezvous.
    Coordinates are interpreted as metres and the path is converted to seconds
    using the physical MBR speed before applying the frozen 0.7/0.3 weights.
    """
    if makespan < 0 or system_distance < 0 or mbr_speed <= 0:
        raise ValueError("makespan-distance inputs must be non-negative")
    return (
        OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT * float(makespan)
        + OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT
        * float(system_distance)
        / float(mbr_speed)
    )


def objective_value_for_mode(
    mode: str,
    total_tardiness: float,
    system_distance: float,
    n_tasks: int,
    speed: float,
    *,
    makespan: float = None,
    mbr_speed: float = None,
) -> float:
    """Evaluate one of the shared scalar objective modes."""
    mode = normalize_objective_mode(mode)
    if mode == "v4":
        return objective_value(total_tardiness, system_distance, n_tasks, speed)
    if mode == "distance_tardiness":
        return distance_tardiness_objective(
            total_tardiness,
            system_distance,
            speed if mbr_speed is None else mbr_speed,
        )
    if mode == "makespan_distance":
        if makespan is None:
            raise ValueError("makespan_distance objective requires makespan")
        return makespan_distance_objective(
            float(makespan),
            float(system_distance),
            float(speed if mbr_speed is None else mbr_speed),
        )
    if makespan is None:
        raise ValueError(f"{mode} objective requires makespan")
    if mode == "makespan":
        return float(makespan)
    return engineering_objective(
        float(makespan),
        float(system_distance),
        float(speed if mbr_speed is None else mbr_speed),
    )


def engineering_objective_tensor(makespan, system_distance, mbr_speed):
    """Torch-compatible engineering objective in physical time units."""
    return (
        OBJECTIVE_MAKESPAN_WEIGHT * makespan
        + OBJECTIVE_TRAVEL_TIME_WEIGHT * system_distance / mbr_speed
    )


def objective_tensor(
    total_tardiness,
    system_distance,
    n_tasks: int,
    speed,
    *,
    objective_mode: str = "v4",
    makespan=None,
    mbr_speed=None,
):
    """Torch-compatible form of the selected objective mode."""
    objective_mode = normalize_objective_mode(objective_mode)
    if objective_mode == "v4":
        if n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        speed = speed.reshape(-1)
        return (
            OBJECTIVE_TARDINESS_WEIGHT * total_tardiness / n_tasks
            + OBJECTIVE_DISTANCE_WEIGHT * system_distance / (n_tasks * speed)
        )
    if objective_mode == "distance_tardiness":
        if mbr_speed is None:
            mbr_speed = speed
        if not hasattr(mbr_speed, "reshape"):
            mbr_speed = system_distance.new_tensor(float(mbr_speed))
        else:
            mbr_speed = mbr_speed.reshape(-1)
        return (
            OBJECTIVE_DISTANCE_TIME_WEIGHT * system_distance / mbr_speed
            + OBJECTIVE_TARDINESS_TIME_WEIGHT * total_tardiness
        )
    if objective_mode == "makespan_distance":
        if makespan is None:
            raise ValueError("makespan_distance objective requires makespan")
        if mbr_speed is None:
            mbr_speed = speed
        if not hasattr(mbr_speed, "reshape"):
            mbr_speed = system_distance.new_tensor(float(mbr_speed))
        else:
            mbr_speed = mbr_speed.reshape(-1)
        return (
            OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT * makespan
            + OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT
            * system_distance
            / mbr_speed
        )
    if makespan is None:
        raise ValueError(f"{objective_mode} objective requires makespan")
    if objective_mode == "makespan":
        return makespan
    if mbr_speed is None:
        mbr_speed = speed
    if not hasattr(mbr_speed, "reshape"):
        mbr_speed = system_distance.new_tensor(float(mbr_speed))
    else:
        mbr_speed = mbr_speed.reshape(-1)
    return engineering_objective_tensor(makespan, system_distance, mbr_speed)


def objective_tensor_v4(total_tardiness, system_distance, n_tasks: int, speed):
    """Backward-compatible explicit v4 tensor objective."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    speed = speed.reshape(-1)
    return (
        OBJECTIVE_TARDINESS_WEIGHT * total_tardiness / n_tasks
        + OBJECTIVE_DISTANCE_WEIGHT * system_distance / (n_tasks * speed)
    )


def deadlock_penalty(tasks: Sequence[TaskSpec]) -> float:
    if not tasks:
        raise ValueError("at least one task is required")
    return 10.0 * max(task.due_time for task in tasks)


def protocol_episode_timeout(task_count: int) -> float:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return max(1_000.0, 500.0 * task_count)


def protocol_failure_penalty(task_count: int) -> float:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return 100.0 * task_count


def evaluate_centralized_plan(
    tasks: Sequence[TaskSpec],
    assignments: Iterable[Tuple[int, int, int]],
    n_mbr: int,
    n_dor: int,
    speed: float,
    dock_time: float = DEFAULT_DOCK_TIME,
    detach_time: float = DEFAULT_DETACH_TIME,
    *,
    mbr_speed: Optional[float] = None,
    dor_speed: Optional[float] = None,
    mbr_speeds: Sequence[float] = None,
    dor_speeds: Sequence[float] = None,
    mbr_initial_positions: Sequence[Position] = None,
    dor_initial_positions: Sequence[Position] = None,
    objective_mode: str = "v4",
) -> PlanMetrics:
    """Evaluate an explicit global order of ``(task, MBR, DOR)`` triples."""
    if not tasks:
        raise ValueError("at least one task is required")
    if speed <= 0:
        raise ValueError("speed must be positive")
    objective_mode = normalize_objective_mode(objective_mode)
    def resolve_role_speed(
        role_speed: Optional[float],
        profile: Optional[Sequence[float]],
        count: int,
        name: str,
    ) -> float:
        if role_speed is not None and profile is not None:
            raise ValueError(f"provide {name}_speed or {name}_speeds, not both")
        if profile is not None:
            values = [float(value) for value in profile]
            if len(values) != count:
                raise ValueError(f"{name} speed profile must match fleet size")
            if not values or any(value <= 0 for value in values):
                raise ValueError("robot speeds must be positive")
            if any(abs(value - values[0]) > 1e-12 for value in values[1:]):
                raise ValueError(
                    f"all {name.upper()} robots must share one speed; "
                    "use a scalar role speed"
                )
            return values[0]
        value = float(
            (DEFAULT_MBR_SPEED if name == "mbr" else DEFAULT_DOR_SPEED)
            if role_speed is None
            else role_speed
        )
        if value <= 0:
            raise ValueError("robot speeds must be positive")
        return value

    resolved_mbr_speed = resolve_role_speed(mbr_speed, mbr_speeds, n_mbr, "mbr")
    resolved_dor_speed = resolve_role_speed(dor_speed, dor_speeds, n_dor, "dor")
    validate_role_speed_order(resolved_mbr_speed, resolved_dor_speed)

    def resolve_positions(
        positions: Optional[Sequence[Position]], count: int, name: str
    ) -> Tuple[Position, ...]:
        if positions is None:
            return tuple((0.0, 0.0) for _ in range(count))
        values = tuple((float(item[0]), float(item[1])) for item in positions)
        if len(values) != count:
            raise ValueError(f"{name} initial positions must match fleet size")
        return values

    mbr_positions = resolve_positions(mbr_initial_positions, n_mbr, "MBR")
    dor_positions = resolve_positions(dor_initial_positions, n_dor, "DOR")
    mbrs = [
        RobotState(position=mbr_positions[index], speed=resolved_mbr_speed)
        for index in range(n_mbr)
    ]
    dors = [
        RobotState(position=dor_positions[index], speed=resolved_dor_speed)
        for index in range(n_dor)
    ]
    completions = [float("nan")] * len(tasks)
    seen = set()
    system_distance = mbr_distance = dor_distance = 0.0
    total_mbr_wait = total_dor_wait = 0.0
    executions = []

    for task_id, mbr_id, dor_id in assignments:
        if task_id in seen:
            raise ValueError(f"task {task_id} appears more than once")
        if not 0 <= task_id < len(tasks):
            raise IndexError(f"invalid task id {task_id}")
        if not 0 <= mbr_id < n_mbr or not 0 <= dor_id < n_dor:
            raise IndexError("invalid robot id")
        seen.add(task_id)
        mbr_before = mbrs[mbr_id]
        dor_before = dors[dor_id]
        result = marsupial_transition(
            mbr_before,
            dor_before,
            tasks[task_id],
            None,
            dock_time,
            detach_time,
        )
        destination = tasks[task_id].destination
        mbrs[mbr_id] = RobotState(
            destination, result.mbr_release_time, resolved_mbr_speed
        )
        dors[dor_id] = RobotState(
            destination, result.dor_release_time, resolved_dor_speed
        )
        completions[task_id] = result.task_completion_time
        system_distance += result.system_distance
        mbr_distance += result.mbr_distance
        dor_distance += result.dor_distance
        total_mbr_wait += result.mbr_wait
        total_dor_wait += result.dor_wait
        executions.append(TaskExecution(task_id, mbr_id, dor_id, mbr_before, dor_before, result))

    success = len(seen) == len(tasks)
    if not success:
        raise ValueError(f"plan completed {len(seen)} of {len(tasks)} tasks")
    makespan = max(completions)
    tardiness = [max(0.0, completions[i] - tasks[i].due_time) for i in range(len(tasks))]
    total_tardiness = sum(tardiness)
    objective = objective_value_for_mode(
        objective_mode,
        total_tardiness,
        system_distance,
        len(tasks),
        speed,
        makespan=makespan,
        mbr_speed=resolved_mbr_speed,
    )
    return PlanMetrics(
        objective=objective,
        total_tardiness=total_tardiness,
        average_tardiness=total_tardiness / len(tasks),
        on_time_rate=sum(value == 0.0 for value in tardiness) / len(tasks),
        system_distance=system_distance,
        mbr_distance=mbr_distance,
        dor_distance=dor_distance,
        average_mbr_wait=total_mbr_wait / len(tasks),
        average_dor_wait=total_dor_wait / len(tasks),
        makespan=makespan,
        success=True,
        completion_times=tuple(completions),
        executions=tuple(executions),
    )


def metrics_from_executions(
    tasks: Sequence[TaskSpec],
    executions: Sequence[TaskExecution],
    speed: float,
    mbr_speed: float,
    objective_mode: str,
) -> PlanMetrics:
    """Summarize already executed transitions without nominal-time replay.

    This path is used by evaluation-only uncertainty runs.  Task observations
    retain nominal durations, while each stored transition contains the actual
    realized travel and operation times.
    """
    if not tasks:
        raise ValueError("at least one task is required")
    if speed <= 0 or mbr_speed <= 0:
        raise ValueError("speeds must be positive")
    objective_mode = normalize_objective_mode(objective_mode)
    completions = [float("nan")] * len(tasks)
    seen = set()
    system_distance = mbr_distance = dor_distance = 0.0
    total_mbr_wait = total_dor_wait = 0.0
    for execution in executions:
        task_id = int(execution.task_id)
        if task_id in seen:
            raise ValueError(f"task {task_id} appears more than once")
        if not 0 <= task_id < len(tasks):
            raise IndexError(f"invalid task id {task_id}")
        seen.add(task_id)
        result = execution.result
        completions[task_id] = result.task_completion_time
        system_distance += result.system_distance
        mbr_distance += result.mbr_distance
        dor_distance += result.dor_distance
        total_mbr_wait += result.mbr_wait
        total_dor_wait += result.dor_wait
    if len(seen) != len(tasks):
        raise ValueError(f"execution completed {len(seen)} of {len(tasks)} tasks")
    makespan = max(completions)
    tardiness = [
        max(0.0, completions[index] - tasks[index].due_time)
        for index in range(len(tasks))
    ]
    total_tardiness = sum(tardiness)
    objective = objective_value_for_mode(
        objective_mode,
        total_tardiness,
        system_distance,
        len(tasks),
        speed,
        makespan=makespan,
        mbr_speed=mbr_speed,
    )
    return PlanMetrics(
        objective=objective,
        total_tardiness=total_tardiness,
        average_tardiness=total_tardiness / len(tasks),
        on_time_rate=sum(value == 0.0 for value in tardiness) / len(tasks),
        system_distance=system_distance,
        mbr_distance=mbr_distance,
        dor_distance=dor_distance,
        average_mbr_wait=total_mbr_wait / len(tasks),
        average_dor_wait=total_dor_wait / len(tasks),
        makespan=makespan,
        success=True,
        completion_times=tuple(completions),
        executions=tuple(executions),
    )
