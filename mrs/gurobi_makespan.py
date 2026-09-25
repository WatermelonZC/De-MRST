"""Centralized Gurobi MILPs for makespan and the v18 weighted objective.

This model uses the same transition convention as :mod:`mrs.core`: an MBR
travels to the DOR's current location, the pair docks, and the MBR then
executes the coupled transport. It supports maximum DOR release time and
the v18 0.7 makespan + 0.3 carrier-travel-time objective, never tardiness.
The older top-level ``run_gurobi.py`` solves a different tardiness-
distance VRP objective and is intentionally not used for this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, nan
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from .core import (
    DEFAULT_DOR_SPEED,
    DEFAULT_MBR_SPEED,
    OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT,
    OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT,
    TaskSpec,
    evaluate_centralized_plan,
    manhattan_distance,
    validate_role_speed_order,
)


@dataclass(frozen=True)
class GurobiMakespanResult:
    makespan: float
    evaluated_makespan: float
    lower_bound: float
    mip_gap: float
    runtime_s: float
    status: str
    success: bool
    plan: Tuple[Tuple[int, int, int], ...]
    objective: float = nan
    evaluated_objective: float = nan
    system_distance: float = nan
    objective_mode: str = "makespan"
    formulation: str = "legacy"
    build_time_s: float = 0.0
    total_time_s: float = 0.0
    node_count: float = 0.0
    work: float = 0.0
    num_vars: int = 0
    num_bin_vars: int = 0
    num_constrs: int = 0
    model_system_distance: Optional[float] = None
    distance_residual: Optional[float] = None
    distance_residual_budget: Optional[float] = None
    objective_residual_budget: Optional[float] = None
    consistency_status: Optional[str] = None
    replay_mip_gap: Optional[float] = None
    solution_artifacts_dir: Optional[str] = None


def _position(
    tasks: Sequence[TaskSpec],
    task_id: int,
    initial_positions: Sequence[Tuple[float, float]],
    robot_id: int,
) -> Tuple[float, float]:
    return (
        tuple(initial_positions[robot_id])
        if task_id < 0
        else tasks[task_id].destination
    )


def solve_centralized_makespan(
    tasks: Sequence[TaskSpec],
    n_mbr: int,
    n_dor: int,
    speed: float,
    *,
    mbr_speed: Optional[float] = DEFAULT_MBR_SPEED,
    dor_speed: Optional[float] = DEFAULT_DOR_SPEED,
    dock_time: float = 8.0,
    detach_time: float = 8.0,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    output_flag: bool = False,
    initial_plan: Optional[Sequence[Tuple[int, int, int]]] = None,
    mbr_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    dor_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    objective_mode: str = "makespan",
    formulation: str = "compact",
    threads: Optional[int] = None,
    seed: int = 0,
    log_file: Optional[str] = None,
    progress_callback=None,
    solution_artifacts_dir: Optional[str] = None,
) -> GurobiMakespanResult:
    """Solve and replay a plan; bounds/gap always refer to ``objective_mode``.

    ``makespan`` remains the API default for historical callers. The latest
    protocol runner explicitly selects ``makespan_distance``. ``legacy`` keeps
    the original robot-indexed formulation as a same-objective benchmark.
    Compact incumbents are archived before validation; ``solution_artifacts_dir``
    selects the parent directory (otherwise next to the log or in system temp).
    ``status``/``mip_gap`` describe the raw optimizer, not the replay check.
    """
    started = perf_counter()
    if objective_mode not in ("makespan", "makespan_distance"):
        raise ValueError("Gurobi supports makespan or makespan_distance")
    if formulation not in ("legacy", "compact"):
        raise ValueError("formulation must be legacy or compact")
    if not tasks:
        raise ValueError("at least one task is required")
    if n_mbr <= 0 or n_dor <= 0:
        raise ValueError("both robot roles require at least one robot")
    if speed <= 0:
        raise ValueError("speed must be positive")
    mbr_speed = float(DEFAULT_MBR_SPEED if mbr_speed is None else mbr_speed)
    dor_speed = float(DEFAULT_DOR_SPEED if dor_speed is None else dor_speed)
    validate_role_speed_order(mbr_speed, dor_speed)
    if dock_time < 0 or detach_time < 0:
        raise ValueError("dock and detach times must be non-negative")

    mbr_initial_positions = (
        tuple((0.0, 0.0) for _ in range(n_mbr))
        if mbr_initial_positions is None
        else tuple((float(item[0]), float(item[1])) for item in mbr_initial_positions)
    )
    dor_initial_positions = (
        tuple((0.0, 0.0) for _ in range(n_dor))
        if dor_initial_positions is None
        else tuple((float(item[0]), float(item[1])) for item in dor_initial_positions)
    )
    if len(mbr_initial_positions) != n_mbr or len(dor_initial_positions) != n_dor:
        raise ValueError("initial positions must match fleet sizes")

    if formulation == "compact":
        from .gurobi_compact import solve_compact
        return solve_compact(
            tasks, n_mbr, n_dor, speed, mbr_speed=mbr_speed, dor_speed=dor_speed,
            dock_time=dock_time, detach_time=detach_time,
            time_limit_s=time_limit_s, mip_gap=mip_gap, output_flag=output_flag,
            initial_plan=initial_plan, mbr_initial_positions=mbr_initial_positions,
            dor_initial_positions=dor_initial_positions, objective_mode=objective_mode,
            threads=threads, seed=seed, log_file=log_file, progress_callback=progress_callback,
            solution_artifacts_dir=solution_artifacts_dir,
        )

    n = len(tasks)
    previous = tuple([-1, *range(n)])
    model = gp.Model("MarsupialPureMakespan")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.Seed = seed
    if threads is not None:
        model.Params.Threads = threads
    if log_file is not None:
        model.Params.LogFile = log_file
        model.Params.LogToConsole = int(output_flag)
        model.Params.OutputFlag = 1
    if time_limit_s is not None:
        model.Params.TimeLimit = float(time_limit_s)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    pair_keys = [(task, mbr, dor) for task in range(n)
                 for mbr in range(n_mbr) for dor in range(n_dor)]
    y = model.addVars(pair_keys, vtype=GRB.BINARY, name="pair")

    mbr_arc_keys = [
        (previous_task, task, mbr)
        for mbr in range(n_mbr)
        for task in range(n)
        for previous_task in previous
        if previous_task != task
    ]
    dor_arc_keys = [
        (previous_task, task, dor)
        for dor in range(n_dor)
        for task in range(n)
        for previous_task in previous
        if previous_task != task
    ]
    mbr_arc = model.addVars(mbr_arc_keys, vtype=GRB.BINARY, name="mbr_arc")
    dor_arc = model.addVars(dor_arc_keys, vtype=GRB.BINARY, name="dor_arc")

    z_keys = [
        (task, mbr, dor, previous_mbr, previous_dor)
        for task, mbr, dor in pair_keys
        for previous_mbr in previous
        if previous_mbr != task
        for previous_dor in previous
        if previous_dor != task
    ]
    z = model.addVars(z_keys, vtype=GRB.BINARY, name="predecessor_pair")

    mbr_release = model.addVars(range(n), lb=0.0, vtype=GRB.CONTINUOUS, name="mbr_release")
    dor_release = model.addVars(range(n), lb=0.0, vtype=GRB.CONTINUOUS, name="dor_release")
    makespan = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="makespan")

    # One pair per task.
    for task in range(n):
        model.addConstr(
            gp.quicksum(y[task, mbr, dor] for mbr in range(n_mbr) for dor in range(n_dor)) == 1,
            name=f"assign_{task}",
        )

    # Every assigned task has one predecessor in each role route.
    for task in range(n):
        for mbr in range(n_mbr):
            model.addConstr(
                gp.quicksum(mbr_arc[previous_task, task, mbr]
                            for previous_task in previous if previous_task != task)
                == gp.quicksum(y[task, mbr, dor] for dor in range(n_dor)),
                name=f"mbr_in_{task}_{mbr}",
            )
        for dor in range(n_dor):
            model.addConstr(
                gp.quicksum(dor_arc[previous_task, task, dor]
                            for previous_task in previous if previous_task != task)
                == gp.quicksum(y[task, mbr, dor] for mbr in range(n_mbr)),
                name=f"dor_in_{task}_{dor}",
            )

    # Each route starts at the depot at most once and each task has at most one
    # successor.  Positive transition times plus the release constraints below
    # eliminate disconnected cycles.
    for mbr in range(n_mbr):
        model.addConstr(
            gp.quicksum(mbr_arc[-1, task, mbr] for task in range(n)) <= 1,
            name=f"mbr_start_{mbr}",
        )
        for previous_task in range(n):
            model.addConstr(
                gp.quicksum(mbr_arc[previous_task, task, mbr]
                            for task in range(n) if task != previous_task)
                <= gp.quicksum(y[previous_task, mbr, dor] for dor in range(n_dor)),
                name=f"mbr_out_{previous_task}_{mbr}",
            )
    for dor in range(n_dor):
        model.addConstr(
            gp.quicksum(dor_arc[-1, task, dor] for task in range(n)) <= 1,
            name=f"dor_start_{dor}",
        )
        for previous_task in range(n):
            model.addConstr(
                gp.quicksum(dor_arc[previous_task, task, dor]
                            for task in range(n) if task != previous_task)
                <= gp.quicksum(y[previous_task, mbr, dor] for mbr in range(n_mbr)),
                name=f"dor_out_{previous_task}_{dor}",
            )

    horizon = max(
        10_000.0,
        2.0 * n * (
            200.0 / mbr_speed
            + dock_time
            + detach_time
            + max(task.pickup_time + task.handling_time for task in tasks)
        )
        + 100.0,
    )

    def release_value(variable, task_id: int):
        return 0.0 if task_id < 0 else variable[task_id]

    # Couple pair assignment with the selected predecessor in both role routes.
    # z also activates the exact physical transition inequalities.
    distance_terms = []
    for task, mbr, dor in pair_keys:
        combinations = [
            (previous_mbr, previous_dor)
            for previous_mbr in previous if previous_mbr != task
            for previous_dor in previous if previous_dor != task
        ]
        model.addConstr(
            gp.quicksum(z[task, mbr, dor, previous_mbr, previous_dor]
                        for previous_mbr, previous_dor in combinations)
            == y[task, mbr, dor],
            name=f"predecessor_choice_{task}_{mbr}_{dor}",
        )
        for previous_mbr, previous_dor in combinations:
            item = z[task, mbr, dor, previous_mbr, previous_dor]
            model.addConstr(
                item <= mbr_arc[previous_mbr, task, mbr],
                name=f"z_mbr_arc_{task}_{mbr}_{dor}_{previous_mbr}_{previous_dor}",
            )
            model.addConstr(
                item <= dor_arc[previous_dor, task, dor],
                name=f"z_dor_arc_{task}_{mbr}_{dor}_{previous_mbr}_{previous_dor}",
            )

            mbr_position = _position(tasks, previous_mbr, mbr_initial_positions, mbr)
            dor_position = _position(tasks, previous_dor, dor_initial_positions, dor)
            rendezvous = manhattan_distance(mbr_position, dor_position)
            rendezvous_to_source = manhattan_distance(dor_position, tasks[task].source)
            transport = manhattan_distance(tasks[task].source, tasks[task].destination)
            distance_terms.append((rendezvous + rendezvous_to_source + transport) * item)
            post_dock = (
                dock_time
                + rendezvous_to_source / mbr_speed
                + tasks[task].pickup_time
                + transport / mbr_speed
                + detach_time
            )
            model.addConstr(
                mbr_release[task]
                >= release_value(mbr_release, previous_mbr)
                + rendezvous / mbr_speed
                + post_dock
                - horizon * (1 - item),
                name=f"mbr_transition_{task}_{mbr}_{dor}_{previous_mbr}_{previous_dor}",
            )
            model.addConstr(
                mbr_release[task]
                >= release_value(dor_release, previous_dor)
                + post_dock
                - horizon * (1 - item),
                name=f"dock_sync_{task}_{mbr}_{dor}_{previous_mbr}_{previous_dor}",
            )

    for task in range(n):
        model.addConstr(
            dor_release[task] == mbr_release[task] + tasks[task].handling_time,
            name=f"handling_{task}",
        )
        model.addConstr(makespan >= dor_release[task], name=f"makespan_{task}")

    if initial_plan is not None:
        # Warm-start only.  The plan is not fixed, so Gurobi can still search
        # the full centralized assignment/order space.
        initial_plan = tuple(initial_plan)
        initial_metrics = evaluate_centralized_plan(
            tasks,
            initial_plan,
            n_mbr,
            n_dor,
            speed,
            dock_time,
            detach_time,
            mbr_speed=mbr_speed,
            dor_speed=dor_speed,
            mbr_initial_positions=mbr_initial_positions,
            dor_initial_positions=dor_initial_positions,
            objective_mode=objective_mode,
        )
        for variables in (y, mbr_arc, dor_arc, z):
            for variable in variables.values():
                variable.Start = 0.0
        mbr_previous = [-1] * n_mbr
        dor_previous = [-1] * n_dor
        for execution in initial_metrics.executions:
            task, mbr, dor = execution.task_id, execution.mbr_id, execution.dor_id
            y[task, mbr, dor].Start = 1.0
            mbr_arc[mbr_previous[mbr], task, mbr].Start = 1.0
            dor_arc[dor_previous[dor], task, dor].Start = 1.0
            previous_mbr = mbr_previous[mbr]
            previous_dor = dor_previous[dor]
            z[task, mbr, dor, previous_mbr, previous_dor].Start = 1.0
            mbr_previous[mbr] = task
            dor_previous[dor] = task
            mbr_release[task].Start = execution.result.mbr_release_time
            dor_release[task].Start = execution.result.dor_release_time
        makespan.Start = initial_metrics.makespan

    distance = gp.quicksum(distance_terms)
    objective = makespan if objective_mode == "makespan" else (
        OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT * makespan
        + OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT * distance / mbr_speed
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()
    build_time = perf_counter() - started
    model.optimize()

    diagnostics = dict(
        objective_mode=objective_mode, formulation="legacy", build_time_s=build_time,
        node_count=float(model.NodeCount), work=float(model.Work),
        num_vars=model.NumVars, num_bin_vars=model.NumBinVars, num_constrs=model.NumConstrs,
    )

    status = str(model.Status)
    if model.SolCount <= 0:
        result = GurobiMakespanResult(
            makespan=inf,
            evaluated_makespan=inf,
            lower_bound=float(model.ObjBound),
            mip_gap=inf,
            runtime_s=float(model.Runtime),
            status=status,
            success=False,
            plan=tuple(),
            objective=inf, evaluated_objective=inf, system_distance=inf,
            total_time_s=perf_counter() - started, **diagnostics,
        )
        model.dispose()
        return result

    selected_pairs: Dict[int, Tuple[int, int]] = {}
    for task, mbr, dor in pair_keys:
        if y[task, mbr, dor].X > 0.5:
            selected_pairs[task] = (mbr, dor)

    predecessors: Dict[int, List[int]] = {task: [] for task in range(n)}
    successors: Dict[int, List[int]] = {task: [] for task in range(n)}
    for previous_task, task, mbr in mbr_arc_keys:
        if mbr_arc[previous_task, task, mbr].X > 0.5 and previous_task >= 0:
            successors[previous_task].append(task)
            predecessors[task].append(previous_task)
    for previous_task, task, dor in dor_arc_keys:
        if dor_arc[previous_task, task, dor].X > 0.5 and previous_task >= 0:
            successors[previous_task].append(task)
            predecessors[task].append(previous_task)

    indegree = {task: len(set(predecessors[task])) for task in range(n)}
    ready = sorted(
        (task for task in range(n) if indegree[task] == 0),
        key=lambda task: (mbr_release[task].X, dor_release[task].X, task),
    )
    order: List[int] = []
    while ready:
        task = ready.pop(0)
        order.append(task)
        for successor in set(successors[task]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort(key=lambda item: (mbr_release[item].X, dor_release[item].X, item))
    if len(order) != n:
        order = sorted(range(n), key=lambda task: (dor_release[task].X, task))

    plan = tuple((task, *selected_pairs[task]) for task in order)
    replay = evaluate_centralized_plan(
        tasks,
        plan,
        n_mbr,
        n_dor,
        speed,
        dock_time,
        detach_time,
        mbr_speed=mbr_speed,
        dor_speed=dor_speed,
        mbr_initial_positions=mbr_initial_positions,
        dor_initial_positions=dor_initial_positions,
        objective_mode=objective_mode,
    )
    result = GurobiMakespanResult(
        makespan=float(makespan.X),
        evaluated_makespan=float(replay.makespan),
        lower_bound=float(model.ObjBound),
        mip_gap=float(model.MIPGap),
        runtime_s=float(model.Runtime),
        status=status,
        success=True,
        plan=plan,
        objective=float(model.ObjVal), evaluated_objective=float(replay.objective),
        system_distance=float(replay.system_distance), total_time_s=perf_counter() - started,
        **diagnostics,
    )
    model.dispose()
    return result
