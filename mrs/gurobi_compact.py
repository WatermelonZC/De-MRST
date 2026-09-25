"""Exact path-cover MILP for homogeneous-per-role Marsupial fleets.

Each physical robot has its own dummy source (including its actual position).
Task predecessors need no robot index: following the path to its source
recovers that index. Continuous transportation variables couple the two
binary predecessor choices; their marginals force an integral joint choice.
No fixed partnerships, candidate truncation, or task-order restriction is used.
"""

from dataclasses import asdict
import json
from math import inf, isfinite
from pathlib import Path
import tempfile
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB

from .gurobi_validation import validate_compact_replay

from .core import (
    OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT,
    OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT,
    RobotState,
    evaluate_centralized_plan,
    manhattan_distance,
    marsupial_transition,
)


def _write_diagnostics(path, payload):
    def safe(value):
        if isinstance(value, float) and not isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(v) for v in value]
        return value
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _archive_before_validation(model, raw_distance, log_file, requested_directory):
    """Never lose a returned incumbent if extraction or replay validation fails."""
    root = (Path(requested_directory) if requested_directory is not None else
            Path(log_file).parent / (Path(log_file).stem + "_solutions") if log_file is not None else
            Path(tempfile.gettempdir()) / "mrs_gurobi_solutions")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="solution_", dir=str(root))).resolve()
    model.write(str(folder / "model.mps"))
    model.write(str(folder / "incumbent.sol"))
    # Whitelist solver settings: never dump environment/license credentials.
    report = dict(solver_status=str(model.Status), model_objective=float(model.ObjVal),
                  lower_bound=float(model.ObjBound), solver_mip_gap=float(model.MIPGap),
                  model_distance=raw_distance, fingerprint=hex(model.Fingerprint & 0xffffffff),
                  solution_count=model.SolCount,
                  solver_parameters={name: getattr(model.Params, name) for name in
                      ("TimeLimit", "MIPGap", "Threads", "Seed", "FeasibilityTol", "IntFeasTol")},
                  quality={name: float(getattr(model, name)) for name in ("ConstrVio", "IntVio", "BoundVio")},
                  verification=dict(status="pending", accepted=False))
    _write_diagnostics(folder / "validation.json", report)
    return folder, report


def greedy_start(tasks, n_mbr, n_dor, speed, *, mbr_speed=1.2, dor_speed=2.4,
                 dock_time=8.0, detach_time=8.0, mbr_initial_positions=None,
                 dor_initial_positions=None, objective_mode="makespan_distance"):
    """Deterministic feasible start; also supplied identically to both benchmarks."""
    mbrs = [RobotState((0.0, 0.0) if mbr_initial_positions is None else
                      tuple(mbr_initial_positions[k]), speed=mbr_speed) for k in range(n_mbr)]
    dors = [RobotState((0.0, 0.0) if dor_initial_positions is None else
                      tuple(dor_initial_positions[k]), speed=dor_speed) for k in range(n_dor)]
    remaining = set(range(len(tasks)))
    plan, span, distance = [], 0.0, 0.0
    alpha, beta = ((1.0, 0.0) if objective_mode == "makespan" else
                   (OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT, OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT))
    while remaining:
        best = None
        for j in sorted(remaining):
            for m in range(n_mbr):
                for o in range(n_dor):
                    execution = marsupial_transition(mbrs[m], dors[o], tasks[j],
                                                     None, dock_time, detach_time)
                    score = (alpha * max(span, execution.dor_release_time)
                             + beta * (distance + execution.system_distance) / mbr_speed)
                    key = (score, j, m, o)
                    if best is None or key < best[0]:
                        best = (key, execution)
        (_, j, m, o), execution = best
        plan.append((j, m, o))
        remaining.remove(j)
        span = max(span, execution.dor_release_time)
        distance += execution.system_distance
        mbrs[m] = RobotState(tasks[j].destination, execution.mbr_release_time, mbr_speed)
        dors[o] = RobotState(tasks[j].destination, execution.dor_release_time, dor_speed)
    return tuple(plan)


def solve_compact(tasks, n_mbr, n_dor, speed, *, mbr_speed, dor_speed, dock_time,
                  detach_time, time_limit_s, mip_gap, output_flag, initial_plan,
                  mbr_initial_positions, dor_initial_positions, objective_mode,
                  threads, seed, log_file, progress_callback=None, solution_artifacts_dir=None):
    from .gurobi_makespan import GurobiMakespanResult

    started = perf_counter()
    n = len(tasks)
    for task in tasks:
        if any(not isfinite(v) or v < 0 for v in (task.pickup_time, task.handling_time)):
            raise ValueError("task pickup/handling times must be finite and non-negative")
    coordinates = [*mbr_initial_positions, *dor_initial_positions]
    coordinates += [p for task in tasks for p in (task.source, task.destination)]
    if any(not isfinite(float(c)) for position in coordinates for c in position):
        raise ValueError("positions must be finite")
    if any(not isfinite(v) for v in (speed, mbr_speed, dor_speed, dock_time, detach_time)):
        raise ValueError("speeds and coupling times must be finite")
    replay_kwargs = dict(mbr_speed=mbr_speed, dor_speed=dor_speed,
                         dock_time=dock_time, detach_time=detach_time,
                         mbr_initial_positions=mbr_initial_positions,
                         dor_initial_positions=dor_initial_positions,
                         objective_mode=objective_mode)
    if initial_plan is None:
        initial_plan = greedy_start(tasks, n_mbr, n_dor, speed, **replay_kwargs)
    initial_plan = tuple(initial_plan)
    incumbent = evaluate_centralized_plan(tasks, initial_plan, n_mbr, n_dor, speed,
                                          **replay_kwargs)
    alpha, beta = ((1.0, 0.0) if objective_mode == "makespan" else
                   (OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT, OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT))
    # Negative predecessor -1-k is physical robot k's source, not a shared depot.
    mp = {j: tuple(p for p in range(-n_mbr, n) if p != j) for j in range(n)}
    dp = {j: tuple(q for q in range(-n_dor, n) if q != j) for j in range(n)}
    mpos = {p: (mbr_initial_positions[-1-p] if p < 0 else tasks[p].destination)
            for p in range(-n_mbr, n)}
    dpos = {q: (dor_initial_positions[-1-q] if q < 0 else tasks[q].destination)
            for q in range(-n_dor, n)}
    transport = {j: manhattan_distance(t.source, t.destination) for j, t in enumerate(tasks)}
    post = {(q, j): dock_time + detach_time + tasks[j].pickup_time
            + (manhattan_distance(dpos[q], tasks[j].source) + transport[j]) / mbr_speed
            for j in range(n) for q in dp[j]}
    keys = [(j, p, q) for j in range(n) for p in mp[j] for q in dp[j]]
    distances = {(j, p, q): manhattan_distance(mpos[p], dpos[q])
                 + manhattan_distance(dpos[q], tasks[j].source) + transport[j]
                 for j, p, q in keys}
    duration = {(j, p, q): manhattan_distance(mpos[p], dpos[q]) / mbr_speed + post[q, j]
                for j, p, q in keys}
    lower = {j: min(duration[j, p, q] for p in mp[j] for q in dp[j]) for j in range(n)}
    distance_lb = sum(min(distances[j, p, q] for p in mp[j] for q in dp[j]) for j in range(n))
    # A weighted-objective incumbent does NOT bound makespan by its own makespan.
    # Every improving solution satisfies alpha*C + beta*D/v <= incumbent J.
    horizon = (incumbent.objective - beta * distance_lb / mbr_speed) / alpha
    horizon += max(1e-6, abs(horizon) * 1e-9)
    upper = {j: horizon - tasks[j].handling_time for j in range(n)}

    with gp.Model("MarsupialCompact") as model:
        model.Params.OutputFlag = int(output_flag)
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
        a = model.addVars([(p, j) for j in range(n) for p in mp[j]], vtype=GRB.BINARY, name="mbr_arc")
        b = model.addVars([(q, j) for j in range(n) for q in dp[j]], vtype=GRB.BINARY, name="dor_arc")
        z = model.addVars(keys, lb=0.0, ub=1.0, name="joint_predecessor")
        release = model.addVars(range(n), lb=lower, ub=upper, name="mbr_release")
        makespan = model.addVar(lb=max(lower[j] + tasks[j].handling_time for j in range(n)),
                                ub=horizon, name="makespan")
        # Shared ranks exclude even zero-duration cycles across the union of routes.
        rank = model.addVars(range(n), lb=0.0, ub=n-1, name="rank")
        for j in range(n):
            model.addConstr(gp.quicksum(a[p, j] for p in mp[j]) == 1, name=f"mbr_in_{j}")
            model.addConstr(gp.quicksum(b[q, j] for q in dp[j]) == 1, name=f"dor_in_{j}")
            model.addConstr(makespan >= release[j] + tasks[j].handling_time)
            # Transportation marginals: binary a,b imply exactly one z=1.
            for p in mp[j]:
                model.addConstr(gp.quicksum(z[j, p, q] for q in dp[j]) == a[p, j])
                work = gp.quicksum(duration[j, p, q] * z[j, p, q] for q in dp[j])
                previous_release = 0.0 if p < 0 else release[p]
                big_m = 0.0 if p < 0 else max(0.0, upper[p] - lower[j])
                model.addConstr(release[j] >= previous_release + work - big_m * (1-a[p, j]))
                if p >= 0:
                    model.addConstr(rank[j] >= rank[p] + 1 - n * (1-a[p, j]))
            for q in dp[j]:
                model.addConstr(gp.quicksum(z[j, p, q] for p in mp[j]) == b[q, j])
                previous_release = 0.0 if q < 0 else release[q] + tasks[q].handling_time
                big_m = 0.0 if q < 0 else max(0.0, horizon - lower[j])
                model.addConstr(release[j] >= previous_release + post[q, j] * b[q, j]
                                - big_m * (1-b[q, j]))
                if q >= 0:
                    model.addConstr(rank[j] >= rank[q] + 1 - n * (1-b[q, j]))
            model.addConstr(release[j] >= gp.quicksum(duration[j, p, q] * z[j, p, q]
                                                     for p in mp[j] for q in dp[j]))
        for p in range(-n_mbr, n):
            model.addConstr(gp.quicksum(a[p, j] for j in range(n) if j != p) <= 1)
        for q in range(-n_dor, n):
            model.addConstr(gp.quicksum(b[q, j] for j in range(n) if j != q) <= 1)
        distance = gp.quicksum(distances[key] * z[key] for key in keys)
        mbr_work = distance / mbr_speed + sum(dock_time + detach_time + t.pickup_time for t in tasks)
        model.addConstr(n_mbr * makespan >= mbr_work, name="mbr_workload_bound")
        dor_work = gp.quicksum((post[q, j] + tasks[j].handling_time) * b[q, j]
                              for j in range(n) for q in dp[j])
        model.addConstr(n_dor * makespan >= dor_work, name="dor_workload_bound")
        model.setObjective(alpha * makespan + beta * distance / mbr_speed, GRB.MINIMIZE)
        for variables in (a, b, z):
            for variable in variables.values():
                variable.Start = 0.0
        prev_m, prev_d = [-1-k for k in range(n_mbr)], [-1-k for k in range(n_dor)]
        for order, execution in enumerate(incumbent.executions):
            j, m, o = execution.task_id, execution.mbr_id, execution.dor_id
            p, q = prev_m[m], prev_d[o]
            a[p, j].Start = b[q, j].Start = z[j, p, q].Start = 1.0
            release[j].Start = execution.result.mbr_release_time
            rank[j].Start = order
            prev_m[m] = prev_d[o] = j
        makespan.Start = incumbent.makespan
        model.update()
        build_time = perf_counter() - started
        if progress_callback is None:
            model.optimize()
        else:
            last_report = [-30.0]
            def report_progress(active, where):
                if where not in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
                    return
                elapsed = active.cbGet(GRB.Callback.RUNTIME)
                if where == GRB.Callback.MIP and elapsed - last_report[0] < 30.0:
                    return
                last_report[0] = elapsed
                if where == GRB.Callback.MIPSOL:
                    incumbent_value = active.cbGet(GRB.Callback.MIPSOL_OBJBST)
                    bound = active.cbGet(GRB.Callback.MIPSOL_OBJBND)
                else:
                    incumbent_value = active.cbGet(GRB.Callback.MIP_OBJBST)
                    bound = active.cbGet(GRB.Callback.MIP_OBJBND)
                progress_callback(incumbent_value, elapsed, bound)
            model.optimize(report_progress)
        diagnostics = dict(objective_mode=objective_mode, formulation="compact",
                           build_time_s=build_time, runtime_s=float(model.Runtime),
                           lower_bound=float(model.ObjBound), status=str(model.Status),
                           node_count=float(model.NodeCount), work=float(model.Work),
                           num_vars=model.NumVars, num_bin_vars=model.NumBinVars,
                           num_constrs=model.NumConstrs)
        if model.SolCount <= 0:
            return GurobiMakespanResult(makespan=inf, evaluated_makespan=inf, mip_gap=inf,
                                       success=False, plan=(), objective=inf,
                                       evaluated_objective=inf, system_distance=inf,
                                       total_time_s=perf_counter()-started, **diagnostics)
        raw_distance = float(distance.getValue())
        artifact_dir, verification_record = _archive_before_validation(
            model, raw_distance, log_file, solution_artifacts_dir)
        try:
            raw = dict(a={k: v.X for k, v in a.items()}, b={k: v.X for k, v in b.items()},
                       z={k: v.X for k, v in z.items()}, release={k: v.X for k, v in release.items()},
                       distance=raw_distance, makespan=float(makespan.X), objective=float(model.ObjVal))
            pm, pd = {}, {}
            for j in range(n):
                selected_m = [p for p in mp[j] if raw["a"][p,j] > .5]
                selected_d = [q for q in dp[j] if raw["b"][q,j] > .5]
                if len(selected_m) != 1 or len(selected_d) != 1:
                    raise RuntimeError("Gurobi returned non-unique task predecessors")
                pm[j], pd[j] = selected_m[0], selected_d[0]
            # Topological extraction, never an arbitrary time-sort fallback.
            remaining, plan, owner_m, owner_d = set(range(n)), [], {}, {}
            while remaining:
                ready = [j for j in sorted(remaining) if pm[j] not in remaining and pd[j] not in remaining]
                if not ready:
                    raise RuntimeError("Gurobi returned a cyclic predecessor graph")
                for j in ready:
                    owner_m[j] = -1-pm[j] if pm[j] < 0 else owner_m[pm[j]]
                    owner_d[j] = -1-pd[j] if pd[j] < 0 else owner_d[pd[j]]
                    plan.append((j, owner_m[j], owner_d[j]))
                    remaining.remove(j)
            verification_record["plan"] = plan
            replay = evaluate_centralized_plan(tasks, plan, n_mbr, n_dor, speed, **replay_kwargs)
            verification_record["replay"] = asdict(replay)
            validation = validate_compact_replay(
                tasks=tasks, plan=plan, replay=replay, raw=raw,
                geometry=dict(mp=mp, dp=dp, distances=distances, duration=duration,
                              post=post, lower=lower, upper=upper, horizon=horizon),
                alpha=alpha, beta=beta, mbr_speed=mbr_speed,
                feasibility_tol=model.Params.FeasibilityTol, integrality_tol=model.Params.IntFeasTol)
            verification_record["verification"] = validation
        except Exception as error:
            verification_record["verification"] = dict(status="failed", accepted=False,
                error_type=type(error).__name__, error=str(error), diagnostics=getattr(error, "diagnostics", None))
            _write_diagnostics(artifact_dir / "validation.json", verification_record)
            raise RuntimeError(f"{error}; raw solution preserved at {artifact_dir}") from error
        _write_diagnostics(artifact_dir / "validation.json", verification_record)
        replay_gap = (abs(replay.objective-model.ObjBound)/abs(replay.objective)
                      if replay.objective != 0 else (0.0 if model.ObjBound == 0 else inf))
        return GurobiMakespanResult(
            makespan=float(makespan.X), evaluated_makespan=replay.makespan,
            mip_gap=float(model.MIPGap), success=True, plan=tuple(plan),
            objective=float(model.ObjVal), evaluated_objective=replay.objective,
            system_distance=replay.system_distance, total_time_s=perf_counter()-started,
            model_system_distance=raw_distance, distance_residual=validation["distance_residual"],
            distance_residual_budget=validation["distance_residual_budget"],
            objective_residual_budget=validation["objective_residual_budget"],
            consistency_status=validation["status"], replay_mip_gap=float(replay_gap),
            solution_artifacts_dir=str(artifact_dir),
            **diagnostics)
