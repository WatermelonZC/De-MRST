"""Numerical consistency checks independent of Gurobi's Python runtime.

Validate discrete routes and solver residuals *before* using coefficient-
weighted residuals as an error budget. Physical replay remains authoritative;
this does not change or certify the optimizer's termination status.
"""

from collections import Counter
from math import fsum, isfinite


class CompactConsistencyError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def validate_compact_replay(*, tasks, plan, replay, raw, geometry,
                            alpha, beta, mbr_speed, feasibility_tol,
                            integrality_tol):
    """Return diagnostics or reject structural/material inconsistencies.

    raw contains a/b/z/release dictionaries and distance/makespan/objective.
    geometry contains mp/dp, distances/duration/post, lower/upper and horizon.
    Time error is propagated over the selected predecessor DAG; distance
    error is bounded by sum(abs(d)*abs(z-z_logical)), not by objective units.
    """
    report = dict(version=2, feasibility_tol=feasibility_tol,
                  integrality_tol=integrality_tol, accepted=False)

    def require(condition, message):
        if not condition:
            raise CompactConsistencyError(message, report)

    n = len(tasks)
    a, b, z, release = (raw[k] for k in ("a", "b", "z", "release"))
    mp, dp = geometry["mp"], geometry["dp"]
    distances, duration, post = (geometry[k] for k in ("distances", "duration", "post"))
    values = [*a.values(), *b.values(), *z.values(), *release.values(),
              raw["distance"], raw["makespan"], raw["objective"],
              replay.system_distance, replay.makespan, replay.objective]
    require(all(isfinite(v) for v in values), "non-finite solution or replay value")
    require(len(plan) == n and set(j for j, _, _ in plan) == set(range(n)),
            "plan does not contain each task exactly once")
    # Arithmetic allowances are in the units of the expression being checked.
    equation_roundoff = 1e-9
    time_roundoff = 1e-8 + 1e-12 * max(1.0, abs(geometry["horizon"]), abs(raw["makespan"]))
    feasibility = feasibility_tol + equation_roundoff
    binary_values = [*a.values(), *b.values()]
    report["max_binary_integrality_residual"] = max(abs(v-round(v)) for v in binary_values)
    report["max_variable_bound_residual"] = max(max(0.0, -v, v-1.0) for v in [*binary_values, *z.values()])
    require(report["max_binary_integrality_residual"] <= integrality_tol + equation_roundoff,
            "binary predecessor values exceed integrality tolerance")
    require(report["max_variable_bound_residual"] <= feasibility,
            "predecessor or joint-choice values exceed bound tolerance")
    pm, pd = {}, {}
    max_assignment = max_marginal = 0.0
    for j in range(n):
        selected_m = [p for p in mp[j] if a[p,j] > .5]
        selected_d = [q for q in dp[j] if b[q,j] > .5]
        require(len(selected_m) == len(selected_d) == 1, "task lacks a unique selected predecessor")
        pm[j], pd[j] = selected_m[0], selected_d[0]
        max_assignment = max(max_assignment, abs(fsum(a[p,j] for p in mp[j])-1),
                              abs(fsum(b[q,j] for q in dp[j])-1))
        for p in mp[j]:
            max_marginal = max(max_marginal, abs(fsum(z[j,p,q] for q in dp[j])-a[p,j]))
        for q in dp[j]:
            max_marginal = max(max_marginal, abs(fsum(z[j,p,q] for p in mp[j])-b[q,j]))
    report.update(max_assignment_residual=max_assignment, max_marginal_residual=max_marginal)
    require(max_assignment <= feasibility, "assignment residual exceeds feasibility tolerance")
    require(max_marginal <= feasibility, "joint-choice marginal residual exceeds feasibility tolerance")
    require(max(Counter(pm.values()).values()) <= 1 and max(Counter(pd.values()).values()) <= 1,
            "rounded predecessor routes branch or reuse a robot source")

    exact_distance = fsum(distances[j,pm[j],pd[j]] for j in range(n))
    weighted_distance = fsum(distances[k]*v for k,v in z.items())
    rounding_mass = fsum(abs(distances[j,p,q]) * abs(v-float(p == pm[j] and q == pd[j]))
                         for (j,p,q),v in z.items())
    distance_roundoff = 1e-8 + 1e-12 * max(1.0, abs(exact_distance),
        abs(replay.system_distance), fsum(abs(distances[k]*v) for k,v in z.items()))
    report.update(model_distance=raw["distance"], weighted_distance=weighted_distance,
                  logical_distance=exact_distance, replay_distance=replay.system_distance,
                  distance_residual=raw["distance"]-replay.system_distance,
                  distance_residual_budget=rounding_mass+distance_roundoff,
                  distance_arithmetic_tolerance=distance_roundoff)
    require(abs(raw["distance"]-weighted_distance) <= distance_roundoff,
            "model distance expression disagrees with its coefficients")
    require(abs(exact_distance-replay.system_distance) <= distance_roundoff,
            "logical predecessor distance disagrees with scalar replay")
    require(abs(raw["distance"]-replay.system_distance) <= rounding_mass+distance_roundoff,
            "distance mismatch is not explained by numerical residuals")

    errors, owners_m, owners_d = {}, {}, {}
    replay_release = {e.task_id:e.result.mbr_release_time for e in replay.executions}
    require(set(replay_release) == set(range(n)), "replay lacks task release times")
    for j, m, o in plan:
        p, q = pm[j], pd[j]
        require((p < 0 or p in errors) and (q < 0 or q in errors),
                "plan is not topological or predecessor graph is cyclic")
        owners_m[j] = -1-p if p < 0 else owners_m[p]
        owners_d[j] = -1-q if q < 0 else owners_d[q]
        require((m,o) == (owners_m[j],owners_d[j]), "plan robot IDs disagree with route sources")
        previous_m = 0.0 if p < 0 else release[p]
        previous_d = 0.0 if q < 0 else release[q]+tasks[q].handling_time
        m_big = 0.0 if p < 0 else max(0.0,geometry["upper"][p]-geometry["lower"][j])
        d_big = 0.0 if q < 0 else max(0.0,geometry["horizon"]-geometry["lower"][j])
        row_duration = fsum(duration[j,p,r]*z[j,p,r] for r in dp[j])
        m_leak = m_big*(1-a[p,j])
        d_leak = d_big*(1-b[q,j])
        require(previous_m+row_duration-m_leak-release[j] <= feasibility_tol+time_roundoff,
                "selected MBR time constraint exceeds feasibility tolerance")
        require(previous_d+post[q,j]*b[q,j]-d_leak-release[j] <= feasibility_tol+time_roundoff,
                "selected DOR time constraint exceeds feasibility tolerance")
        m_error = abs(duration[j,p,q]-row_duration)+abs(m_leak)+feasibility_tol+time_roundoff
        d_error = abs(post[q,j]*(1-b[q,j]))+abs(d_leak)+feasibility_tol+time_roundoff
        errors[j] = max((0.0 if p < 0 else errors[p])+m_error,
                        (0.0 if q < 0 else errors[q])+d_error)
        require(replay_release[j]-release[j] <= errors[j]+time_roundoff,
                "scalar release time exceeds propagated numerical error budget")
        require(release[j]+tasks[j].handling_time-raw["makespan"] <= feasibility_tol+time_roundoff,
                "model makespan is inconsistent with its release times")
    span_budget = max(errors.values())+feasibility_tol+time_roundoff
    objective_roundoff = 1e-8 + 1e-12*max(1.0,abs(raw["objective"]),abs(replay.objective))
    objective_budget = alpha*span_budget + beta*(rounding_mass+distance_roundoff)/mbr_speed + objective_roundoff
    report.update(model_objective=raw["objective"], replay_objective=replay.objective,
                  objective_residual=replay.objective-raw["objective"],
                  objective_residual_budget=objective_budget, makespan_residual_budget=span_budget,
                  per_task_release_error_budget=errors)
    require(abs(raw["objective"]-(alpha*raw["makespan"]+beta*raw["distance"]/mbr_speed)) <= objective_roundoff,
            "model objective expression disagrees with its components")
    require(abs(replay.objective-(alpha*replay.makespan+beta*replay.system_distance/mbr_speed)) <= objective_roundoff,
            "scalar objective disagrees with its components")
    require(replay.makespan-raw["makespan"] <= span_budget,
            "replayed makespan exceeds numerical error budget")
    require(replay.objective-raw["objective"] <= objective_budget,
            "model objective understates scalar replay beyond numerical error budget")
    report["accepted"] = True
    report["status"] = ("consistent_with_numerical_residual" if
        abs(report["distance_residual"]) > distance_roundoff or report["objective_residual"] > objective_roundoff
        else "consistent")
    return report
