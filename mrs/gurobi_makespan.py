"""Centralized Gurobi MILP for the paper objective.

This model uses the same transition convention as :mod:`mrs.core`: an MBR
travels to the DOR's current location, the pair docks, and the MBR then
executes the coupled transport. It supports maximum DOR release time and
the 0.7 makespan + 0.3 carrier-travel-time objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import Optional, Sequence, Tuple

from .core import (
    DEFAULT_DOR_SPEED,
    DEFAULT_MBR_SPEED,
    TaskSpec,
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
    formulation: str = "compact"
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
    """Solve the paper's compact MILP and replay its resulting plan."""
    if objective_mode not in ("makespan", "makespan_distance"):
        raise ValueError("Gurobi supports makespan or makespan_distance")
    if formulation != "compact":
        raise ValueError("only the paper's compact formulation is available")
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
