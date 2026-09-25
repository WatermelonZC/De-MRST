"""Run the paper's event-level rules or centralized Gurobi on one instance."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrs.instances import read_instance, write_instance_excel
from mrs.decentralized_market import run_decentralized_market_episode


METHODS = ("d_murdoch", "d_coalition_auction", "d_min_min", "gurobi", "ga", "alns", "am")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="centralized search time limit in seconds")
    parser.add_argument("--checkpoint", type=Path, help="required for the centralized AM baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    instance = read_instance(args.instance)
    kwargs = dict(
        dock_time=instance.dock_time, detach_time=instance.detach_time,
        mbr_speed=instance.resolved_mbr_speed,
        dor_speed=instance.resolved_dor_speed,
        mbr_initial_positions=instance.mbr_initial_positions,
        dor_initial_positions=instance.dor_initial_positions,
        objective_mode="makespan_distance",
    )
    if args.method == "gurobi":
        from mrs.gurobi_makespan import solve_centralized_makespan
        outcome = solve_centralized_makespan(
            instance.tasks, instance.n_mbr, instance.n_dor, instance.speed,
            time_limit_s=args.time_limit, formulation="compact", **kwargs,
        )
        result = dict(instance=instance.instance_id, method="Gurobi",
                      success=outcome.success, objective=outcome.evaluated_objective,
                      plan=outcome.plan, runtime_s=outcome.total_time_s)
    elif args.method in {"ga", "alns", "am"}:
        from scripts.run_centralized import run_method
        if args.method == "am" and args.checkpoint is None:
            parser.error("--checkpoint is required for am")
        with tempfile.TemporaryDirectory(prefix="de_mrst_baseline_") as temp:
            excel = Path(temp) / "instance.xlsx"
            write_instance_excel(instance, excel)
            central_args = SimpleNamespace(
                method=args.method, instance=excel, max_runtime=args.time_limit,
                iterations=None, greedy_cache=None, checkpoint=args.checkpoint,
                device="cpu", model_seed=1103, objective_mode="makespan_distance",
            )
            _, metrics, elapsed, _ = run_method(central_args)
        result = dict(instance=instance.instance_id, method=args.method.upper(),
                      success=metrics.success, objective=metrics.objective,
                      makespan=metrics.makespan, carrier_distance=metrics.mbr_distance,
                      runtime_s=elapsed)
    else:
        outcome = run_decentralized_market_episode(
            args.method, instance.tasks, instance.n_mbr, instance.n_dor,
            instance.speed, env_seed=instance.data_seed, **kwargs,
        )
        result = dict(instance=instance.instance_id, method=args.method,
                      success=outcome.success, objective=outcome.cost,
                      plan=outcome.plan)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
