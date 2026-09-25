"""Run a trained decentralized policy on one paper-format JSON instance."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrs.decentralized_rollout import run_decentralized_episode
from mrs.instances import read_instance
from mrs.policies import load_policy_checkpoint
from mrs.tensorized_rollout import run_tensorized_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--method", choices=("auto", "medp_1r", "d_am", "hetmrta_mrs"), default="auto")
    parser.add_argument("--samples", type=int, default=0,
                        help="0: greedy; positive: return the best of this many sampled rollouts")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="number of sampled rollouts evaluated together")
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    instance = read_instance(args.instance)
    checkpoint = args.checkpoint or ROOT / "checkpoints" / f"n{len(instance.tasks)}" / "best.pt"
    device = torch.device(args.device)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    method_by_name = {
        "De-MRST": "medp_1r", "D-AM": "d_am", "HetMRTA-RL-MRS": "hetmrta_mrs"
    }
    stored_method = method_by_name.get(metadata.get("method"))
    if stored_method is None:
        raise ValueError("checkpoint method is not supported by this predictor")
    method = stored_method if args.method == "auto" else args.method
    if method != stored_method:
        raise ValueError(f"checkpoint contains {stored_method}, not {method}")
    model, payload = load_policy_checkpoint(method, checkpoint, device)
    model.eval()
    config = getattr(model, "medp_config", model.config)
    backbone = getattr(config, "backbone", config)
    if (instance.n_mbr, instance.n_dor) != (backbone.n_mbr, backbone.n_dor):
        raise ValueError("instance fleet size differs from the checkpoint input width")

    if args.samples == 0:
        with torch.inference_mode():
            best = run_decentralized_episode(
                model, instance.tasks, instance.n_mbr, instance.n_dor, instance.speed,
                sample=False, training_cfm=False,
                env_seed=instance.data_seed, action_seed=args.seed,
                dock_time=instance.dock_time, detach_time=instance.detach_time,
                mbr_speed=instance.resolved_mbr_speed,
                dor_speed=instance.resolved_dor_speed,
                mbr_initial_positions=instance.mbr_initial_positions,
                dor_initial_positions=instance.dor_initial_positions,
                objective_mode="makespan_distance",
            )
        success, cost, plan = best.success, best.cost, best.plan
        metrics = (
            {
                "makespan": best.metrics.makespan,
                "carrier_distance": best.metrics.mbr_distance,
                "worker_distance": best.metrics.dor_distance,
            }
            if best.metrics is not None else None
        )
        failure_reason = best.failure_reason
    else:
        winner = None
        for start in range(0, args.samples, args.batch_size):
            size = min(args.batch_size, args.samples - start)
            batch = run_tensorized_batch(
                model, [instance], pomo_size=size, sample=True,
                training_cfm=False, env_seed=instance.data_seed,
                action_seed=args.seed + start, record_decisions=False,
                objective_mode="makespan_distance",
            )
            index = int(batch.costs.argmin().item())
            cost_here = float(batch.costs[index].item())
            if winner is None or cost_here < winner[0]:
                length = int(batch.plan_lengths[index].item())
                winner = (
                    cost_here, bool(batch.success[index].item()),
                    batch.plans[index, :length].detach().cpu().tolist(),
                    float(batch.makespan[index].item()),
                    float(batch.system_distance[index].item()),
                )
        cost, success, plan, makespan, carrier_distance = winner
        metrics = {"makespan": makespan, "carrier_distance": carrier_distance}
        failure_reason = "" if success else "rollout failed"
    result = {
        "instance": instance.instance_id,
        "method": payload["method"],
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload["step"],
        "decoding": "greedy" if args.samples == 0 else f"sample_{args.samples}",
        "success": success,
        "objective": cost,
        "plan": plan,
        "metrics": metrics,
        "failure_reason": failure_reason,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
