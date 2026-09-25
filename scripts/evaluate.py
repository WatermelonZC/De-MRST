"""Evaluate one paper method on every JSON instance in a directory."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("de_mrst", "d_am", "hetmrta_mrs")
BASELINES = ("d_min_min", "d_murdoch", "d_coalition_auction", "ga", "alns", "gurobi", "am")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--method", choices=(*LEARNED, *BASELINES), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.method in {"d_am", "hetmrta_mrs", "am"} and args.checkpoint is None:
        parser.error(f"--checkpoint is required for {args.method}")
    if args.samples and args.method not in LEARNED:
        parser.error("--samples applies only to learned decentralized methods")
    instances = sorted(args.instances.glob("*.json"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        instances = instances[:args.limit]
    if not instances:
        parser.error("no JSON instances found")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    actions_dir = args.output_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance in instances:
        output = actions_dir / instance.name
        if args.method in LEARNED:
            command = [sys.executable, str(ROOT / "scripts" / "predict.py"),
                       "--instance", str(instance), "--device", args.device,
                       "--samples", str(args.samples), "--batch-size", str(args.batch_size),
                       "--output", str(output)]
            if args.checkpoint:
                command.extend(("--checkpoint", str(args.checkpoint)))
            if args.method != "de_mrst":
                command.extend(("--method", args.method))
        else:
            command = [sys.executable, str(ROOT / "scripts" / "baselines.py"),
                       "--instance", str(instance), "--method", args.method,
                       "--time-limit", str(args.time_limit), "--output", str(output)]
            if args.checkpoint:
                command.extend(("--checkpoint", str(args.checkpoint)))
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{instance.name}: {completed.stderr.strip()}")
        row = json.loads(output.read_text(encoding="utf-8"))
        rows.append(row)
        print(f"{instance.name}: success={int(row['success'])} objective={row['objective']:.6f}", flush=True)

    summary = {
        "method": args.method,
        "instance_count": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "mean_objective": statistics.mean(row["objective"] for row in rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
