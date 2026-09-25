#!/usr/bin/env python
"""Formal protocol trainer for the centralized C-AM reference."""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrs.c_am import CAmConfig, build_c_am, save_c_am_checkpoint
from mrs.core import objective_tensor
from mrs.instances import generate_instance, instances_to_tensor_batch
from mrs.protocol import load_frozen_protocol, select_training_setting
from mrs.repro import RunRecorder
from mrs.seed_stream import training_seed_count, training_seed_slice
from mrs.stats import paired_ttest_less
from nets.attention_model import set_decode_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="v1")
    parser.add_argument("--setting-id")
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_instances(seeds, setting):
    return [
        generate_instance(
            setting["task_count"],
            setting["n_mbr"],
            setting["n_dor"],
            setting["distribution"],
            seed,
            speed=setting["speed"],
        )
        for seed in seeds
    ]


def evaluate(model, seeds, setting, device, batch_size):
    started = time.perf_counter()
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(seeds), batch_size):
            instances = make_instances(seeds[start:start + batch_size], setting)
            data = instances_to_tensor_batch(instances, device=device)
            set_decode_type(model, "greedy")
            (distance, tardiness), _ = model(data)
            values.append(objective_tensor(tardiness, distance, setting["task_count"], data["v"]).detach().cpu())
    values = torch.cat(values).tolist()
    return {
        "mean_objective": sum(values) / len(values),
        "success_rate": 1.0,
        "objectives": values,
        "elapsed_s": time.perf_counter() - started,
    }


def main():
    args = parse_args()
    frozen = load_frozen_protocol(PROJECT_ROOT, args.protocol_id)
    c_am_protocol = frozen.protocol["centralized"]["C-AM"]
    seeds_manifest = frozen.training_validation_seeds
    if args.model_seed not in c_am_protocol["model_seeds"]:
        raise ValueError(f"model seed is not in experiment_protocol_{args.protocol_id}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    batch_size = int(c_am_protocol["training_batch_size"])
    if args.batch_size is not None and int(args.batch_size) != batch_size:
        raise ValueError("C-AM batch size differs from the frozen protocol")
    selected_setting = select_training_setting(frozen, args.setting_id)
    setting_id = selected_setting["setting_id"]
    setting = {key: value for key, value in selected_setting.items() if key != "setting_id"}
    allowed_settings = c_am_protocol.get("training_settings")
    if allowed_settings is not None and setting_id not in {
        item["setting_id"] for item in allowed_settings
    }:
        raise RuntimeError("selected setting is not authorized for C-AM training")
    validation_seeds = seeds_manifest["validation_instance_seeds"]
    epochs = c_am_protocol["epochs"]
    instances_per_epoch = c_am_protocol["instances_per_epoch"]
    if args.smoke:
        if "smoke" not in str(args.output_dir).lower():
            raise ValueError("development smoke output must be under a smoke path")
        epochs = 1
        instances_per_epoch = 16 * batch_size
        validation_seeds = validation_seeds[:64]
    if epochs * instances_per_epoch > training_seed_count(seeds_manifest):
        raise ValueError("C-AM training budget exceeds frozen seed stream")
    if instances_per_epoch % batch_size:
        raise ValueError("C-AM epoch size must be divisible by its batch size")

    amp_dtype = torch.float16
    use_grad_scaler = True

    training_config = {
        "protocol_sha256": frozen.canonical_sha256,
        "protocol_mode": "development_smoke" if args.smoke else "formal",
        "method": "C-AM",
        "model_seed": args.model_seed,
        "data_seed": seeds_manifest["training_data_root_seed"],
        "setting": setting,
        "epochs": epochs,
        "instances_per_epoch": instances_per_epoch,
        "training_instances": epochs * instances_per_epoch,
        "batch_size": batch_size,
        "mixed_precision": str(amp_dtype).removeprefix("torch."),
        "amp_overflow_behavior": (
            "skip non-finite optimizer update and reduce GradScaler scale"
        ),
        "learning_rate": c_am_protocol["learning_rate"],
        "max_gradient_norm": c_am_protocol["max_gradient_norm"],
        "baseline": c_am_protocol["reinforce_baseline"],
        "validation_instances": len(validation_seeds),
        "checkpoint_selection": c_am_protocol["checkpoint_selection"],
    }
    if frozen.protocol_id != "v1":
        training_config.update({"protocol_id": frozen.protocol_id, "setting_id": setting_id})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    state_path = args.output_dir / "trainer_state.pt"
    history_path = args.output_dir / "validation_history.json"
    progress_path = args.output_dir / "progress.json"

    random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    config = CAmConfig(setting["n_mbr"], setting["n_dor"])
    model = build_c_am(config).to(device)
    baseline_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c_am_protocol["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["training_config"] != training_config:
            raise RuntimeError("resume configuration differs from trainer state")
        model.load_state_dict(state["model_state_dict"], strict=True)
        baseline_model.load_state_dict(state["baseline_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state.get("grad_scaler_state_dict", {}))
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(device)
        start_epoch = state["next_epoch"]
        history = state["history"]
        best_key = tuple(state["best_key"]) if state["best_key"] is not None else None
        optimizer_step = int(state.get("optimizer_step", start_epoch * (instances_per_epoch // batch_size)))
        skipped_amp_updates = int(state.get("skipped_amp_updates", 0))
        recorder = RunRecorder.resume(args.output_dir)
    else:
        if state_path.exists() or (args.output_dir / "run_metadata.json").exists():
            raise FileExistsError("run directory already contains trainer state; use --resume")
        start_epoch = 0
        history = []
        best_key = None
        optimizer_step = 0
        skipped_amp_updates = 0
        recorder = RunRecorder(
            args.output_dir,
            "C-AM",
            training_config,
            data_seed=seeds_manifest["training_data_root_seed"],
            model_seed=args.model_seed,
            project_dir=PROJECT_ROOT,
        )

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            set_decode_type(model, "sampling")
            losses = []
            gradient_norms = []
            epoch_skipped_amp_updates = 0
            epoch_started = time.perf_counter()
            epoch_seed_start = epoch * instances_per_epoch
            for start in range(0, instances_per_epoch, batch_size):
                batch_seeds = training_seed_slice(
                    seeds_manifest,
                    epoch_seed_start + start,
                    epoch_seed_start + start + batch_size,
                )
                instances = make_instances(batch_seeds, setting)
                data = instances_to_tensor_batch(instances, device=device)
                set_decode_type(model, "sampling")
                with torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=True
                ):
                    (distance, tardiness), likelihood = model(data)
                    cost = objective_tensor(
                        tardiness, distance, setting["task_count"], data["v"]
                    )
                    baseline_model.eval()
                    set_decode_type(baseline_model, "greedy")
                    with torch.no_grad():
                        (base_distance, base_tardiness), _ = baseline_model(data)
                        base_cost = objective_tensor(
                            base_tardiness,
                            base_distance,
                            setting["task_count"],
                            data["v"],
                        )
                loss = (
                    (cost.float() - base_cost.float()).detach()
                    * likelihood.float()
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("C-AM formal loss is not finite")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = [
                    parameter.grad for parameter in model.parameters() if parameter.grad is not None
                ]
                if not gradients:
                    raise FloatingPointError("C-AM formal update produced no gradients")
                gradients_finite = all(
                    torch.isfinite(gradient).all() for gradient in gradients
                )
                scale_before = scaler.get_scale()
                if not gradients_finite:
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= scale_before:
                        raise FloatingPointError(
                            "GradScaler did not reduce scale after non-finite gradients"
                        )
                    skipped_amp_updates += 1
                    epoch_skipped_amp_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), c_am_protocol["max_gradient_norm"]
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError(
                            "C-AM finite gradients produced a non-finite global norm"
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step += 1
                    losses.append(loss.item())
                    gradient_norms.append(float(norm))

                completed_instances = start + batch_size
                completed_batches = completed_instances // batch_size
                if completed_batches % 100 == 0 or completed_instances == instances_per_epoch:
                    progress = {
                        "status": "training",
                        "method": "C-AM",
                        "model_seed": args.model_seed,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "completed_instances_in_epoch": completed_instances,
                        "instances_per_epoch": instances_per_epoch,
                        "optimizer_step": optimizer_step,
                        "skipped_amp_updates": skipped_amp_updates,
                        "amp_scale": scaler.get_scale(),
                        "elapsed_s": time.perf_counter() - epoch_started,
                    }
                    atomic_json(progress_path, progress)
                    print(json.dumps(progress, sort_keys=True), flush=True)

            if not losses:
                raise RuntimeError("C-AM epoch completed no finite optimizer update")
            candidate = evaluate(model, validation_seeds, setting, device, batch_size)
            baseline_values = evaluate(baseline_model, validation_seeds, setting, device, batch_size)
            _, baseline_p = paired_ttest_less(candidate["objectives"], baseline_values["objectives"])
            baseline_updated = False
            if candidate["mean_objective"] < baseline_values["mean_objective"] and baseline_p < 0.05:
                baseline_model.load_state_dict(model.state_dict(), strict=True)
                baseline_updated = True
            row = {
                "epoch": epoch + 1,
                "training_instances": (epoch + 1) * instances_per_epoch,
                "optimizer_step": optimizer_step,
                "mean_policy_loss": sum(losses) / len(losses),
                "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
                "skipped_amp_updates": epoch_skipped_amp_updates,
                "cumulative_skipped_amp_updates": skipped_amp_updates,
                "final_amp_scale": scaler.get_scale(),
                "validation": candidate,
                "baseline_validation_mean": baseline_values["mean_objective"],
                "baseline_p_value": baseline_p,
                "baseline_updated": baseline_updated,
                "elapsed_s": time.perf_counter() - epoch_started,
            }
            history.append(row)
            atomic_json(history_path, history)
            key = (candidate["mean_objective"], epoch + 1)
            if best_key is None or key < best_key:
                best_key = key
                save_c_am_checkpoint(
                    best_path,
                    model,
                    config,
                    optimizer,
                    validation_objective=candidate["mean_objective"],
                    validation_success_rate=candidate["success_rate"],
                    training_config=training_config,
                    data_seed=seeds_manifest["training_data_root_seed"],
                    model_seed=args.model_seed,
                    step=epoch + 1,
                    started_at=recorder.metadata.started_at,
                    project_dir=PROJECT_ROOT,
                )
            trainer_state = {
                "training_config": training_config,
                "model_state_dict": model.state_dict(),
                "baseline_state_dict": baseline_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "grad_scaler_state_dict": scaler.state_dict(),
                "optimizer_step": optimizer_step,
                "skipped_amp_updates": skipped_amp_updates,
                "next_epoch": epoch + 1,
                "history": history,
                "best_key": best_key,
            }
            atomic_torch_save(state_path, trainer_state)
            atomic_json(progress_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if not best_path.exists():
            raise RuntimeError("C-AM formal training did not produce a best checkpoint")
        best_row = min(history, key=lambda item: (item["validation"]["mean_objective"], item["epoch"]))
        summary = {
            "status": "completed",
            "method": "C-AM",
            "model_seed": args.model_seed,
            "protocol_sha256": frozen.canonical_sha256,
            "training_instances": epochs * instances_per_epoch,
            "optimizer_steps": optimizer_step,
            "skipped_amp_updates": skipped_amp_updates,
            "best_validation": best_row,
            "validation_history": history,
        }
        atomic_json(args.output_dir / "summary.json", summary)
        atomic_json(progress_path, summary)
        recorder.finish("completed", best_path)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except Exception:
        recorder.finish("failed")
        raise


if __name__ == "__main__":
    main()
