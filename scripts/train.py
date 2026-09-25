#!/usr/bin/env python
"""Tensorized formal trainer for the single-setting decentralized protocol."""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrs.instances import generate_instance, generation_kwargs_from_setting
from mrs.observations import (
    validate_protocol_observation_schema, validate_model_observation_protocol,
    validate_protocol_ablation,
)
from mrs.protocol import validate_protocol_source_files
from mrs.formal_medp import FormalMEDPPolicy
from mrs.policies import METHOD_LABELS, FORMAL_MEDP_METHODS, build_model, save_policy_checkpoint
from mrs.formal_medp import MEDP_ABLATIONS
from mrs.formal_medp_checkpoint import (
    load_formal_medp_checkpoint,
    save_formal_medp_checkpoint,
)
from mrs.protocol import (
    load_frozen_protocol,
    protocol_objective_mode,
    select_training_setting,
)
from mrs.repro import RunRecorder
from mrs.seed_stream import training_seed_count, training_seed_slice
from mrs.tensorized_rollout import (
    TensorizedDecisionBuffer,
    concatenate_decision_buffers,
    run_tensorized_batch,
    sample_decisions_per_episode,
    slice_decision_buffer,
    tensorized_reinforce_loss,
)


LEGACY_AVG_COST_LOG_INTERVAL = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--setting-id")
    parser.add_argument(
        "--method", choices=tuple(METHOD_LABELS), required=True
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--medp-ablation",
        choices=MEDP_ABLATIONS,
        default="full",
        help="MEDP pair-mechanism ablation; only applies to formal MEDP variants",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def create_tensorboard_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is required for formal decentralized training; install "
            "the tensorboard package in the active environment"
        ) from error
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def make_instances(seeds, setting):
    generation_kwargs = generation_kwargs_from_setting(setting)
    return [
        generate_instance(
            setting["task_count"],
            setting["n_mbr"],
            setting["n_dor"],
            setting["distribution"],
            seed,
            **generation_kwargs,
        )
        for seed in seeds
    ]


def rollout_chunk_plan(instance_count, rollout_batch_size):
    """Return exact `(start, size)` chunks without padding the final rollout."""
    instance_count = int(instance_count)
    rollout_batch_size = int(rollout_batch_size)
    if instance_count < 1 or rollout_batch_size < 1:
        raise ValueError("rollout chunk sizes must be positive")
    return tuple(
        (
            start,
            min(rollout_batch_size, instance_count - start),
        )
        for start in range(0, instance_count, rollout_batch_size)
    )


def save_best_checkpoint(
    method,
    path,
    model,
    optimizer,
    training_config,
    data_seed,
    model_seed,
    epoch,
    validation,
    started_at,
):
    kwargs = dict(
        training_config=training_config,
        data_seed=data_seed,
        model_seed=model_seed,
        step=epoch,
        validation_objective=validation["mean_objective"],
        validation_success_rate=validation["success_rate"],
        started_at=started_at,
        project_dir=PROJECT_ROOT,
    )
    save_policy_checkpoint(method, path, model, optimizer, **kwargs)


def train_replay_batch(
    model,
    optimizer,
    decisions,
    microbatch_size,
    max_gradient_norm,
):
    total = decisions.decision_count
    if total < 1:
        raise ValueError("empty tensorized replay batch")
    optimizer.zero_grad(set_to_none=True)
    policy_loss = 0.0
    entropy = 0.0
    negative_log_likelihood = 0.0
    for start in range(0, total, microbatch_size):
        end = min(start + microbatch_size, total)
        result = tensorized_reinforce_loss(model, decisions, start, end)
        weight = result.decision_count / total
        (result.policy_loss * weight).backward()
        policy_loss += float(result.policy_loss.detach()) * weight
        entropy += float(result.entropy.detach()) * weight
        negative_log_likelihood += (
            float(result.negative_log_likelihood.detach()) * weight
        )
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise FloatingPointError("tensorized training produced non-finite gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_gradient_norm
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("tensorized training produced non-finite gradient norm")
    optimizer.step()
    advantages = decisions.cost_advantages
    return {
        "policy_loss": policy_loss,
        "entropy": entropy,
        "negative_log_likelihood": negative_log_likelihood,
        "gradient_norm": float(gradient_norm),
        "clipped_gradient_norm": min(float(gradient_norm), max_gradient_norm),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std(unbiased=False)),
        "advantage_abs_mean": float(advantages.abs().mean()),
        "source_episode_count": int(decisions.episode_index.unique().numel()),
    }


def summarize_updates(updates):
    if not updates:
        raise RuntimeError("an epoch completed without an optimizer update")
    mean_fields = (
        "policy_loss",
        "entropy",
        "negative_log_likelihood",
        "gradient_norm",
        "clipped_gradient_norm",
        "advantage_mean",
        "advantage_std",
        "advantage_abs_mean",
        "source_episode_count",
    )
    return {
        "update_count": len(updates),
        **{
            f"mean_{field}": sum(row[field] for row in updates) / len(updates)
            for field in mean_fields
        },
        "maximum_gradient_norm": max(row["gradient_norm"] for row in updates),
        "minimum_source_episode_count": min(
            row["source_episode_count"] for row in updates
        ),
    }


def log_legacy_training_step(writer, optimizer_step, avg_cost, diagnostics):
    """Match the original trainer's TensorBoard scalar contract.

    The value is already produced by the required sampled training forward
    pass; logging it requires no additional rollout, validation, or replay.
    """
    if optimizer_step % LEGACY_AVG_COST_LOG_INTERVAL == 0:
        writer.add_scalar("avg_cost", avg_cost, optimizer_step)
        writer.add_scalar("actor_loss", diagnostics["policy_loss"], optimizer_step)
        writer.add_scalar("nll", diagnostics["negative_log_likelihood"], optimizer_step)
        writer.add_scalar("grad_norm", diagnostics["gradient_norm"], optimizer_step)
        writer.add_scalar(
            "grad_norm_clipped", diagnostics["clipped_gradient_norm"], optimizer_step
        )


def evaluate(
    model,
    seeds,
    setting,
    batch_size,
    model_seed,
    max_open_tasks,
    objective_mode,
):
    started = time.perf_counter()
    costs = []
    total_tardiness = []
    system_distance = []
    makespan = []
    on_time_rate = []
    successes = 0
    successful_instances = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(seeds), batch_size):
                instances = make_instances(seeds[start : start + batch_size], setting)
                batch = run_tensorized_batch(
                    model,
                    instances,
                    pomo_size=1,
                    sample=False,
                    training_cfm=False,
                    env_seed=model_seed + 8_405_010 + start,
                    action_seed=model_seed,
                    max_open_tasks=max_open_tasks,
                    record_decisions=False,
                    objective_mode=objective_mode,
                )
                costs.append(batch.costs.detach().cpu())
                success = batch.success.detach().cpu()
                successes += int(success.sum().item())
                if success.any():
                    total_tardiness.append(batch.total_tardiness.detach().cpu()[success])
                    system_distance.append(batch.system_distance.detach().cpu()[success])
                    makespan.append(batch.makespan.detach().cpu()[success])
                    on_time_rate.append(
                        (
                            batch.on_time_task_count.detach().cpu()[success]
                            / float(setting["task_count"])
                        )
                    )
                    successful_instances += int(success.sum().item())
    finally:
        model.train(was_training)
    values = torch.cat(costs)
    validation = {
        "mean_objective": float(values.mean()),
        "success_rate": successes / len(seeds),
        "instance_count": len(seeds),
        "elapsed_s": time.perf_counter() - started,
    }
    if successful_instances:
        validation.update({
            "successful_instance_count": successful_instances,
            "mean_total_tardiness": float(torch.cat(total_tardiness).mean()),
            "mean_average_tardiness": float(torch.cat(total_tardiness).mean())
            / float(setting["task_count"]),
            "mean_on_time_rate": float(torch.cat(on_time_rate).mean()),
            "mean_system_distance": float(torch.cat(system_distance).mean()),
            "mean_makespan": float(torch.cat(makespan).mean()),
        })
    else:
        validation.update({
            "successful_instance_count": 0,
            "mean_total_tardiness": None,
            "mean_average_tardiness": None,
            "mean_on_time_rate": None,
            "mean_system_distance": None,
            "mean_makespan": None,
        })
    return validation


def model_metadata(method, model):
    """Return architecture metadata for methods that freeze it explicitly."""
    if method not in FORMAL_MEDP_METHODS:
        return None
    if not isinstance(model, FormalMEDPPolicy):
        raise TypeError("formal MEDP metadata requires FormalMEDPPolicy")
    return {
        "method_id": method,
        "method_label": METHOD_LABELS[method],
        "policy_config": model.medp_config.to_dict(),
        "task_observation_schema": model.task_observation_schema,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "requires_lookahead": bool(model.requires_lookahead),
    }


def main():
    args = parse_args()
    frozen = load_frozen_protocol(PROJECT_ROOT, args.protocol_id)
    validate_protocol_ablation(frozen.protocol, args.method, args.medp_ablation)
    validate_protocol_observation_schema(frozen.protocol, method=args.method)
    validate_protocol_source_files(PROJECT_ROOT, frozen.protocol)
    protocol = frozen.protocol["decentralized_training"]
    allocator_config = protocol.get("cuda_allocator_config")
    if allocator_config is not None:
        allocator_config = str(allocator_config)
        configured = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if configured is not None and configured != allocator_config:
            raise RuntimeError(
                "PYTORCH_CUDA_ALLOC_CONF differs from the frozen protocol: "
                f"expected {allocator_config!r}, got {configured!r}"
            )
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config
    objective_mode = protocol_objective_mode(frozen)
    if protocol.get("rollout_engine") != "tensorized_gpu_v1":
        raise RuntimeError("protocol does not authorize the tensorized trainer")
    if args.model_seed not in protocol["model_seeds"]:
        raise ValueError("model seed is not frozen in the protocol")
    if args.method not in FORMAL_MEDP_METHODS and args.medp_ablation != "full":
        raise ValueError("--medp-ablation only applies to formal MEDP variants")
    authorized_methods = protocol.get("authorized_training_methods")
    if authorized_methods and args.method not in authorized_methods:
        raise ValueError("method is not authorized by the frozen protocol")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal tensorized trainer requires CUDA")
    setting_with_id = select_training_setting(frozen, args.setting_id)
    setting_id = setting_with_id["setting_id"]
    setting = {
        key: value for key, value in setting_with_id.items() if key != "setting_id"
    }
    allowed_ids = {item["setting_id"] for item in protocol["training_settings"]}
    if setting_id not in allowed_ids:
        raise RuntimeError("training setting is not authorized by the protocol")

    epochs = int(protocol["epochs"])
    instances_per_epoch = int(protocol["unique_training_instances_per_epoch"])
    rollout_batch_size = int(protocol["rollout_unique_batch_size"])
    pomo_size = int(protocol["pomo_repeats_per_instance"])
    samples_per_episode = int(protocol["sampled_decisions_per_episode"])
    decision_sampling_runtime = protocol.get(
        "decision_sampling_runtime", "posthoc_full_trajectory_v1"
    )
    if decision_sampling_runtime not in {
        "posthoc_full_trajectory_v1",
        "online_reservoir_v1",
    }:
        raise RuntimeError("unsupported decision sampling runtime")
    online_decision_sampling = decision_sampling_runtime == "online_reservoir_v1"
    decision_batch_size = int(protocol["decision_batch_size"])
    microbatch_size = int(protocol["replay_microbatch_size"])
    float32_matmul_precision = protocol.get(
        "float32_matmul_precision", "highest"
    )
    if float32_matmul_precision not in {"highest", "high", "medium"}:
        raise RuntimeError("unsupported float32 matmul precision")
    torch.set_float32_matmul_precision(float32_matmul_precision)
    validation_seeds = list(
        frozen.training_validation_seeds["validation_instance_seeds"]
    )
    replay_decisions_per_epoch = (
        instances_per_epoch * pomo_size * samples_per_episode
    )
    if replay_decisions_per_epoch != int(protocol["replay_decisions_per_epoch"]):
        raise RuntimeError("protocol replay decision budget is inconsistent")
    if replay_decisions_per_epoch % decision_batch_size:
        raise RuntimeError("formal epoch must contain complete optimizer batches")
    required_seeds = epochs * instances_per_epoch
    if required_seeds > training_seed_count(frozen.training_validation_seeds):
        raise RuntimeError("formal training exceeds the frozen seed stream")

    if args.smoke:
        if "smoke" not in str(args.output_dir).lower():
            raise ValueError("smoke output must be under a smoke path")
        epochs = 1
        rollout_batch_size = min(rollout_batch_size, 1_024)
        retained_per_rollout = (
            rollout_batch_size * pomo_size * samples_per_episode
        )
        instances_per_epoch = rollout_batch_size * math.ceil(
            decision_batch_size / retained_per_rollout
        )
        validation_seeds = validation_seeds[:64]
        replay_decisions_per_epoch = (
            instances_per_epoch * pomo_size * samples_per_episode
        )

    data_seed = int(frozen.training_validation_seeds["training_data_root_seed"])
    training_config = {
        "protocol_id": frozen.protocol_id,
        "protocol_sha256": frozen.canonical_sha256,
        "protocol_mode": "development_smoke" if args.smoke else "formal",
        "method": args.method,
        "model_seed": args.model_seed,
        "data_seed": data_seed,
        "setting_id": setting_id,
        "setting": setting,
        "epochs": epochs,
        "unique_training_instances_per_epoch": instances_per_epoch,
        "pomo_size": pomo_size,
        "complete_episodes_per_epoch": instances_per_epoch * pomo_size,
        "executed_decisions_per_episode": int(
            protocol["executed_decisions_per_episode"]
        ),
        "sampled_decisions_per_episode": samples_per_episode,
        "decision_sampling_runtime": decision_sampling_runtime,
        "replay_decisions_per_epoch": replay_decisions_per_epoch,
        "effective_decision_batch_size": decision_batch_size,
        "replay_microbatch_size": microbatch_size,
        "float32_matmul_precision": float32_matmul_precision,
        "rollout_unique_batch_size": rollout_batch_size,
        "learning_rate": protocol["learning_rate"],
        "max_gradient_norm": protocol["max_gradient_norm"],
        "max_open_tasks": protocol["max_open_tasks_training"],
        "objective_mode": objective_mode,
        "objective": frozen.protocol["objective"],
        "validation_instances": len(validation_seeds),
        "checkpoint_selection": protocol["checkpoint_selection"],
        "torch_compile": False,
        "cuda_allocator_config": allocator_config,
    }
    if args.method in FORMAL_MEDP_METHODS:
        training_config["medp_ablation"] = args.medp_ablation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    state_path = args.output_dir / "trainer_state.pt"
    progress_path = args.output_dir / "progress.json"
    history_path = args.output_dir / "validation_history.json"
    random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    frozen_forward_chunk = None
    frozen_decoder_glimpses = 2
    frozen_query_fusion_contexts = 4
    if args.method in FORMAL_MEDP_METHODS:
        frozen_forward_chunk = protocol.get("medp_policy_forward_chunk_size")
        if frozen_forward_chunk is None:
            raise RuntimeError(
                "MEDP protocol must freeze its policy forward chunk size"
            )
        frozen_decoder_glimpses = frozen.protocol.get(
            "architecture_freeze", {}
        ).get("medp_decoder_glimpses", 2)
        frozen_query_fusion_contexts = frozen.protocol.get(
            "architecture_freeze", {}
        ).get("medp_query_fusion_contexts", 4)
    model = build_model(
        args.method,
        device,
        args.medp_ablation,
        medp_forward_chunk_size=frozen_forward_chunk,
        medp_decoder_glimpses=frozen_decoder_glimpses,
        medp_query_fusion_contexts=frozen_query_fusion_contexts,
        n_mbr=int(setting["n_mbr"]),
        n_dor=int(setting["n_dor"]),
    )
    validate_model_observation_protocol(frozen.protocol, args.method, model)
    if args.method in FORMAL_MEDP_METHODS:
        if int(frozen_forward_chunk) != model.medp_config.forward_chunk_size:
            raise RuntimeError(
                "MEDP policy forward chunk differs from the frozen protocol"
            )
    active_model_metadata = model_metadata(args.method, model)
    if active_model_metadata is not None:
        training_config["model_metadata"] = active_model_metadata
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])

    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["training_config"] != training_config:
            raise RuntimeError("resume configuration differs from trainer state")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(device)
        start_epoch = int(state["next_epoch"])
        optimizer_step = int(state["optimizer_step"])
        history = state["history"]
        best_key = tuple(state["best_key"]) if state["best_key"] else None
        total_episode_successes = int(state["total_episode_successes"])
        recorder = RunRecorder.resume(args.output_dir)
    else:
        if state_path.exists() or (args.output_dir / "run_metadata.json").exists():
            raise FileExistsError("run directory already exists; use --resume")
        start_epoch = 0
        optimizer_step = 0
        history = []
        best_key = None
        total_episode_successes = 0
        recorder = RunRecorder(
            args.output_dir,
            METHOD_LABELS[args.method],
            training_config,
            data_seed=data_seed,
            model_seed=args.model_seed,
            project_dir=PROJECT_ROOT,
        )

    run_started = time.perf_counter()
    writer = None
    try:
        writer = create_tensorboard_writer(args.output_dir)
        for epoch in range(start_epoch, epochs):
            writer.add_scalar(
                "learnrate_pg0", optimizer.param_groups[0]["lr"], optimizer_step
            )
            epoch_started = time.perf_counter()
            epoch_seed_start = epoch * int(
                protocol["unique_training_instances_per_epoch"]
            )
            pending = None
            epoch_updates = []
            epoch_successes = 0
            rollout_plan = rollout_chunk_plan(
                instances_per_epoch, rollout_batch_size
            )
            rollout_count = len(rollout_plan)
            progress_interval = max(1, min(25, math.ceil(rollout_count / 10)))
            for rollout_index, (
                local_start,
                current_rollout_size,
            ) in enumerate(rollout_plan):
                global_start = epoch_seed_start + local_start
                seeds = training_seed_slice(
                    frozen.training_validation_seeds,
                    global_start,
                    global_start + current_rollout_size,
                )
                instances = make_instances(seeds, setting)
                seed_base = args.model_seed * 1_000_003 + global_start
                rollout = run_tensorized_batch(
                    model,
                    instances,
                    pomo_size=pomo_size,
                    sample=True,
                    training_cfm=True,
                    env_seed=seed_base + 17,
                    action_seed=seed_base + 29,
                    max_open_tasks=protocol["max_open_tasks_training"],
                    record_decisions=True,
                    decision_samples_per_episode=(
                        samples_per_episode if online_decision_sampling else None
                    ),
                    decision_sample_seed=seed_base + 43,
                    objective_mode=objective_mode,
                )
                if rollout.decisions is None:
                    raise RuntimeError("training rollout did not record decisions")
                training_avg_cost = float(rollout.mean_costs.mean())
                epoch_successes += int(rollout.success.sum().item())
                retained = (
                    rollout.decisions
                    if online_decision_sampling
                    else sample_decisions_per_episode(
                        rollout.decisions,
                        episode_count=rollout.episode_count,
                        samples_per_episode=samples_per_episode,
                        seed=seed_base + 43,
                    )
                )
                retained = replace(
                    retained,
                    episode_index=retained.episode_index + global_start * pomo_size,
                )
                pending = (
                    retained
                    if pending is None
                    else concatenate_decision_buffers((pending, retained))
                )
                del rollout, retained, instances

                while pending.decision_count >= decision_batch_size:
                    replay = slice_decision_buffer(
                        pending, 0, decision_batch_size
                    )
                    remainder = slice_decision_buffer(
                        pending, decision_batch_size
                    )
                    diagnostics = train_replay_batch(
                        model,
                        optimizer,
                        replay,
                        microbatch_size,
                        protocol["max_gradient_norm"],
                    )
                    optimizer_step += 1
                    epoch_updates.append(diagnostics)
                    log_legacy_training_step(
                        writer, optimizer_step, training_avg_cost, diagnostics
                    )
                    pending = remainder if remainder.decision_count else None
                    del replay, remainder
                    if pending is None:
                        break

                completed_instances = local_start + current_rollout_size
                if (
                    (rollout_index + 1) % progress_interval == 0
                    or rollout_index + 1 == rollout_count
                ):
                    progress = {
                        "status": "training",
                        "method": METHOD_LABELS[args.method],
                        "model_seed": args.model_seed,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "completed_unique_instances_in_epoch": completed_instances,
                        "unique_instances_per_epoch": instances_per_epoch,
                        "complete_episodes_in_epoch": completed_instances * pomo_size,
                        "optimizer_step": optimizer_step,
                        "pending_decisions": (
                            pending.decision_count if pending is not None else 0
                        ),
                        "elapsed_s": time.perf_counter() - run_started,
                    }
                    atomic_json(progress_path, progress)
                    print(json.dumps(progress, sort_keys=True), flush=True)

            pending_count = pending.decision_count if pending is not None else 0
            if not args.smoke and pending_count:
                raise RuntimeError("formal epoch ended with a partial decision batch")
            expected_updates = replay_decisions_per_epoch // decision_batch_size
            if len(epoch_updates) != expected_updates:
                raise RuntimeError("epoch optimizer update count differs from protocol")
            total_episode_successes += epoch_successes
            validation = evaluate(
                model,
                validation_seeds,
                setting,
                rollout_batch_size,
                args.model_seed,
                protocol["max_open_tasks_training"],
                objective_mode,
            )
            row = {
                "epoch": epoch + 1,
                "unique_training_instances": (epoch + 1) * instances_per_epoch,
                "complete_training_episodes": (
                    (epoch + 1) * instances_per_epoch * pomo_size
                ),
                "optimizer_step": optimizer_step,
                "dropped_partial_decisions": pending_count,
                "training_episode_success_rate": epoch_successes
                / (instances_per_epoch * pomo_size),
                "training": summarize_updates(epoch_updates),
                "validation": validation,
                "elapsed_s": time.perf_counter() - epoch_started,
            }
            history.append(row)
            atomic_json(history_path, history)
            writer.add_scalar("training/policy_loss", row["training"]["mean_policy_loss"], epoch + 1)
            writer.add_scalar("training/entropy", row["training"]["mean_entropy"], epoch + 1)
            writer.add_scalar(
                "training/gradient_norm",
                row["training"]["mean_gradient_norm"],
                epoch + 1,
            )
            writer.add_scalar(
                "training/episode_success_rate",
                row["training_episode_success_rate"],
                epoch + 1,
            )
            writer.add_scalar(
                "validation/mean_objective",
                validation["mean_objective"],
                epoch + 1,
            )
            writer.add_scalar(
                "validation/success_rate",
                validation["success_rate"],
                epoch + 1,
            )
            for field in (
                "mean_total_tardiness",
                "mean_average_tardiness",
                "mean_on_time_rate",
                "mean_system_distance",
                "mean_makespan",
            ):
                value = validation.get(field)
                if value is not None:
                    writer.add_scalar(
                        f"validation/{field}",
                        value,
                        epoch + 1,
                    )
            writer.add_scalar("runtime/epoch_seconds", row["elapsed_s"], epoch + 1)
            writer.flush()
            key = (
                validation["mean_objective"],
                -validation["success_rate"],
                epoch + 1,
            )
            if best_key is None or key < best_key:
                best_key = key
                save_best_checkpoint(
                    args.method,
                    best_path,
                    model,
                    optimizer,
                    training_config,
                    data_seed,
                    args.model_seed,
                    epoch + 1,
                    validation,
                    recorder.metadata.started_at,
                )
            state = {
                "training_config": training_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "history": history,
                "best_key": best_key,
                "total_episode_successes": total_episode_successes,
            }
            atomic_torch_save(state_path, state)
            atomic_json(progress_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if not best_path.exists():
            raise RuntimeError("formal training produced no validation checkpoint")
        if args.method in FORMAL_MEDP_METHODS:
            audited_model, audited_payload = load_formal_medp_checkpoint(
                best_path, torch.device("cpu")
            )
            if audited_payload["training_config"] != training_config:
                raise RuntimeError(
                    "MEDP best checkpoint metadata differs from the run"
                )
            del audited_model, audited_payload
        best_row = min(
            history,
            key=lambda item: (
                item["validation"]["mean_objective"],
                -item["validation"]["success_rate"],
                item["epoch"],
            ),
        )
        total_episodes = epochs * instances_per_epoch * pomo_size
        summary = {
            "status": "completed",
            "method": METHOD_LABELS[args.method],
            "model_seed": args.model_seed,
            "protocol_sha256": frozen.canonical_sha256,
            "epochs": epochs,
            "unique_training_instances": epochs * instances_per_epoch,
            "complete_training_episodes": total_episodes,
            "replayed_decisions": epochs * replay_decisions_per_epoch,
            "optimizer_steps": optimizer_step,
            "training_episode_success_rate": total_episode_successes
            / total_episodes,
            "best_validation": best_row,
            "validation_history": history,
        }
        if active_model_metadata is not None:
            summary["model_metadata"] = active_model_metadata
        atomic_json(args.output_dir / "summary.json", summary)
        atomic_json(progress_path, summary)
        recorder.finish("completed", best_path)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except Exception:
        recorder.finish("failed")
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
