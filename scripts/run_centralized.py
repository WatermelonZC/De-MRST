#!/usr/bin/env python
"""Run one centralized reference on one instance and emit a canonical row."""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heuristic.ALNS import ALNS
from heuristic.GA import CROSSOVER_RATE, MUTATION_RATE, POP_SIZE, GA
from heuristic.Util.ALNS_config import ALNSConfig
from heuristic.Util.Config import Config
from heuristic.Util.generate_init_solution import generate_solution_greedy
from heuristic.Util.load_data import read_excel
from mrs.c_am import load_c_am_checkpoint
from mrs.core import OBJECTIVE_MODES, TaskSpec, evaluate_centralized_plan
from mrs.greedy_cache import load_greedy_cache, save_greedy_cache
from mrs.repro import RunRecorder
from mrs.results import ResultRecord, write_records_csv
from nets.attention_model import set_decode_type
from utils.data_utils import read_excel_to_tensor


METHOD_NAMES = {
    'greedy': 'C-Greedy',
    'ga': 'C-GA',
    'alns': 'C-ALNS',
    'am': 'C-AM',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHOD_NAMES, required=True)
    parser.add_argument('--instance', type=Path, required=True)
    parser.add_argument('--distribution', default='unknown')
    parser.add_argument('--data-seed', type=int, required=True)
    parser.add_argument('--model-seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--greedy-cache', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--objective-mode', choices=OBJECTIVE_MODES, default='v4',
        help='central search objective; default is the frozen v4 J',
    )
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument('--iterations', type=int)
    stop.add_argument('--max-runtime', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations is None and args.max_runtime is None:
        args.max_runtime = 30
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_method(args):
    Config.objective_mode = getattr(args, 'objective_mode', 'v4')
    if Config.objective_mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"objective_mode must be one of {OBJECTIVE_MODES}"
        )
    seed_everything(args.model_seed)
    checkpoint = None
    if args.method != 'am':
        instance = read_excel(args.instance)
        if args.greedy_cache is not None and args.greedy_cache.exists():
            cached = load_greedy_cache(
                args.greedy_cache, args.instance, instance
            )
            initial_solution = cached.solution
            greedy_solve_time = cached.solve_time_s
        else:
            greedy_start = time.perf_counter()
            initial_solution = generate_solution_greedy(instance)
            greedy_solve_time = time.perf_counter() - greedy_start
            if args.greedy_cache is not None:
                save_greedy_cache(
                    args.greedy_cache, args.instance,
                    initial_solution, greedy_solve_time,
                )

    if args.method == 'greedy':
        solution = initial_solution
        elapsed = greedy_solve_time
    elif args.method == 'ga':
        search_start = time.perf_counter()
        solution, _ = GA(
            str(args.instance), max_runtime=args.max_runtime,
            count=None if args.max_runtime is not None else args.iterations,
            initial_solution=initial_solution,
        )
        elapsed = greedy_solve_time + time.perf_counter() - search_start
    elif args.method == 'alns':
        search_start = time.perf_counter()
        _, solution, _, _ = ALNS(
            str(args.instance), max_runtime=args.max_runtime,
            count=None if args.max_runtime is not None else args.iterations,
            initial_solution=initial_solution,
        )
        elapsed = greedy_solve_time + time.perf_counter() - search_start
    else:
        if args.checkpoint is None:
            raise ValueError('--checkpoint is required for C-AM')
        instance = read_excel(args.instance)
        device = torch.device(args.device)
        model, config, _ = load_c_am_checkpoint(
            args.checkpoint,
            device,
            fleet_override=(instance.mbr_count, instance.dor_count),
        )
        data = read_excel_to_tensor(args.instance)
        data = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
        set_decode_type(model, 'greedy')
        start = time.perf_counter()
        with torch.no_grad():
            _, _, actions = model(data, return_pi=True)
        task_specs = []
        for row in instance:
            pickup_time = instance.pickup_time
            if len(row) > 7 and not np.isnan(float(row[7])):
                pickup_time = float(row[7])
            task_specs.append(TaskSpec(
                (float(row[1]), float(row[2])),
                (float(row[3]), float(row[4])),
                float(row[5]), pickup_time, float(row[6]),
            ))
        assignments = [
            (
                int(action[0]),
                int(action[2]) - instance.dor_count,
                int(action[1]),
            )
            for action in actions[0].detach().cpu().tolist()
        ]
        metrics = evaluate_centralized_plan(
            task_specs, assignments, instance.mbr_count, instance.dor_count,
            instance.speed, instance.dock_time, instance.detach_time,
            mbr_speed=getattr(instance, 'mbr_speed', instance.speed),
            dor_speed=getattr(instance, 'dor_speed', instance.speed),
            mbr_initial_positions=getattr(instance, 'mbr_initial_positions', None) or None,
            dor_initial_positions=getattr(instance, 'dor_initial_positions', None) or None,
            objective_mode=Config.objective_mode,
        )
        elapsed = time.perf_counter() - start
        checkpoint = str(args.checkpoint)
        return instance, metrics, elapsed, checkpoint
    solution.get_fitness()
    if not solution.feasible or solution.metrics is None:
        raise RuntimeError(f'{METHOD_NAMES[args.method]} produced an infeasible plan')
    return instance, solution.metrics, elapsed, checkpoint


def main():
    args = parse_args()
    config = {
        'method': args.method,
        'instance': str(args.instance),
        'distribution': args.distribution,
        'iterations': args.iterations,
        'max_runtime': args.max_runtime,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'greedy_cache': str(args.greedy_cache) if args.greedy_cache else None,
        'device': args.device,
        'objective_mode': args.objective_mode,
    }
    if args.method == 'ga':
        config.update({
            'population_size': POP_SIZE,
            'crossover_rate': CROSSOVER_RATE,
            'mutation_rate': MUTATION_RATE,
        })
    elif args.method == 'alns':
        config.update({
            name: value
            for name, value in vars(ALNSConfig).items()
            if not name.startswith('_') and not callable(value)
        })
    recorder = RunRecorder(
        args.run_dir,
        METHOD_NAMES[args.method],
        config,
        args.data_seed,
        args.model_seed,
        project_dir=PROJECT_ROOT,
    )
    try:
        instance, metrics, elapsed, checkpoint = run_method(args)
        record = ResultRecord.from_metrics(
            METHOD_NAMES[args.method],
            args.instance.stem,
            args.distribution,
            instance.mbr_count,
            instance.dor_count,
            args.data_seed,
            args.model_seed,
            metrics,
            elapsed,
            checkpoint=checkpoint,
        )
        write_records_csv([record], args.output)
        (args.run_dir / 'result.json').write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True), encoding='utf-8'
        )
        recorder.finish('completed')
        print(json.dumps(asdict(record), sort_keys=True))
    except Exception:
        recorder.finish('failed')
        raise


if __name__ == '__main__':
    main()
