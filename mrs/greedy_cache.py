"""Structured cache for the deterministic centralized greedy construction."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from heuristic.Util.Solution import Solution


@dataclass(frozen=True)
class CachedGreedy:
    solution: Solution
    solve_time_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_greedy_cache(
    path: Path,
    instance_path: Path,
    solution: Solution,
    solve_time_s: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'instance_sha256': _sha256(instance_path),
        'solve_time_s': float(solve_time_s),
        'objective': float(solution.get_fitness()),
        'sequence_map': solution.get_sequence_map(),
        'path_init_task_map': solution.get_path_init_task_map(),
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    os.replace(temporary, path)


def load_greedy_cache(path: Path, instance_path: Path, instance) -> CachedGreedy:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('unsupported greedy cache schema')
    if payload['instance_sha256'] != _sha256(instance_path):
        raise ValueError('greedy cache does not match the instance file')
    sequence_map = {
        int(task): values for task, values in payload['sequence_map'].items()
    }
    path_init_task_map = {
        int(robot): int(task)
        for robot, task in payload['path_init_task_map'].items()
    }
    solution = Solution(instance, sequence_map, path_init_task_map)
    objective = solution.get_fitness()
    if abs(objective - float(payload['objective'])) > 1e-9:
        raise ValueError('greedy cache objective does not reproduce')
    return CachedGreedy(solution, float(payload['solve_time_s']))
