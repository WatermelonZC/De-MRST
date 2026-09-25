"""Canonical per-instance result schema shared by all six methods."""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Sequence

from .core import PlanMetrics


@dataclass(frozen=True)
class ResultRecord:
    method: str
    instance_id: str
    distribution: str
    task_count: int
    n_mbr: int
    n_dor: int
    data_seed: int
    model_seed: int
    objective: float
    total_tardiness: float
    average_tardiness: float
    on_time_rate: float
    system_distance: float
    mbr_distance: float
    dor_distance: float
    average_mbr_wait: float
    average_dor_wait: float
    makespan: float
    success: bool
    decision_time_ms: float
    episode_solve_time_s: float
    checkpoint: Optional[str] = None
    failure_reason: Optional[str] = None

    @classmethod
    def from_metrics(
        cls,
        method: str,
        instance_id: str,
        distribution: str,
        n_mbr: int,
        n_dor: int,
        data_seed: int,
        model_seed: int,
        metrics: PlanMetrics,
        episode_solve_time_s: float,
        checkpoint: Optional[str] = None,
    ):
        return cls(
            method=method,
            instance_id=instance_id,
            distribution=distribution,
            task_count=len(metrics.completion_times),
            n_mbr=n_mbr,
            n_dor=n_dor,
            data_seed=data_seed,
            model_seed=model_seed,
            objective=metrics.objective,
            total_tardiness=metrics.total_tardiness,
            average_tardiness=metrics.average_tardiness,
            on_time_rate=metrics.on_time_rate,
            system_distance=metrics.system_distance,
            mbr_distance=metrics.mbr_distance,
            dor_distance=metrics.dor_distance,
            average_mbr_wait=metrics.average_mbr_wait,
            average_dor_wait=metrics.average_dor_wait,
            makespan=metrics.makespan,
            success=metrics.success,
            decision_time_ms=1000.0 * episode_solve_time_s / len(metrics.completion_times),
            episode_solve_time_s=episode_solve_time_s,
            checkpoint=checkpoint,
        )

    @classmethod
    def from_decentralized_outcome(
        cls,
        method: str,
        instance_id: str,
        distribution: str,
        n_mbr: int,
        n_dor: int,
        data_seed: int,
        model_seed: int,
        outcome,
        episode_solve_time_s: float,
        checkpoint: Optional[str] = None,
        task_count: Optional[int] = None,
    ):
        decision_time_ms = 1000.0 * outcome.decision_time_s / max(1, len(outcome.records))
        if outcome.metrics is None:
            nan = float('nan')
            return cls(
                method=method,
                instance_id=instance_id,
                distribution=distribution,
                task_count=int(task_count if task_count is not None else len(outcome.plan)),
                n_mbr=n_mbr,
                n_dor=n_dor,
                data_seed=data_seed,
                model_seed=model_seed,
                objective=outcome.cost,
                total_tardiness=nan,
                average_tardiness=nan,
                on_time_rate=nan,
                system_distance=nan,
                mbr_distance=nan,
                dor_distance=nan,
                average_mbr_wait=nan,
                average_dor_wait=nan,
                makespan=nan,
                success=False,
                decision_time_ms=decision_time_ms,
                episode_solve_time_s=episode_solve_time_s,
                checkpoint=checkpoint,
                failure_reason=outcome.failure_reason or 'failed',
            )
        metrics = outcome.metrics
        return cls(
            method=method,
            instance_id=instance_id,
            distribution=distribution,
            task_count=len(metrics.completion_times),
            n_mbr=n_mbr,
            n_dor=n_dor,
            data_seed=data_seed,
            model_seed=model_seed,
            objective=metrics.objective,
            total_tardiness=metrics.total_tardiness,
            average_tardiness=metrics.average_tardiness,
            on_time_rate=metrics.on_time_rate,
            system_distance=metrics.system_distance,
            mbr_distance=metrics.mbr_distance,
            dor_distance=metrics.dor_distance,
            average_mbr_wait=metrics.average_mbr_wait,
            average_dor_wait=metrics.average_dor_wait,
            makespan=metrics.makespan,
            success=True,
            decision_time_ms=decision_time_ms,
            episode_solve_time_s=episode_solve_time_s,
            checkpoint=checkpoint,
            failure_reason=None,
        )


def write_records_csv(records: Sequence[ResultRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(ResultRecord)]
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    temporary.replace(path)


def read_records_csv(path: Path) -> List[ResultRecord]:
    boolean = {'True': True, 'False': False}
    integer_fields = {'task_count', 'n_mbr', 'n_dor', 'data_seed', 'model_seed'}
    float_fields = {
        'objective', 'total_tardiness', 'average_tardiness', 'on_time_rate',
        'system_distance', 'mbr_distance', 'dor_distance', 'average_mbr_wait',
        'average_dor_wait', 'makespan', 'decision_time_ms',
        'episode_solve_time_s',
    }
    records = []
    with Path(path).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            for name in integer_fields:
                row[name] = int(row[name])
            for name in float_fields:
                row[name] = float(row[name])
            row['success'] = boolean[row['success']]
            row['checkpoint'] = row['checkpoint'] or None
            row['failure_reason'] = row['failure_reason'] or None
            records.append(ResultRecord(**row))
    return records
