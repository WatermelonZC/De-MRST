"""Physical pair forecasts used by the market and MEPO baselines only."""

from typing import Tuple

from .core import TransitionResult, manhattan_distance, marsupial_transition
from .event_env import DecentralizedMRSEnv


def transition_features(
    env: DecentralizedMRSEnv,
    task_id: int,
    mbr_id: int,
    dor_id: int,
) -> Tuple[Tuple[float, ...], TransitionResult]:
    """Return the nine physical-unit features and their authoritative transition."""
    task = env.tasks[task_id]
    result = marsupial_transition(
        env.mbr_state[mbr_id],
        env.dor_state[dor_id],
        task,
        None,
        env.dock_time,
        env.detach_time,
    )
    features = (
        manhattan_distance(env.mbr_state[mbr_id].position, env.dor_state[dor_id].position),
        result.dock_time,
        result.mbr_wait,
        result.dor_wait,
        result.dor_distance,
        result.task_completion_time,
        max(0.0, result.task_completion_time - task.due_time),
        result.mbr_release_time,
        result.dor_release_time,
    )
    return features, result
