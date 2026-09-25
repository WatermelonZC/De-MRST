"""Tensorized decentralized rollouts for fixed-size training batches.

The scalar event environment remains the correctness reference. This module
keeps a batch of independent episodes on one torch device so policy inference,
observation construction, and event advancement avoid the per-decision
host/device synchronization of the reference implementation. Learned policies
consume only the common communicated task and robot observations.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .core import (
    OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT,
    OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT,
    normalize_objective_mode,
    protocol_episode_timeout,
    protocol_failure_penalty,
)
from .instances import MarsupialInstance
from .observations import (
    AGENT_FEATURE_DIM,
    insert_owner_one_hot, task_feature_dim,
)


EMPTY = 0
OPEN = 1
IN_PROGRESS = 2
COMPLETED = 3
READY_PHASE = 0
CFM_PHASE = 1
ADVANCE_PHASE = 2


@dataclass(frozen=True)
class TensorizedDecisionBuffer:
    task_features: torch.Tensor
    agent_features: torch.Tensor
    action_mask: torch.Tensor
    agent_index: torch.Tensor
    actions: torch.Tensor
    episode_index: torch.Tensor
    cost_advantages: torch.Tensor

    @property
    def decision_count(self) -> int:
        return int(self.actions.numel())


def _index_decision_buffer(
    decisions: TensorizedDecisionBuffer,
    indices: torch.Tensor,
) -> TensorizedDecisionBuffer:
    return TensorizedDecisionBuffer(
        task_features=decisions.task_features[indices],
        agent_features=decisions.agent_features[indices],
        action_mask=decisions.action_mask[indices],
        agent_index=decisions.agent_index[indices],
        actions=decisions.actions[indices],
        episode_index=decisions.episode_index[indices],
        cost_advantages=decisions.cost_advantages[indices],
    )


def concatenate_decision_buffers(
    buffers: Sequence[TensorizedDecisionBuffer],
) -> TensorizedDecisionBuffer:
    """Concatenate compatible replay buffers without moving them off device."""
    buffers = tuple(buffers)
    if not buffers:
        raise ValueError("at least one decision buffer is required")
    return TensorizedDecisionBuffer(
        task_features=torch.cat([item.task_features for item in buffers]),
        agent_features=torch.cat([item.agent_features for item in buffers]),
        action_mask=torch.cat([item.action_mask for item in buffers]),
        agent_index=torch.cat([item.agent_index for item in buffers]),
        actions=torch.cat([item.actions for item in buffers]),
        episode_index=torch.cat([item.episode_index for item in buffers]),
        cost_advantages=torch.cat([item.cost_advantages for item in buffers]),
    )


def slice_decision_buffer(
    decisions: TensorizedDecisionBuffer,
    start: int,
    end: Optional[int] = None,
) -> TensorizedDecisionBuffer:
    end = decisions.decision_count if end is None else int(end)
    start = int(start)
    if not 0 <= start <= end <= decisions.decision_count:
        raise ValueError("invalid decision buffer slice")
    indices = torch.arange(start, end, device=decisions.actions.device)
    return _index_decision_buffer(decisions, indices)


def sample_decisions_per_episode(
    decisions: TensorizedDecisionBuffer,
    *,
    episode_count: int,
    samples_per_episode: int = 1,
    seed: int = 0,
) -> TensorizedDecisionBuffer:
    """Uniformly retain a fixed number of decisions from every full episode."""
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be positive")
    if decisions.decision_count < episode_count * samples_per_episode:
        raise ValueError("decision buffer is too small for the requested episode sample")
    episode_index = decisions.episode_index
    if episode_index.ndim != 1 or episode_index.numel() != decisions.decision_count:
        raise ValueError("decision episode index has an invalid shape")
    if (episode_index < 0).any() or (episode_index >= episode_count).any():
        raise ValueError("decision episode index is outside the episode range")

    counts = torch.bincount(episode_index, minlength=episode_count)
    if bool((counts < samples_per_episode).any().item()):
        raise ValueError("every episode must contain enough recorded decisions")

    generator = torch.Generator(device=episode_index.device)
    generator.manual_seed(int(seed))
    randomized = torch.randperm(
        decisions.decision_count,
        generator=generator,
        device=episode_index.device,
    )
    grouped = randomized[
        torch.argsort(episode_index[randomized], stable=True)
    ]
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=episode_index.device),
            counts.cumsum(dim=0)[:-1],
        )
    )
    positions = offsets[:, None] + torch.arange(
        samples_per_episode,
        dtype=torch.long,
        device=episode_index.device,
    )[None, :]
    selected = grouped[positions.reshape(-1)]
    selected = selected[
        torch.randperm(
            selected.numel(),
            generator=generator,
            device=episode_index.device,
        )
    ]
    return _index_decision_buffer(decisions, selected)


@dataclass(frozen=True)
class TensorizedRolloutBatch:
    costs: torch.Tensor
    mean_costs: torch.Tensor
    success: torch.Tensor
    total_tardiness: torch.Tensor
    system_distance: torch.Tensor
    makespan: torch.Tensor
    on_time_task_count: torch.Tensor
    group_index: torch.Tensor
    plans: torch.Tensor
    plan_lengths: torch.Tensor
    decisions: Optional[TensorizedDecisionBuffer]
    elapsed_s: float
    policy_forward_calls: int
    deferred_attempts: int

    @property
    def episode_count(self) -> int:
        return int(self.costs.numel())


@dataclass(frozen=True)
class TensorizedReinforceLoss:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    negative_log_likelihood: torch.Tensor
    decision_count: int


class _OnlineDecisionReservoir:
    """Uniformly retain K decisions per episode without storing trajectories."""

    def __init__(
        self,
        episode_count: int,
        samples_per_episode: int,
        *,
        device: torch.device,
        seed: int,
    ):
        if episode_count < 1 or samples_per_episode < 1:
            raise ValueError("online decision sampling sizes must be positive")
        self.episode_count = int(episode_count)
        self.samples_per_episode = int(samples_per_episode)
        self.device = torch.device(device)
        self.counts = torch.zeros(
            self.episode_count, dtype=torch.long, device=self.device
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.task_features = None
        self.agent_features = None
        self.action_mask = None
        self.agent_index = None
        self.actions = None

    def _allocate(
        self,
        task_features: torch.Tensor,
        agent_features: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        prefix = (self.episode_count, self.samples_per_episode)

        def empty_like_rows(tensor: torch.Tensor) -> torch.Tensor:
            return torch.empty(
                prefix + tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )

        self.task_features = empty_like_rows(task_features)
        self.agent_features = empty_like_rows(agent_features)
        self.action_mask = empty_like_rows(action_mask)
        self.agent_index = empty_like_rows(agent_index)
        self.actions = empty_like_rows(actions)

    def add(
        self,
        rows: torch.Tensor,
        task_features: torch.Tensor,
        agent_features: torch.Tensor,
        action_mask: torch.Tensor,
        agent_index: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if self.task_features is None:
            self._allocate(
                task_features,
                agent_features,
                action_mask,
                agent_index,
                actions,
            )
        if rows.ndim != 1 or rows.numel() != actions.numel():
            raise ValueError("online decision rows have an invalid shape")
        previous_counts = self.counts[rows]
        new_counts = previous_counts + 1
        random_slots = torch.floor(
            torch.rand(
                rows.numel(), generator=self.generator, device=self.device
            )
            * new_counts.to(torch.float32)
        ).to(torch.long)
        slots = torch.where(
            previous_counts < self.samples_per_episode,
            previous_counts,
            random_slots,
        )
        retained = slots < self.samples_per_episode
        retained_rows = rows[retained]
        retained_slots = slots[retained]
        self.counts[rows] = new_counts

        def store(target: torch.Tensor, source: torch.Tensor) -> None:
            target[retained_rows, retained_slots] = source[retained]

        store(self.task_features, task_features)
        store(self.agent_features, agent_features)
        store(self.action_mask, action_mask)
        store(self.agent_index, agent_index)
        store(self.actions, actions)

    def build(self, advantages_by_episode: torch.Tensor) -> TensorizedDecisionBuffer:
        if self.task_features is None:
            raise RuntimeError("online decision reservoir is empty")
        if bool((self.counts < self.samples_per_episode).any().item()):
            raise RuntimeError("an episode has too few decisions for online sampling")
        retained_count = self.episode_count * self.samples_per_episode
        episode_index = torch.arange(
            self.episode_count, dtype=torch.long, device=self.device
        )[:, None].expand(-1, self.samples_per_episode).reshape(-1)

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(retained_count, *tensor.shape[2:])

        decisions = TensorizedDecisionBuffer(
            task_features=flatten(self.task_features),
            agent_features=flatten(self.agent_features),
            action_mask=flatten(self.action_mask),
            agent_index=flatten(self.agent_index),
            actions=flatten(self.actions),
            episode_index=episode_index,
            cost_advantages=advantages_by_episode[episode_index],
        )
        order = torch.randperm(
            retained_count, generator=self.generator, device=self.device
        )
        return _index_decision_buffer(decisions, order)


def _policy_device(policy: nn.Module) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalized_order(order: torch.Tensor) -> torch.Tensor:
    """Compress finite per-row priorities to stable ranks starting at zero."""
    width = order.size(1)
    indices = order.argsort(dim=1, stable=True)
    ranks = torch.empty_like(indices)
    values = torch.arange(width, device=order.device).expand_as(indices)
    ranks.scatter_(1, indices, values)
    return torch.where(torch.isfinite(order), ranks.to(order.dtype), torch.inf)


def _combine_orders(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = _normalized_order(first)
    second = torch.where(torch.isfinite(first), torch.inf, second)
    second = _normalized_order(second)
    first_count = torch.isfinite(first).sum(dim=1, keepdim=True).to(first.dtype)
    return torch.where(
        torch.isfinite(first),
        first,
        torch.where(torch.isfinite(second), first_count + second, torch.inf),
    )


class TensorizedMRSEnv:
    """A fixed-shape batch of training environments on one torch device."""

    def __init__(
        self,
        instances: Sequence[MarsupialInstance],
        *,
        pomo_size: int,
        device: torch.device,
        max_open_tasks: int,
        env_seed: int,
        training_cfm: bool = True,
        objective_mode: str = "v4",
        task_release_times: Optional[Sequence[Sequence[float]]] = None,
        travel_time_factors: Optional[Sequence[Sequence[float]]] = None,
        operation_time_factors: Optional[Sequence[Sequence[float]]] = None,
    ):
        if not instances:
            raise ValueError("at least one instance is required")
        if pomo_size < 1:
            raise ValueError("pomo_size must be positive")
        reference = instances[0]
        signature = (len(reference.tasks), reference.n_mbr, reference.n_dor)
        if any(
            (len(item.tasks), item.n_mbr, item.n_dor) != signature
            for item in instances
        ):
            raise ValueError("tensorized rollouts require one fixed task and fleet size")
        if any(
            (
                item.speed,
                item.resolved_mbr_speed,
                item.resolved_dor_speed,
                item.dock_time,
                item.detach_time,
            )
            != (
                reference.speed,
                reference.resolved_mbr_speed,
                reference.resolved_dor_speed,
                reference.dock_time,
                reference.detach_time,
            )
            for item in instances
        ):
            raise ValueError("tensorized rollouts require common physical parameters")
        if max_open_tasks <= 0:
            raise ValueError("max_open_tasks must be positive")

        self.device = torch.device(device)
        self.unique_count = len(instances)
        self.pomo_size = int(pomo_size)
        self.batch_size = self.unique_count * self.pomo_size
        self.task_count, self.n_mbr, self.n_dor = signature
        self.agent_count = self.n_mbr + self.n_dor
        self.max_open_tasks = int(max_open_tasks)
        self.training_cfm = bool(training_cfm)
        self.objective_mode = normalize_objective_mode(objective_mode)
        self.physical_dtype = torch.float64
        self.speed = float(reference.speed)
        self.mbr_speed = float(reference.resolved_mbr_speed)
        self.dor_speed = float(reference.resolved_dor_speed)
        self.dock_time = float(reference.dock_time)
        self.detach_time = float(reference.detach_time)

        def task_tensor(attribute):
            values = [
                [getattr(task, attribute) for task in item.tasks]
                for item in instances
            ]
            return torch.tensor(values, dtype=self.physical_dtype, device=self.device).repeat_interleave(
                self.pomo_size, dim=0
            )

        self.sources = task_tensor("source")
        self.destinations = task_tensor("destination")
        self.due_times = task_tensor("due_time")
        self.pickup_times = task_tensor("pickup_time")
        self.handling_times = task_tensor("handling_time")

        def scenario_tensor(values, default, name, *, allow_zero=False):
            if values is None:
                values = [[default] * self.task_count for _ in instances]
            if len(values) != self.unique_count or any(
                len(row) != self.task_count for row in values
            ):
                raise ValueError(f"{name} must have one task vector per instance")
            result = torch.tensor(
                values, dtype=self.physical_dtype, device=self.device
            ).repeat_interleave(self.pomo_size, dim=0)
            if not torch.isfinite(result).all() or (
                (result < 0).any() if allow_zero else (result <= 0).any()
            ):
                raise ValueError(f"{name} must contain finite valid values")
            return result

        self.task_release_times = scenario_tensor(
            task_release_times, 0.0, "task release times", allow_zero=True
        )
        self.travel_time_factors = scenario_tensor(
            travel_time_factors, 1.0, "travel time factors"
        )
        self.operation_time_factors = scenario_tensor(
            operation_time_factors, 1.0, "operation time factors"
        )
        self.task_released = self.task_release_times <= 0.0
        self.source_destination_distance = (
            self.sources - self.destinations
        ).abs().sum(dim=-1)
        # Compact observations use physical scales for every objective mode.
        physical_scale = torch.stack(
            (
                self.pickup_times + self.handling_times,
                torch.full_like(self.pickup_times, self.dock_time + self.detach_time),
                torch.full_like(self.pickup_times, 200.0 / self.speed),
            ),
            dim=-1,
        ).sum(dim=-1).max(dim=1).values
        self.time_scale = physical_scale.clamp_min(1.0)
        if self.objective_mode != "v4":
            self.time_limit = torch.full_like(
                self.time_scale,
                protocol_episode_timeout(self.task_count),
            )
        else:
            self.time_limit = 10.0 * self.due_times.max(dim=1).values
        self.group_index = torch.arange(
            self.unique_count, dtype=torch.long, device=self.device
        ).repeat_interleave(self.pomo_size)

        batch = self.batch_size
        self.time = torch.zeros(batch, dtype=self.physical_dtype, device=self.device)
        self.task_status = torch.full(
            (batch, self.task_count), EMPTY, dtype=torch.long, device=self.device
        )
        self.task_mbr = torch.full_like(self.task_status, -1)
        self.task_dor = torch.full_like(self.task_status, -1)

        self.mbr_position = torch.tensor(
            [item.mbr_initial_positions for item in instances],
            dtype=self.physical_dtype,
            device=self.device,
        ).repeat_interleave(self.pomo_size, dim=0)
        self.dor_position = torch.tensor(
            [item.dor_initial_positions for item in instances],
            dtype=self.physical_dtype,
            device=self.device,
        ).repeat_interleave(self.pomo_size, dim=0)
        self.mbr_available = torch.zeros(
            (batch, self.n_mbr), dtype=self.physical_dtype, device=self.device
        )
        self.dor_available = torch.zeros(
            (batch, self.n_dor), dtype=self.physical_dtype, device=self.device
        )
        self.mbr_travel_until = torch.zeros_like(self.mbr_available)
        self.dor_travel_until = torch.zeros_like(self.dor_available)
        self.mbr_busy = torch.zeros(
            (batch, self.n_mbr), dtype=torch.bool, device=self.device
        )
        self.dor_busy = torch.zeros(
            (batch, self.n_dor), dtype=torch.bool, device=self.device
        )
        self.mbr_waiting = torch.zeros_like(self.mbr_busy)
        self.dor_waiting = torch.zeros_like(self.dor_busy)
        self.mbr_current_task = torch.full(
            (batch, self.n_mbr), -1, dtype=torch.long, device=self.device
        )
        self.dor_current_task = torch.full(
            (batch, self.n_dor), -1, dtype=torch.long, device=self.device
        )

        self.total_tardiness = torch.zeros_like(self.time)
        self.system_distance = torch.zeros_like(self.time)
        self.on_time_task_count = torch.zeros_like(self.time)
        self.plan_tasks = torch.full(
            (batch, self.task_count), -1, dtype=torch.long, device=self.device
        )
        self.plan_mbr = torch.full_like(self.plan_tasks, -1)
        self.plan_dor = torch.full_like(self.plan_tasks, -1)
        self.plan_length = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.decision_count = torch.zeros_like(self.plan_length)
        self.event_steps = torch.ones_like(self.plan_length)
        self.deferred_attempts = torch.zeros_like(self.plan_length)
        self.success = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros_like(self.success)
        self.alive = torch.ones_like(self.success)

        self.phase = torch.full(
            (batch,), READY_PHASE, dtype=torch.long, device=self.device
        )
        self.ready_order = torch.full(
            (batch, self.agent_count), torch.inf, device=self.device
        )
        self.deferred_order = torch.full_like(self.ready_order, torch.inf)
        self.new_blocked_order = torch.full_like(self.ready_order, torch.inf)
        self.cfm_order = torch.full_like(self.ready_order, torch.inf)
        self.cfm_next_order = torch.full_like(self.ready_order, torch.inf)
        self.cfm_progress = torch.zeros(batch, dtype=torch.bool, device=self.device)

        self.env_generator = torch.Generator(device=self.device)
        self.env_generator.manual_seed(int(env_seed))
        initial_ready = torch.ones(
            (batch, self.agent_count), dtype=torch.bool, device=self.device
        )
        self.ready_order = self._random_order(initial_ready)

    def _random_order(self, mask: torch.Tensor) -> torch.Tensor:
        priorities = torch.rand(
            mask.shape, generator=self.env_generator, device=self.device
        )
        return _normalized_order(torch.where(mask, priorities, torch.inf))

    def _candidate_exists(self) -> torch.Tensor:
        return self.alive & (
            ((self.phase == READY_PHASE) & torch.isfinite(self.ready_order).any(dim=1))
            | ((self.phase == CFM_PHASE) & torch.isfinite(self.cfm_order).any(dim=1))
        )

    def _advance_rows(self, rows: torch.Tensor) -> None:
        done = rows & (self.task_status == COMPLETED).all(dim=1)
        self.success[done] = True
        self.alive[done] = False
        rows = rows & ~done
        if not rows.any():
            return

        has_pending = (
            self.mbr_busy.any(dim=1)
            | self.dor_busy.any(dim=1)
            | (~self.task_released).any(dim=1)
        )
        deadlocked = rows & ~has_pending
        self.failed[deadlocked] = True
        self.alive[deadlocked] = False
        rows = rows & has_pending
        if not rows.any():
            return

        mbr_times = torch.where(self.mbr_busy, self.mbr_available, torch.inf)
        dor_times = torch.where(self.dor_busy, self.dor_available, torch.inf)
        release_times = torch.where(
            ~self.task_released, self.task_release_times, torch.inf
        )
        next_time = torch.minimum(
            torch.minimum(mbr_times.min(dim=1).values, dor_times.min(dim=1).values),
            release_times.min(dim=1).values,
        )
        timed_out = rows & (next_time > self.time_limit)
        self.time[timed_out] = self.time_limit[timed_out]
        self.failed[timed_out] = True
        self.alive[timed_out] = False
        rows = rows & ~timed_out
        if not rows.any():
            return

        self.time[rows] = next_time[rows]
        release_mbr = rows[:, None] & self.mbr_busy & torch.isclose(
            self.mbr_available, next_time[:, None], atol=1e-6, rtol=0.0
        )
        release_dor = rows[:, None] & self.dor_busy & torch.isclose(
            self.dor_available, next_time[:, None], atol=1e-6, rtol=0.0
        )
        release_tasks = rows[:, None] & ~self.task_released & torch.isclose(
            self.task_release_times, next_time[:, None], atol=1e-6, rtol=0.0
        )
        self.task_released[release_tasks] = True
        dor_rows, dor_ids = release_dor.nonzero(as_tuple=True)
        if dor_rows.numel():
            completed_tasks = self.dor_current_task[dor_rows, dor_ids]
            self.task_status[dor_rows, completed_tasks] = COMPLETED

        self.mbr_busy[release_mbr] = False
        self.dor_busy[release_dor] = False
        self.mbr_current_task[release_mbr] = -1
        self.dor_current_task[release_dor] = -1
        self.mbr_travel_until[release_mbr] = 0.0
        self.dor_travel_until[release_dor] = 0.0

        ready = torch.cat((release_mbr, release_dor), dim=1)
        if release_tasks.any():
            available_after_event = ~torch.cat(
                (self.mbr_busy | self.mbr_waiting, self.dor_busy | self.dor_waiting),
                dim=1,
            )
            ready = ready | (
                release_tasks.any(dim=1)[:, None] & available_after_event
            )
        new_order = self._random_order(ready)
        self.ready_order[rows] = new_order[rows]
        self.new_blocked_order[rows] = torch.inf
        self.phase[rows] = READY_PHASE
        self.event_steps[rows] += 1
        event_limit = 4 * self.task_count + self.agent_count + 4
        exceeded = rows & (self.event_steps > event_limit)
        self.failed[exceeded] = True
        self.alive[exceeded] = False

    def normalize_scheduler(self) -> None:
        for _ in range(8):
            if bool((~self.alive | self._candidate_exists()).all().item()):
                return

            ready_empty = (
                self.alive
                & (self.phase == READY_PHASE)
                & ~torch.isfinite(self.ready_order).any(dim=1)
            )
            if ready_empty.any():
                if self.training_cfm:
                    combined = _combine_orders(
                        self.deferred_order, self.new_blocked_order
                    )
                    self.cfm_order[ready_empty] = combined[ready_empty]
                    self.deferred_order[ready_empty] = torch.inf
                    self.new_blocked_order[ready_empty] = torch.inf
                    self.cfm_next_order[ready_empty] = torch.inf
                    self.cfm_progress[ready_empty] = False
                    self.phase[ready_empty] = CFM_PHASE
                else:
                    self.phase[ready_empty] = ADVANCE_PHASE

            cfm_empty = (
                self.alive
                & (self.phase == CFM_PHASE)
                & ~torch.isfinite(self.cfm_order).any(dim=1)
            )
            next_exists = torch.isfinite(self.cfm_next_order).any(dim=1)
            restart = cfm_empty & self.cfm_progress & next_exists
            if restart.any():
                normalized = _normalized_order(self.cfm_next_order)
                self.cfm_order[restart] = normalized[restart]
                self.cfm_next_order[restart] = torch.inf
                self.cfm_progress[restart] = False

            finish = cfm_empty & ~restart
            if finish.any():
                stalled = finish & ~self.cfm_progress
                if stalled.any():
                    normalized = _normalized_order(self.cfm_next_order)
                    self.deferred_order[stalled] = normalized[stalled]
                progressed = finish & self.cfm_progress
                self.deferred_order[progressed] = torch.inf
                self.cfm_next_order[finish] = torch.inf
                self.cfm_progress[finish] = False
                self.phase[finish] = ADVANCE_PHASE

            advancing = self.alive & (self.phase == ADVANCE_PHASE)
            if advancing.any():
                self._advance_rows(advancing)
        raise RuntimeError("tensorized scheduler failed to reach a decision boundary")

    def _combined_agent_state(self, rows: torch.Tensor):
        position = torch.cat((self.mbr_position[rows], self.dor_position[rows]), dim=1)
        available = torch.cat((self.mbr_available[rows], self.dor_available[rows]), dim=1)
        travel_until = torch.cat(
            (self.mbr_travel_until[rows], self.dor_travel_until[rows]), dim=1
        )
        busy = torch.cat((self.mbr_busy[rows], self.dor_busy[rows]), dim=1)
        waiting = torch.cat((self.mbr_waiting[rows], self.dor_waiting[rows]), dim=1)
        return position, available, travel_until, busy, waiting

    def legal_task_mask(
        self, rows: torch.Tensor, agent_index: torch.Tensor
    ) -> torch.Tensor:
        status = self.task_status[rows]
        is_mbr = agent_index < self.n_mbr
        open_count = (status == OPEN).sum(dim=1)
        empty_legal = status == EMPTY
        if self.training_cfm:
            empty_legal = empty_legal & (
                open_count < self.max_open_tasks
            )[:, None]
        missing_mbr = self.task_mbr[rows] < 0
        missing_dor = self.task_dor[rows] < 0
        open_legal = (status == OPEN) & torch.where(
            is_mbr[:, None], missing_mbr, missing_dor
        )
        return (empty_legal | open_legal) & self.task_released[rows]

    def build_owner_indices(self, rows: torch.Tensor) -> torch.Tensor:
        """Current assignment rows in the public robot table; no lookahead."""
        cr = self.task_mbr[rows]
        wr = self.task_dor[rows]
        return torch.stack((cr, torch.where(wr >= 0, wr + self.n_mbr, -1)), dim=-1).long()

    def build_observation(
        self, rows: torch.Tensor, agent_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position, available, travel_until, busy, waiting = self._combined_agent_state(rows)
        gather_index = agent_index[:, None, None].expand(-1, 1, 2)
        deciding_position = position.gather(1, gather_index).squeeze(1)
        status = self.task_status[rows]
        active = (status == EMPTY) | (status == OPEN)
        source = self.sources[rows]
        destination = self.destinations[rows]
        location_scale = 100.0
        time_scale = self.time_scale[rows, None]

        source_delta = (source - deciding_position[:, None, :]) / location_scale
        destination_delta = (
            destination - deciding_position[:, None, :]
        ) / location_scale
        requirements = (
            (active & (self.task_mbr[rows] < 0)).float().unsqueeze(-1),
            (active & (self.task_dor[rows] < 0)).float().unsqueeze(-1),
        )
        attributes = (
            source_delta, destination_delta,
            (self.pickup_times[rows] / time_scale).unsqueeze(-1),
            (self.handling_times[rows] / time_scale).unsqueeze(-1),
        )
        completed = (status == COMPLETED).to(self.physical_dtype).unsqueeze(-1)
        task_features = torch.cat((*requirements, *attributes, completed), dim=-1)

        remaining_travel = torch.where(
            busy,
            (travel_until - self.time[rows, None]).clamp_min(0.0),
            0.0,
        )
        remaining_execution = torch.where(
            busy,
            (available - torch.maximum(self.time[rows, None], travel_until)).clamp_min(0.0),
            0.0,
        )
        waiting_time = torch.where(
            waiting,
            (self.time[rows, None] - available).clamp_min(0.0),
            0.0,
        )
        role_ids = torch.arange(self.agent_count, device=self.device)
        is_mbr_agent = (role_ids < self.n_mbr).to(self.physical_dtype)[None, :].expand(
            rows.numel(), -1
        )
        relative = (position - deciding_position[:, None, :]) / location_scale
        assignment = torch.where(
            busy,
            torch.ones_like(available),
            torch.where(waiting, torch.zeros_like(available), -torch.ones_like(available)),
        )
        agent_features = torch.cat(
            (
                is_mbr_agent.unsqueeze(-1),
                (1.0 - is_mbr_agent).unsqueeze(-1),
                (remaining_travel / time_scale).unsqueeze(-1),
                (remaining_execution / time_scale).unsqueeze(-1),
                (waiting_time / time_scale).unsqueeze(-1),
                relative,
                assignment.unsqueeze(-1),
            ),
            dim=-1,
        )
        action_mask = ~self.legal_task_mask(rows, agent_index)
        task_features = insert_owner_one_hot(
            task_features, self.build_owner_indices(rows), self.n_mbr, self.n_dor
        )
        task_dim = task_feature_dim(self.n_mbr, self.n_dor)
        if task_features.shape != (rows.numel(), self.task_count, task_dim):
            raise AssertionError("unexpected tensorized task observation shape")
        if agent_features.shape != (rows.numel(), self.agent_count, AGENT_FEATURE_DIM):
            raise AssertionError("unexpected tensorized agent observation shape")
        return task_features.float(), agent_features.float(), action_mask


    def next_decisions(self):
        while True:
            self.normalize_scheduler()
            rows = self._candidate_exists().nonzero(as_tuple=True)[0]
            if not rows.numel():
                return None
            order = torch.where(
                (self.phase[rows] == READY_PHASE)[:, None],
                self.ready_order[rows],
                self.cfm_order[rows],
            )
            agent_index = order.argmin(dim=1)
            priority = order.gather(1, agent_index[:, None]).squeeze(1)

            _, _, _, busy, waiting = self._combined_agent_state(rows)
            available = ~(busy | waiting).gather(1, agent_index[:, None]).squeeze(1)
            unavailable_rows = rows[~available]
            unavailable_agents = agent_index[~available]
            if unavailable_rows.numel():
                ready = self.phase[unavailable_rows] == READY_PHASE
                self.ready_order[
                    unavailable_rows[ready], unavailable_agents[ready]
                ] = torch.inf
                self.cfm_order[
                    unavailable_rows[~ready], unavailable_agents[~ready]
                ] = torch.inf

            candidate_rows = rows[available]
            candidate_agents = agent_index[available]
            candidate_priority = priority[available]
            if not candidate_rows.numel():
                continue
            legal = self.legal_task_mask(candidate_rows, candidate_agents)
            has_legal = legal.any(dim=1)
            blocked_rows = candidate_rows[~has_legal]
            blocked_agents = candidate_agents[~has_legal]
            blocked_priority = candidate_priority[~has_legal]
            if blocked_rows.numel():
                ready = self.phase[blocked_rows] == READY_PHASE
                ready_rows, ready_agents = blocked_rows[ready], blocked_agents[ready]
                self.ready_order[ready_rows, ready_agents] = torch.inf
                self.new_blocked_order[ready_rows, ready_agents] = blocked_priority[ready]
                cfm_rows, cfm_agents = blocked_rows[~ready], blocked_agents[~ready]
                self.cfm_order[cfm_rows, cfm_agents] = torch.inf
                self.cfm_next_order[cfm_rows, cfm_agents] = blocked_priority[~ready]
                self.deferred_attempts[blocked_rows] += 1

            decision_rows = candidate_rows[has_legal]
            decision_agents = candidate_agents[has_legal]
            if not decision_rows.numel():
                continue
            task_features, agent_features, action_mask = self.build_observation(
                decision_rows, decision_agents
            )
            return (
                decision_rows,
                decision_agents,
                task_features,
                agent_features,
                action_mask,
            )

    def apply_actions(
        self,
        rows: torch.Tensor,
        agent_index: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if rows.numel() != actions.numel() or rows.numel() != agent_index.numel():
            raise ValueError("rows, agents, and actions must have equal length")
        legal = self.legal_task_mask(rows, agent_index)
        if (~legal.gather(1, actions[:, None]).squeeze(1)).any():
            raise ValueError("tensorized policy selected an illegal task")

        ready = self.phase[rows] == READY_PHASE
        self.ready_order[rows[ready], agent_index[ready]] = torch.inf
        self.cfm_order[rows[~ready], agent_index[~ready]] = torch.inf
        self.cfm_progress[rows[~ready]] = True
        self.decision_count[rows] += 1

        is_mbr = agent_index < self.n_mbr
        mbr_rows, mbr_tasks = rows[is_mbr], actions[is_mbr]
        mbr_ids = agent_index[is_mbr]
        if mbr_rows.numel():
            self.task_mbr[mbr_rows, mbr_tasks] = mbr_ids
            self.mbr_waiting[mbr_rows, mbr_ids] = True
            self.mbr_current_task[mbr_rows, mbr_ids] = mbr_tasks
        dor_rows, dor_tasks = rows[~is_mbr], actions[~is_mbr]
        dor_ids = agent_index[~is_mbr] - self.n_mbr
        if dor_rows.numel():
            self.task_dor[dor_rows, dor_tasks] = dor_ids
            self.dor_waiting[dor_rows, dor_ids] = True
            self.dor_current_task[dor_rows, dor_ids] = dor_tasks
        self.task_status[rows, actions] = OPEN

        paired = (
            (self.task_mbr[rows, actions] >= 0)
            & (self.task_dor[rows, actions] >= 0)
        )
        pair_rows = rows[paired]
        pair_tasks = actions[paired]
        if not pair_rows.numel():
            return
        pair_mbr = self.task_mbr[pair_rows, pair_tasks]
        pair_dor = self.task_dor[pair_rows, pair_tasks]
        mbr_position = self.mbr_position[pair_rows, pair_mbr]
        dor_position = self.dor_position[pair_rows, pair_dor]
        mbr_available = self.mbr_available[pair_rows, pair_mbr]
        dor_available = self.dor_available[pair_rows, pair_dor]
        rendezvous = (mbr_position - dor_position).abs().sum(dim=-1)
        travel_factor = self.travel_time_factors[pair_rows, pair_tasks]
        operation_factor = self.operation_time_factors[pair_rows, pair_tasks]
        mbr_arrival = mbr_available + travel_factor * rendezvous / self.mbr_speed
        synchronized = torch.maximum(dor_available, mbr_arrival)
        dock_finish = synchronized + self.dock_time
        dor_to_source = (
            dor_position - self.sources[pair_rows, pair_tasks]
        ).abs().sum(dim=-1)
        transport = self.source_destination_distance[pair_rows, pair_tasks]
        destination_time = (
            dock_finish
            + travel_factor * dor_to_source / self.mbr_speed
            + self.pickup_times[pair_rows, pair_tasks]
            + travel_factor * transport / self.mbr_speed
        )
        mbr_release = destination_time + self.detach_time
        dor_release = (
            mbr_release + operation_factor * self.handling_times[pair_rows, pair_tasks]
        )
        destination = self.destinations[pair_rows, pair_tasks]

        self.mbr_position[pair_rows, pair_mbr] = destination
        self.dor_position[pair_rows, pair_dor] = destination
        self.mbr_available[pair_rows, pair_mbr] = mbr_release
        self.dor_available[pair_rows, pair_dor] = dor_release
        self.mbr_travel_until[pair_rows, pair_mbr] = destination_time
        self.dor_travel_until[pair_rows, pair_dor] = destination_time
        self.mbr_waiting[pair_rows, pair_mbr] = False
        self.dor_waiting[pair_rows, pair_dor] = False
        self.mbr_busy[pair_rows, pair_mbr] = True
        self.dor_busy[pair_rows, pair_dor] = True
        self.task_status[pair_rows, pair_tasks] = IN_PROGRESS
        task_tardiness = (
            dor_release - self.due_times[pair_rows, pair_tasks]
        ).clamp_min(0.0)
        self.total_tardiness[pair_rows] += task_tardiness
        self.on_time_task_count[pair_rows] += (task_tardiness <= 0.0).to(
            self.on_time_task_count.dtype
        )
        self.system_distance[pair_rows] += rendezvous + dor_to_source + transport

        slots = self.plan_length[pair_rows]
        self.plan_tasks[pair_rows, slots] = pair_tasks
        self.plan_mbr[pair_rows, slots] = pair_mbr
        self.plan_dor[pair_rows, slots] = pair_dor
        self.plan_length[pair_rows] += 1

    def costs(self) -> torch.Tensor:
        if self.objective_mode == "makespan":
            partial = self.time
            penalty = torch.full_like(
                partial, protocol_failure_penalty(self.task_count)
            )
        elif self.objective_mode == "engineering":
            partial = 0.9 * self.time + 0.1 * self.system_distance / self.mbr_speed
            penalty = torch.full_like(
                partial, protocol_failure_penalty(self.task_count)
            )
        elif self.objective_mode == "distance_tardiness":
            partial = (
                0.1 * self.system_distance / self.mbr_speed
                + 0.9 * self.total_tardiness
            )
            penalty = torch.full_like(
                partial, protocol_failure_penalty(self.task_count)
            )
        elif self.objective_mode == "makespan_distance":
            partial = (
                OBJECTIVE_MAKESPAN_DISTANCE_WEIGHT * self.time
                + OBJECTIVE_CARRIER_DISTANCE_TIME_WEIGHT
                * self.system_distance
                / self.mbr_speed
            )
            penalty = torch.full_like(
                partial, protocol_failure_penalty(self.task_count)
            )
        else:
            partial = (
                0.95 * self.total_tardiness / self.task_count
                + 0.05 * self.system_distance / (self.task_count * self.speed)
            )
            penalty = 10.0 * self.due_times.max(dim=1).values
        return partial + torch.where(self.failed, penalty, 0.0)

    def plans(self) -> torch.Tensor:
        return torch.stack((self.plan_tasks, self.plan_mbr, self.plan_dor), dim=-1)


def run_tensorized_batch(
    policy: nn.Module,
    instances: Sequence[MarsupialInstance],
    *,
    pomo_size: int = 10,
    sample: bool = True,
    training_cfm: bool = True,
    env_seed: int = 0,
    action_seed: int = 0,
    max_open_tasks: int = 5,
    record_decisions: bool = True,
    decision_samples_per_episode: Optional[int] = None,
    decision_sample_seed: int = 0,
    objective_mode: str = "v4",
    task_release_times: Optional[Sequence[Sequence[float]]] = None,
    travel_time_factors: Optional[Sequence[Sequence[float]]] = None,
    operation_time_factors: Optional[Sequence[Sequence[float]]] = None,
) -> TensorizedRolloutBatch:
    """Run equal-sized instances and all POMO repeats as one device batch."""
    device = _policy_device(policy)
    environment = TensorizedMRSEnv(
        instances,
        pomo_size=pomo_size,
        device=device,
        max_open_tasks=max_open_tasks,
        env_seed=env_seed,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
        task_release_times=task_release_times,
        travel_time_factors=travel_time_factors,
        operation_time_factors=operation_time_factors,
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(int(action_seed))
    if decision_samples_per_episode is not None:
        if not record_decisions:
            raise ValueError("online decision sampling requires decision recording")
        if decision_samples_per_episode < 1:
            raise ValueError("decision_samples_per_episode must be positive")
    decision_reservoir = (
        _OnlineDecisionReservoir(
            environment.batch_size,
            decision_samples_per_episode,
            device=device,
            seed=decision_sample_seed,
        )
        if decision_samples_per_episode is not None
        else None
    )
    task_chunks = []
    agent_chunks = []
    mask_chunks = []
    agent_index_chunks = []
    action_chunks = []
    episode_chunks = []
    policy_forward_calls = 0
    was_training = policy.training
    policy.eval()
    started = perf_counter()

    try:
        with torch.inference_mode():
            while environment.alive.any():
                request = environment.next_decisions()
                if request is None:
                    continue
                rows, agent_index, task_features, agent_features, action_mask = request
                probabilities, log_probabilities = policy(
                    task_features, agent_features, action_mask, agent_index,
                )
                policy_forward_calls += 1
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("tensorized policy produced non-finite probabilities")
                if sample:
                    actions = torch.multinomial(
                        probabilities, 1, generator=action_generator
                    ).squeeze(1)
                else:
                    actions = probabilities.argmax(dim=1)
                selected_log = log_probabilities.gather(1, actions[:, None]).squeeze(1)
                if not torch.isfinite(selected_log).all():
                    raise FloatingPointError("tensorized selected log probability is non-finite")
                if action_mask.gather(1, actions[:, None]).any():
                    raise AssertionError("tensorized policy selected a masked action")
                if record_decisions:
                    if decision_reservoir is not None:
                        decision_reservoir.add(
                            rows,
                            task_features,
                            agent_features,
                            action_mask,
                            agent_index,
                            actions,
                        )
                    else:
                        task_chunks.append(task_features)
                        agent_chunks.append(agent_features)
                        mask_chunks.append(action_mask)
                        agent_index_chunks.append(agent_index)
                        action_chunks.append(actions)
                        episode_chunks.append(rows)
                environment.apply_actions(rows, agent_index, actions)
    finally:
        policy.train(was_training)

    costs = environment.costs()
    grouped_costs = costs.view(environment.unique_count, environment.pomo_size)
    mean_costs = grouped_costs.mean(dim=1)
    decisions = None
    if record_decisions:
        advantages_by_episode = costs - mean_costs[environment.group_index]
        if decision_reservoir is not None:
            decisions = decision_reservoir.build(advantages_by_episode)
        else:
            episode_index = torch.cat(episode_chunks)
            decisions = TensorizedDecisionBuffer(
                task_features=torch.cat(task_chunks),
                agent_features=torch.cat(agent_chunks),
                action_mask=torch.cat(mask_chunks),
                agent_index=torch.cat(agent_index_chunks),
                actions=torch.cat(action_chunks),
                episode_index=episode_index,
                cost_advantages=advantages_by_episode[episode_index],
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return TensorizedRolloutBatch(
        costs=costs,
        mean_costs=mean_costs,
        success=environment.success,
        total_tardiness=environment.total_tardiness,
        system_distance=environment.system_distance,
        makespan=environment.time,
        on_time_task_count=environment.on_time_task_count,
        group_index=environment.group_index,
        plans=environment.plans(),
        plan_lengths=environment.plan_length,
        decisions=decisions,
        elapsed_s=elapsed,
        policy_forward_calls=policy_forward_calls,
        deferred_attempts=int(environment.deferred_attempts.sum().item()),
    )


def tensorized_reinforce_loss(
    policy: nn.Module,
    decisions: TensorizedDecisionBuffer,
    start: int = 0,
    end: Optional[int] = None,
) -> TensorizedReinforceLoss:
    """Recompute action log probabilities for a contiguous trajectory slice."""
    end = decisions.decision_count if end is None else int(end)
    start = int(start)
    if not 0 <= start < end <= decisions.decision_count:
        raise ValueError("invalid tensorized decision slice")
    selected = slice(start, end)
    task_features = decisions.task_features[selected]
    agent_features = decisions.agent_features[selected]
    action_mask = decisions.action_mask[selected]
    agent_index = decisions.agent_index[selected]
    actions = decisions.actions[selected]
    advantages = decisions.cost_advantages[selected]
    probabilities, log_probabilities = policy(
        task_features, agent_features, action_mask, agent_index,
    )
    selected_log_probabilities = log_probabilities.gather(
        1, actions[:, None]
    ).squeeze(1)
    policy_loss = (selected_log_probabilities * advantages.detach()).mean()
    negative_log_likelihood = -selected_log_probabilities.mean()
    safe_log_probabilities = log_probabilities.masked_fill(action_mask, 0.0)
    entropy = -(probabilities * safe_log_probabilities).sum(dim=-1).mean()
    if not (
        torch.isfinite(policy_loss)
        and torch.isfinite(entropy)
        and torch.isfinite(negative_log_likelihood)
    ):
        raise FloatingPointError("non-finite tensorized REINFORCE loss")
    return TensorizedReinforceLoss(
        policy_loss, entropy, negative_log_likelihood, end - start
    )
