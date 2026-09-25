"""Literature-grounded decentralized task-allocation baselines.

The methods in this module are event-level policies.  A robot only commits its
next task from the current broadcast state; no method constructs or evaluates a
complete fleet plan.  The auction methods use exact *one-step* Marsupial pair
consequences, while the EDD policy is a non-auction control group.

The implementations are deliberately named by protocol rather than presented
as exact reproductions of every domain-specific paper variant:

* ``d_edd`` is an earliest-due-date sequential rule;
* ``d_murdoch`` is a MURDOCH/Contract-Net-style event auction;
* ``d_coalition_auction`` is an adaptation of the two-robot coalition auction
  in Deng et al. (IEEE T-RO, 2024);
* ``d_min_slack`` is a deadline-slack pair greedy rule;
* ``d_pi_coupled`` is an event-level coupled performance-impact adaptation;
* ``d_cbta`` is a timetable-consensus adaptation of Wang et al. (IEEE RA-L,
  2022) for simultaneous multi-agent tasks;
* ``d_group_auction`` is a fixed-size group auction adaptation of Bai et al.
  (IEEE T-ASE, 2023);
* ``d_collective_auction`` is a current-pair coalition auction adapted from
  recent decentralized collective-transport work.
* ``d_min_min`` and ``d_max_min`` are classic makespan dispatching rules:
  Min-min selects the easiest current pair first, while Max-min protects the
  hardest current task first.

Marsupial tasks require one MBR and one DOR, so the published single-robot
auction scores are adapted to a current-state pair transition.  This is a
transparent domain adaptation, not a claim that the original papers used the
same vehicle dynamics or objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .core import (
    TaskSpec,
    deadlock_penalty,
    makespan_distance_objective,
    protocol_episode_timeout,
    protocol_failure_penalty,
)
from .decentralized_rollout import AgentKey, DecisionRecord, EpisodeOutcome
from .event_env import DecentralizedMRSEnv, Role, TaskStatus
from .lookahead import transition_features
from .observations import ObservationScales, PolicyObservation, build_policy_observation
from .scalar_execution import AssignmentClaim, ScalarExecutionProtocol


MARKET_METHODS = (
    "d_edd",
    "d_min_min",
    "d_max_min",
    "d_murdoch",
    "d_coalition_auction",
    "d_min_slack",
    "d_pi_coupled",
    "d_cbta",
    "d_group_auction",
    "d_collective_auction",
)
MARKET_LABELS = {
    "d_edd": "D-EDD",
    "d_min_min": "D-MinMin",
    "d_max_min": "D-MaxMin",
    "d_murdoch": "D-MURDOCH-Auction",
    "d_coalition_auction": "D-Coalition-Auction",
    "d_min_slack": "D-MinSlack",
    "d_pi_coupled": "D-PI-Coupled",
    "d_cbta": "D-CBTA",
    "d_group_auction": "D-Group-Auction",
    "d_collective_auction": "D-Collective-Auction",
}
OBJECTIVE_MODES = (
    "v4",
    "makespan",
    "engineering",
    "distance_tardiness",
    "makespan_distance",
)


def _normalize_objective_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"unsupported objective mode: {mode}; "
            f"expected one of {OBJECTIVE_MODES}"
        )
    return mode


def _default_time_limit(
    tasks: Sequence[TaskSpec],
    objective_mode: str,
) -> float:
    """Return a timeout that does not reintroduce due dates in makespan mode."""
    if objective_mode == "v4":
        return deadlock_penalty(tasks)
    return protocol_episode_timeout(len(tasks))


def _failure_penalty(tasks: Sequence[TaskSpec], objective_mode: str) -> float:
    if objective_mode == "v4":
        return deadlock_penalty(tasks)
    return protocol_failure_penalty(len(tasks))


@dataclass(frozen=True)
class MarketBid:
    """A robot's bid for one legal task using one current partner estimate."""

    role: Role
    robot_id: int
    task_id: int
    partner_id: int
    cost: float
    completion_time: float
    due_time: float
    status: TaskStatus
    rendezvous_distance: float
    waiting_time: float

    @property
    def urgency(self) -> float:
        return max(0.0, self.completion_time - self.due_time)


@dataclass(frozen=True)
class PairOffer:
    """A current feasible MBR-DOR coalition offer.

    The offer contains only the immediate transition consequence.  In
    particular, it has no route suffix, future task reservation, or complete
    fleet objective value.
    """

    task_id: int
    mbr_id: int
    dor_id: int
    cost: float
    completion_time: float
    start_time: float
    waiting_time: float
    due_time: float
    status: TaskStatus
    mbr_release_time: float
    dor_release_time: float
    system_distance: float


def _role_count(env: DecentralizedMRSEnv, role: Role) -> int:
    return env.n_mbr if role == Role.MBR else env.n_dor


def _complement_role(role: Role) -> Role:
    return Role.DOR if role == Role.MBR else Role.MBR


def _fixed_partner(env: DecentralizedMRSEnv, role: Role, task_id: int) -> Optional[int]:
    state = env.task_state[task_id]
    if role == Role.MBR:
        return state.dor_id
    return state.mbr_id


def _candidate_partners(
    env: DecentralizedMRSEnv,
    role: Role,
    task_id: int,
) -> Tuple[int, ...]:
    state = env.task_state[task_id]
    fixed = _fixed_partner(env, role, task_id)
    if state.status != TaskStatus.EMPTY:
        return (fixed,) if fixed is not None else tuple()
    return tuple(range(_role_count(env, _complement_role(role))))


def _pair_cost(
    env: DecentralizedMRSEnv,
    task_id: int,
    mbr_id: int,
    dor_id: int,
    *,
    objective_mode: str = "v4",
) -> Tuple[float, float, float, float, float]:
    objective_mode = _normalize_objective_mode(objective_mode)
    values, result = transition_features(env, task_id, mbr_id, dor_id)
    task = env.tasks[task_id]
    if objective_mode == "makespan":
        # A one-step decentralized proxy for the terminal makespan: the
        # predicted completion of the candidate task.  Waiting only resolves
        # exact ties and is not a second optimization objective.
        cost = result.task_completion_time + 1e-6 * (
            result.mbr_wait + result.dor_wait
        )
    elif objective_mode == "engineering":
        # The pair bid uses the same physical time-equivalent travel term as
        # the terminal engineering objective.  It remains one-step only.
        cost = (
            0.9 * result.task_completion_time
            + 0.1 * result.system_distance / env.mbr_speed
            + 1e-6 * (result.mbr_wait + result.dor_wait)
        )
    elif objective_mode == "distance_tardiness":
        tardiness = max(0.0, result.task_completion_time - task.due_time)
        cost = (
            0.1 * result.system_distance / env.mbr_speed
            + 0.9 * tardiness
            + 1e-6 * (result.mbr_wait + result.dor_wait)
        )
    elif objective_mode == "makespan_distance":
        cost = makespan_distance_objective(
            result.task_completion_time,
            result.system_distance,
            env.mbr_speed,
        ) + 1e-6 * (result.mbr_wait + result.dor_wait)
    else:
        tardiness = max(0.0, result.task_completion_time - task.due_time)
        # This is the exact one-task contribution to the frozen J objective.
        # A tiny wait tie-breaker keeps otherwise equal bids deterministic
        # without changing the objective ordering at normal scales.
        cost = (
            0.95 * tardiness
            + 0.05 * result.system_distance / env.speed
            + 1e-6 * (result.mbr_wait + result.dor_wait)
        )
    return (
        float(cost),
        float(result.task_completion_time),
        float(values[0]),
        float(result.mbr_wait + result.dor_wait),
        float(result.system_distance),
    )


def _role_release_time(
    env: DecentralizedMRSEnv,
    role: Role,
    robot_id: int,
    result,
) -> Tuple[float, float]:
    """Return a role's one-step release and marginal release times.

    MURDOCH bids are intentionally role-local.  The transition still uses the
    current candidate partner, but the bid value is only the submitting
    role's release increment; it is not the pair completion or pair waiting
    cost.  No future task or route suffix is evaluated here.
    """
    if role == Role.MBR:
        before = env.mbr_state[robot_id]
        release = float(result.mbr_release_time)
    else:
        before = env.dor_state[robot_id]
        release = float(result.dor_release_time)
    marginal = max(0.0, release - float(before.available_time))
    return release, marginal


def _coalition_offer_cost(offer: PairOffer) -> float:
    """Joint coalition proxy: completion plus current synchronization wait."""
    return float(offer.completion_time + offer.waiting_time)


def _build_bids(
    env: DecentralizedMRSEnv,
    role: Role,
    robot_id: int,
    observation: PolicyObservation,
    *,
    objective_mode: str = "v4",
) -> Tuple[MarketBid, ...]:
    legal = [
        task_id
        for task_id, masked in enumerate(observation.action_mask.tolist())
        if not masked
    ]
    bids: List[MarketBid] = []
    for task_id in legal:
        partners = _candidate_partners(env, role, task_id)
        if not partners:
            continue
        pair_candidates = []
        for partner_id in partners:
            if role == Role.MBR:
                mbr_id, dor_id = robot_id, partner_id
            else:
                mbr_id, dor_id = partner_id, robot_id
            values, result = transition_features(env, task_id, mbr_id, dor_id)
            release, marginal_release = _role_release_time(
                env,
                role,
                robot_id,
                result,
            )
            pair_candidates.append((
                marginal_release,
                float(result.task_completion_time),
                float(values[0]),
                float(result.mbr_wait + result.dor_wait),
                release,
                int(partner_id),
            ))
        marginal_release, completion, rendezvous, waiting, _, partner_id = min(
            pair_candidates,
            key=lambda item: (item[0], item[1], item[5]),
        )
        task = env.tasks[task_id]
        bids.append(MarketBid(
            role=role,
            robot_id=robot_id,
            task_id=task_id,
            partner_id=partner_id,
            cost=marginal_release,
            completion_time=completion,
            due_time=float(task.due_time),
            status=env.task_state[task_id].status,
            rendezvous_distance=rendezvous,
            waiting_time=waiting,
        ))
    return tuple(bids)


def _bid_by_task(bids: Iterable[MarketBid]) -> Dict[int, List[MarketBid]]:
    grouped: Dict[int, List[MarketBid]] = {}
    for bid in bids:
        grouped.setdefault(bid.task_id, []).append(bid)
    return grouped


def _best_bid_for_role(
    bids: Sequence[MarketBid],
    role: Role,
    task_id: int,
) -> Optional[MarketBid]:
    candidates = [bid for bid in bids if bid.role == role and bid.task_id == task_id]
    if not candidates:
        return None
    return min(candidates, key=lambda bid: (bid.cost, bid.robot_id, bid.partner_id))


def _task_candidates(bids_by_agent: Mapping[AgentKey, Sequence[MarketBid]]) -> Dict[int, List[MarketBid]]:
    grouped: Dict[int, List[MarketBid]] = {}
    for bids in bids_by_agent.values():
        for bid in bids:
            grouped.setdefault(bid.task_id, []).append(bid)
    return grouped


def _deadline_bands(
    env: DecentralizedMRSEnv,
    objective_mode: str,
) -> Dict[int, int]:
    """Group deadlines into common urgency bands without imposing exact EDD.

    The band width is three times the median spacing between known deadlines.
    All market methods share this safety ordering in v4 mode, while their
    method-specific score determines the order inside a band.  Makespan mode
    intentionally disables the deadline bands.
    """
    if objective_mode not in ("v4", "distance_tardiness"):
        return {}
    due_times = sorted(float(task.due_time) for task in env.tasks)
    gaps = [right - left for left, right in zip(due_times, due_times[1:]) if right > left]
    spacing = median(gaps) if gaps else 1.0
    width = max(1.0, 3.0 * spacing)
    base = due_times[0]
    return {
        task_id: int((float(task.due_time) - base) // width)
        for task_id, task in enumerate(env.tasks)
    }


def _select_assignments(
    env: DecentralizedMRSEnv,
    bids_by_agent: Mapping[AgentKey, Sequence[MarketBid]],
    method: str,
    *,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Resolve one distributed auction round from a common current snapshot.

    Every robot can recompute this result from the broadcast bids.  The
    simulator applies the selected awards sequentially only to make state
    mutation deterministic.  No bid includes a future action or a complete
    route, so this remains an event-level decentralized protocol.
    """
    objective_mode = _normalize_objective_mode(objective_mode)
    tasks = _task_candidates(bids_by_agent)
    deadline_bands = _deadline_bands(env, objective_mode)
    task_order = []
    for task_id, bids in tasks.items():
        mbr = _best_bid_for_role(bids, Role.MBR, task_id)
        dor = _best_bid_for_role(bids, Role.DOR, task_id)
        if mbr is None and dor is None:
            continue
        combined = (mbr.cost if mbr is not None else 0.0) + (dor.cost if dor is not None else 0.0)
        if method == "d_murdoch":
            # MURDOCH resolves the announcement using the submitted role bids.
            # D-EDD is the separate deadline-priority control; putting due_time
            # first here would collapse the auction into the same policy.
            earliest_completion = min(
                bid.completion_time for bid in bids if bid.task_id == task_id
            )
            key = (
                deadline_bands.get(task_id, 0),
                combined,
                earliest_completion,
                task_id,
            )
        else:
            raise ValueError(f"unsupported auction method: {method}")
        task_order.append((key, task_id, mbr, dor))
    task_order.sort(key=lambda item: item[0])

    used_agents: set[AgentKey] = set()
    awards: List[Tuple[AgentKey, int]] = []
    for _, task_id, mbr_bid, dor_bid in task_order:
        state = env.task_state[task_id]
        role_winners = []
        if mbr_bid is not None and (Role.MBR, mbr_bid.robot_id) not in used_agents:
            role_winners.append(((Role.MBR, mbr_bid.robot_id), mbr_bid))
        if dor_bid is not None and (Role.DOR, dor_bid.robot_id) not in used_agents:
            role_winners.append(((Role.DOR, dor_bid.robot_id), dor_bid))
        if not role_winners:
            continue

        # Do not create more open tasks than the common environment contract.
        # A complete pair is safe because the second award immediately starts
        # the task; a single award on an empty task consumes one open slot.
        if state.status == TaskStatus.EMPTY and len(role_winners) < 2:
            open_count = sum(item.status == TaskStatus.OPEN for item in env.task_state)
            if open_count >= env.max_open_tasks:
                continue

        # MBR first is only a deterministic application order.  The award
        # itself was computed from the same pre-award snapshot for both roles.
        role_winners.sort(key=lambda item: int(item[0][0]))
        for agent, _ in role_winners:
            if agent in used_agents:
                continue
            used_agents.add(agent)
            awards.append((agent, task_id))
    return tuple(awards)


def _current_pair_offers(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[PairOffer, ...]:
    """Build feasible current pair offers from the broadcast snapshot.

    Empty tasks require both roles to be ready.  An open task has one fixed
    owner, so only its missing role must be ready.  The resulting offers are
    local one-step coalitions; no task sequence is encoded in an offer.
    """
    objective_mode = _normalize_objective_mode(objective_mode)
    available = {
        agent for agent in agents if env.is_available(*agent)
    }
    mbrs = sorted(agent[1] for agent in available if agent[0] == Role.MBR)
    dors = sorted(agent[1] for agent in available if agent[0] == Role.DOR)
    offers: List[PairOffer] = []

    for task_id, state in enumerate(env.task_state):
        if state.status == TaskStatus.EMPTY:
            if not mbrs or not dors:
                continue
            mbr_candidates = mbrs
            dor_candidates = dors
        elif state.status == TaskStatus.OPEN:
            if state.mbr_id is not None and state.dor_id is None:
                mbr_candidates = [state.mbr_id]
                dor_candidates = dors
            elif state.dor_id is not None and state.mbr_id is None:
                mbr_candidates = mbrs
                dor_candidates = [state.dor_id]
            else:
                continue
        else:
            continue

        for mbr_id in mbr_candidates:
            for dor_id in dor_candidates:
                mbr_agent = (Role.MBR, mbr_id)
                dor_agent = (Role.DOR, dor_id)
                if state.status == TaskStatus.EMPTY:
                    if mbr_agent not in available or dor_agent not in available:
                        continue
                    if not env.role_mask(
                        Role.MBR, mbr_id, enforce_open_limit=training_cfm
                    )[task_id]:
                        continue
                    if not env.role_mask(
                        Role.DOR, dor_id, enforce_open_limit=training_cfm
                    )[task_id]:
                        continue
                elif state.mbr_id is not None:
                    if dor_agent not in available or not env.role_mask(
                        Role.DOR, dor_id, enforce_open_limit=training_cfm
                    )[task_id]:
                        continue
                else:
                    if mbr_agent not in available or not env.role_mask(
                        Role.MBR, mbr_id, enforce_open_limit=training_cfm
                    )[task_id]:
                        continue

                values, result = transition_features(env, task_id, mbr_id, dor_id)
                cost, completion, _, waiting, _ = _pair_cost(
                    env,
                    task_id,
                    mbr_id,
                    dor_id,
                    objective_mode=objective_mode,
                )
                offers.append(PairOffer(
                    task_id=task_id,
                    mbr_id=mbr_id,
                    dor_id=dor_id,
                    cost=cost,
                    completion_time=completion,
                    start_time=float(values[1]),
                    waiting_time=waiting,
                    due_time=float(env.tasks[task_id].due_time),
                    status=state.status,
                    mbr_release_time=float(result.mbr_release_time),
                    dor_release_time=float(result.dor_release_time),
                    system_distance=float(result.system_distance),
                ))
    return tuple(offers)


def _apply_pair_offers(
    env: DecentralizedMRSEnv,
    offers: Sequence[PairOffer],
    key,
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Apply a conflict-free consensus ordering to one offer snapshot."""
    used: set[AgentKey] = set()
    claimed_tasks: set[int] = set()
    awards: List[Tuple[AgentKey, int]] = []
    for offer in sorted(offers, key=key):
        task_id = offer.task_id
        if task_id in claimed_tasks:
            continue
        state = env.task_state[task_id]
        mbr_agent = (Role.MBR, offer.mbr_id)
        dor_agent = (Role.DOR, offer.dor_id)
        if state.status == TaskStatus.EMPTY:
            if mbr_agent in used or dor_agent in used:
                continue
            used.update((mbr_agent, dor_agent))
            claimed_tasks.add(task_id)
            # Application order is deterministic; the offer was computed from
            # the same pre-award state for both robots.
            awards.extend(((mbr_agent, task_id), (dor_agent, task_id)))
        elif state.status == TaskStatus.OPEN:
            if state.mbr_id is not None and state.dor_id is None:
                if state.mbr_id != offer.mbr_id or dor_agent in used:
                    continue
                used.add(dor_agent)
                claimed_tasks.add(task_id)
                awards.append((dor_agent, task_id))
            elif state.dor_id is not None and state.mbr_id is None:
                if state.dor_id != offer.dor_id or mbr_agent in used:
                    continue
                used.add(mbr_agent)
                claimed_tasks.add(task_id)
                awards.append((mbr_agent, task_id))
    return tuple(awards)


def _select_pair_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Select collective offers by transport efficiency, then resolve conflicts."""
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            offer.system_distance if objective_mode == "v4" else offer.cost,
            offer.completion_time,
            offer.cost,
            offer.waiting_time,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _best_current_offer_per_task(
    offers: Sequence[PairOffer],
    *,
    objective_mode: str = "makespan",
) -> Tuple[PairOffer, ...]:
    """Keep one earliest-finish offer for every task in the current snapshot."""
    best: Dict[int, PairOffer] = {}
    for offer in offers:
        incumbent = best.get(offer.task_id)
        candidate_key = (
            offer.completion_time if objective_mode == "makespan" else offer.cost,
            offer.waiting_time,
            offer.mbr_id,
            offer.dor_id,
        )
        if incumbent is None or candidate_key < (
            incumbent.completion_time
            if objective_mode == "makespan"
            else incumbent.cost,
            incumbent.waiting_time,
            incumbent.mbr_id,
            incumbent.dor_id,
        ):
            best[offer.task_id] = offer
    return tuple(best.values())


def _select_min_min_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "makespan",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Classic Min-min: assign the task with the smallest best finish time."""
    offers = _best_current_offer_per_task(
        _current_pair_offers(
            env,
            agents,
            training_cfm=training_cfm,
            objective_mode=objective_mode,
        ),
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            offer.completion_time if objective_mode == "makespan" else offer.cost,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_max_min_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "makespan",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Classic Max-min: protect the hardest current task first."""
    offers = _best_current_offer_per_task(
        _current_pair_offers(
            env,
            agents,
            training_cfm=training_cfm,
            objective_mode=objective_mode,
        ),
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            -(
                offer.completion_time
                if objective_mode == "makespan"
                else offer.cost
            ),
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_coalition_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Adapt the T-RO coalition auction to fixed two-robot Marsupial tasks.

    The source auction ranks coalition utilities.  In v4, the current pair
    reports the exact one-task contribution to the frozen J objective.  In
    makespan mode the pair uses the completion-plus-synchronization proxy;
    engineering mode uses the shared 0.9 makespan plus 0.1 carrier-travel-time
    score.  Due dates are intentionally outside both time-based objectives.
    The pair is awarded atomically; no role-level bid is substituted for the
    coalition value.
    """
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            offer.cost
            if objective_mode in (
                "v4",
                "engineering",
                "distance_tardiness",
                "makespan_distance",
            )
            else _coalition_offer_cost(offer),
            offer.completion_time,
            offer.waiting_time,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_min_slack_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Select the current coalition with the smallest predicted slack."""
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            (
                offer.completion_time
                if objective_mode == "makespan"
                else offer.cost
                if objective_mode in ("engineering", "makespan_distance")
                else offer.due_time - offer.completion_time
            ),
            offer.cost,
            offer.waiting_time,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_pi_coupled_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Select pairs using a one-step coupled performance-impact score.

    The published PI-Copt family computes insertion/removal impacts over
    local schedules.  A complete schedule is outside this event protocol, so
    this adaptation uses the current pair offers only.  A pair's impact is its
    immediate v4 cost plus the opportunity loss it imposes on each member
    relative to that member's best current offer.  The pair offers then go
    through one coupled conflict-consensus round.
    """
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    if not offers:
        return tuple()
    best_mbr: Dict[int, float] = {}
    best_dor: Dict[int, float] = {}
    for offer in offers:
        best_mbr[offer.mbr_id] = min(best_mbr.get(offer.mbr_id, float("inf")), offer.cost)
        best_dor[offer.dor_id] = min(best_dor.get(offer.dor_id, float("inf")), offer.cost)

    def impact(offer: PairOffer) -> Tuple[float, float, float]:
        mbr_regret = max(0.0, offer.cost - best_mbr[offer.mbr_id])
        dor_regret = max(0.0, offer.cost - best_dor[offer.dor_id])
        opportunity_loss = mbr_regret + dor_regret
        # The small synchronization term makes the coupled nature explicit
        # while leaving the frozen objective ordering unchanged at normal
        # scales.
        sync = abs(offer.mbr_release_time - offer.dor_release_time)
        return opportunity_loss, sync, offer.cost

    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            *impact(offer),
            offer.completion_time,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_cbta_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Adapt CBTA timetable consensus to simultaneous pair starts.

    The timetable priority is the predicted current pair start time, followed
    by completion and waiting.  It is recomputed after every award round, so it
    never commits a future route or a hidden centralized schedule.
    """
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    return _apply_pair_offers(
        env,
        offers,
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            offer.start_time,
            offer.completion_time,
            offer.waiting_time,
            offer.cost,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _select_group_assignments(
    env: DecentralizedMRSEnv,
    agents: Sequence[AgentKey],
    *,
    training_cfm: bool,
    objective_mode: str = "v4",
) -> Tuple[Tuple[AgentKey, int], ...]:
    """Adapt group-based distributed auctioning to two-member groups.

    Each task first keeps its locally best feasible group offer.  The winning
    task groups then reach a deterministic consensus while respecting one
    current assignment per robot.  This mirrors the paper's group formation
    followed by group assignment without planning a route suffix.
    """
    offers = _current_pair_offers(
        env,
        agents,
        training_cfm=training_cfm,
        objective_mode=objective_mode,
    )
    deadline_bands = _deadline_bands(env, objective_mode)
    best_by_task: Dict[int, PairOffer] = {}
    for offer in offers:
        incumbent = best_by_task.get(offer.task_id)
        if incumbent is None or (
            offer.cost,
            offer.completion_time,
            offer.mbr_id,
            offer.dor_id,
        ) < (
            incumbent.cost,
            incumbent.completion_time,
            incumbent.mbr_id,
            incumbent.dor_id,
        ):
            best_by_task[offer.task_id] = offer
    return _apply_pair_offers(
        env,
        tuple(best_by_task.values()),
        key=lambda offer: (
            deadline_bands.get(offer.task_id, 0),
            offer.waiting_time,
            offer.completion_time,
            offer.cost,
            offer.task_id,
            offer.mbr_id,
            offer.dor_id,
        ),
    )


def _choose_edd(bids: Sequence[MarketBid], *, objective_mode: str = "v4") -> int:
    if not bids:
        raise ValueError("EDD policy received no legal bids")
    objective_mode = _normalize_objective_mode(objective_mode)
    return min(
        bids,
        key=lambda bid: (
            bid.completion_time
            if objective_mode in ("makespan", "engineering", "makespan_distance")
            else bid.due_time,
            bid.cost,
            0 if bid.status == TaskStatus.OPEN else 1,
            bid.task_id,
            bid.partner_id,
        ),
    ).task_id


def _assignment_record(
    agent: AgentKey,
    task_id: int,
    *,
    observation: PolicyObservation,
    origins: Mapping[AgentKey, float],
    observation_time: float,
    state_version: int,
    token_sequence: int,
) -> DecisionRecord:
    role, robot_id = agent
    return DecisionRecord(
        task_features=observation.task_features.clone(),
        agent_features=observation.agent_features.clone(),
        action_mask=observation.action_mask.clone(),
        agent_index=observation.agent_index,
        action=task_id,
        role=role,
        robot_id=robot_id,
        observation_time=observation_time,
        origin_time=origins.get(agent, observation_time),

        state_version=state_version,
        token_sequence=token_sequence,
    )


def run_decentralized_market_episode(
    method: str,
    tasks: Sequence[TaskSpec],
    n_mbr: int,
    n_dor: int,
    speed: float,
    *,
    env_seed: int,
    action_seed: int = 0,
    training_cfm: bool = False,
    max_open_tasks: Optional[int] = None,
    time_limit: Optional[float] = None,
    dock_time: float = 8.0,
    detach_time: float = 8.0,
    objective_mode: str = "v4",
    mbr_speed: Optional[float] = None,
    dor_speed: Optional[float] = None,
    mbr_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    dor_initial_positions: Optional[Sequence[Tuple[float, float]]] = None,
    task_release_times: Optional[Sequence[float]] = None,
    travel_time_factors: Optional[Sequence[float]] = None,
    operation_time_factors: Optional[Sequence[float]] = None,
    execution_time_factors: Optional[Sequence[float]] = None,
) -> EpisodeOutcome:
    """Run one event-level decentralized market episode."""
    if not tasks:
        raise ValueError("at least one task is required")
    method = method.lower()
    objective_mode = _normalize_objective_mode(objective_mode)
    if method not in MARKET_METHODS:
        raise ValueError(f"unsupported decentralized market method: {method}")
    env = DecentralizedMRSEnv(
        tasks,
        n_mbr,
        n_dor,
        speed,
        max_open_tasks=max_open_tasks,
        seed=env_seed,
        time_limit=(
            time_limit
            if time_limit is not None
            else _default_time_limit(tasks, objective_mode)
        ),
        dock_time=dock_time,
        detach_time=detach_time,
        mbr_speed=mbr_speed,
        dor_speed=dor_speed,
        mbr_initial_positions=mbr_initial_positions,
        dor_initial_positions=dor_initial_positions,
        objective_mode=objective_mode,
        task_release_times=task_release_times,
        travel_time_factors=travel_time_factors,
        operation_time_factors=operation_time_factors,
        execution_time_factors=execution_time_factors,
    )
    scales = ObservationScales.for_environment(
        env,
    )
    # action_seed is retained in metadata/API for parity with the other
    # baselines.  Tie-breaking is deterministic by robot/task id.
    _ = int(action_seed)
    protocol = ScalarExecutionProtocol(env)
    ready: List[AgentKey] = list(protocol.initial_decisions())
    deferred: List[AgentKey] = []
    origins: Dict[AgentKey, float] = {agent: env.time for agent in ready}
    records: List[DecisionRecord] = []
    decision_time = 0.0
    deferred_attempts = 0
    failure_reason = ""
    max_decisions = 2 * len(tasks)
    event_steps = 0

    while not env.terminal:
        event_steps += 1
        if event_steps > 4 * len(tasks) + n_mbr + n_dor + 4:
            failure_reason = "event_limit"
            break

        candidates: List[AgentKey] = []
        for agent in deferred + ready:
            if agent not in candidates:
                candidates.append(agent)
        deferred = []
        ready = []
        progress = False

        if method == "d_edd":
            for agent in candidates:
                role, robot_id = agent
                if not env.is_available(role, robot_id):
                    continue
                context = protocol.begin_round(agent, (agent,))
                strategy_started = perf_counter()
                observation = build_policy_observation(
                    env,
                    role,
                    robot_id,
                    enforce_open_limit=training_cfm,
                    scales=scales,
                )
                if not observation.has_legal_action:
                    protocol.add_strategy_time(
                        perf_counter() - strategy_started
                    )
                    protocol.record_noop(context)
                    deferred.append(agent)
                    deferred_attempts += 1
                    continue
                started = perf_counter()
                bids = _build_bids(
                    env,
                    role,
                    robot_id,
                    observation,
                    objective_mode=objective_mode,
                )
                action = _choose_edd(bids, objective_mode=objective_mode)
                elapsed = perf_counter() - started
                decision_time += elapsed
                protocol.add_strategy_time(perf_counter() - strategy_started)
                if observation.action_mask[action]:
                    raise AssertionError("D-EDD selected a masked task")
                protocol.commit(
                    context,
                    (AssignmentClaim(agent, action),),
                    enforce_open_limit=training_cfm,
                )
                records.append(_assignment_record(
                    agent,
                    action,
                    observation=observation,
                    origins=origins,
                    observation_time=context.snapshot.time,
                    state_version=context.snapshot.state_version,
                    token_sequence=context.token.sequence,
                ))
                progress = True
        else:
            # Re-run consensus after each set of awards so that a role which
            # becomes eligible for an open task can immediately participate.
            remaining = list(candidates)
            for _round in range(len(candidates) + 1):
                available = [
                    agent for agent in remaining
                    if env.is_available(*agent)
                ]
                if not available:
                    break
                context = protocol.begin_round(available[0], available)
                strategy_started = perf_counter()
                bids_by_agent: Dict[AgentKey, Tuple[MarketBid, ...]] = {}
                observations_by_agent: Dict[AgentKey, PolicyObservation] = {}
                for agent in available:
                    role, robot_id = agent
                    observation = build_policy_observation(
                        env,
                        role,
                        robot_id,
                        enforce_open_limit=training_cfm,
                        scales=scales,
                    )
                    if observation.has_legal_action:
                        observations_by_agent[agent] = observation
                        bids_by_agent[agent] = _build_bids(
                            env,
                            role,
                            robot_id,
                            observation,
                            objective_mode=objective_mode,
                        )
                    else:
                        deferred.append(agent)
                strategy_elapsed = perf_counter() - strategy_started
                protocol.add_strategy_time(strategy_elapsed)
                protocol.add_messages(len(bids_by_agent))
                coordination_started = perf_counter()
                pair_method = method in {
                    "d_min_min",
                    "d_max_min",
                    "d_collective_auction",
                    "d_coalition_auction",
                    "d_min_slack",
                    "d_pi_coupled",
                    "d_cbta",
                    "d_group_auction",
                }
                if pair_method:
                    if method == "d_min_min":
                        pair_awards = _select_min_min_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_max_min":
                        pair_awards = _select_max_min_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_coalition_auction":
                        pair_awards = _select_coalition_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_cbta":
                        pair_awards = _select_cbta_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_min_slack":
                        pair_awards = _select_min_slack_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_pi_coupled":
                        pair_awards = _select_pi_coupled_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    elif method == "d_group_auction":
                        pair_awards = _select_group_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    else:
                        pair_awards = _select_pair_assignments(
                            env,
                            available,
                            training_cfm=training_cfm,
                            objective_mode=objective_mode,
                        )
                    awards = pair_awards
                    awarded_agents = {agent for agent, _ in awards}
                    awarded_tasks = {task_id for _, task_id in awards}
                    fallback_bids = {}
                    for agent, bids in bids_by_agent.items():
                        if agent in awarded_agents:
                            continue
                        remaining_bids = tuple(
                            bid for bid in bids if bid.task_id not in awarded_tasks
                        )
                        if remaining_bids:
                            fallback_bids[agent] = remaining_bids
                    # A pair is preferred.  If no complementary role is
                    # currently ready, retain a one-sided Contract-Net
                    # fallback so the event simulator can open a task and
                    # wait for its partner.
                    fallback = _select_assignments(
                        env,
                        fallback_bids,
                        "d_murdoch",
                        objective_mode=objective_mode,
                    )
                    awards = tuple(awards) + tuple(fallback)
                else:
                    awards = _select_assignments(
                        env,
                        bids_by_agent,
                        method,
                        objective_mode=objective_mode,
                    )
                coordination_elapsed = perf_counter() - coordination_started
                protocol.add_coordination_time(coordination_elapsed)
                decision_time += strategy_elapsed + coordination_elapsed
                if not awards:
                    protocol.record_noop(context)
                    break
                claims = []
                for agent, task_id in awards:
                    observation = observations_by_agent.get(agent)
                    if observation is None or observation.action_mask[task_id]:
                        raise AssertionError("market consensus produced an illegal award")
                    claims.append(AssignmentClaim(agent, task_id))
                protocol.commit(
                    context,
                    claims,
                    enforce_open_limit=training_cfm,
                )
                assigned_agents = {claim.agent for claim in claims}
                records.extend(
                    _assignment_record(
                        claim.agent,
                        claim.task_id,
                        observation=observations_by_agent[claim.agent],
                        origins=origins,
                        observation_time=context.snapshot.time,
                        state_version=context.snapshot.state_version,
                        token_sequence=context.token.sequence,
                    )
                    for claim in claims
                )
                round_progress = bool(claims)
                progress = progress or round_progress
                remaining = [agent for agent in remaining if agent not in assigned_agents]
                if not round_progress:
                    break
            for agent in remaining:
                if env.is_available(*agent) and agent not in deferred:
                    deferred.append(agent)

        if len(records) > max_decisions:
            failure_reason = "decision_limit"
            break
        if env.done:
            break
        if not env.pending_events:
            if deferred and not progress:
                failure_reason = "deadlock"
            elif not progress:
                failure_reason = "deadlock"
            break

        ready = list(protocol.advance())
        if env.timed_out:
            failure_reason = "timeout"
            break
        for agent in ready:
            origins[agent] = env.time

    if env.done:
        metrics = env.metrics()
        cost = metrics.makespan if objective_mode == "makespan" else metrics.objective
        success = True
        failure_reason = ""
    else:
        metrics = None
        partial = env.time if objective_mode == "makespan" else env.partial_objective
        cost = partial + _failure_penalty(tasks, objective_mode)
        success = False
        if not failure_reason:
            failure_reason = "timeout" if env.timed_out else "deadlock"
    return EpisodeOutcome(
        cost=float(cost),
        reward=-float(cost),
        success=success,
        failure_reason=failure_reason,
        metrics=metrics,
        plan=tuple(env.plan),
        records=tuple(records),
        decision_time_s=decision_time,
        deferred_attempts=deferred_attempts,
        flashforward_decisions=0,
        execution_metrics=protocol.metrics,
        protocol_rounds=protocol.rounds,
    )
