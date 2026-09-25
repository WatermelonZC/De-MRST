"""Sampling helper for the centralized attention baseline."""

import time

import torch
import torch.nn.functional as F

from mrs.core import objective_tensor
from problems.hc.problem_hca import HCA


def _repeat_batch(value, repeats, device, batch_size):
    if isinstance(value, dict):
        return {key: _repeat_batch(item, repeats, device, batch_size)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_repeat_batch(item, repeats, device, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_batch(item, repeats, device, batch_size) for item in value)
    if isinstance(value, int):
        return torch.full((batch_size * repeats, 1), value, device=device)
    return value[None, ...].expand(repeats, *value.size()).contiguous().view(
        -1, *value.size()[1:]
    )


def sample_many_hc(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1,
                   attention_model=None):
    """Return the best sampled C-AM plan and its objective."""
    if attention_model is None or not attention_model.is_hca:
        raise ValueError("the C-AM sampling helper requires an HCA model")
    started = time.time()
    embeddings, _, global_embedding = input[1]
    nodes = torch.cat((embeddings, global_embedding), dim=1)
    if "mbr_initial_positions" in input[0] and "dor_initial_positions" in input[0]:
        initial_positions = torch.cat(
            (input[0]["dor_initial_positions"], input[0]["mbr_initial_positions"]),
            dim=1,
        ) / 100.0
        nodes = torch.cat(
            (nodes, attention_model.initial_position_embed(initial_positions)), dim=1
        )
    attention_model.embeddings_with_depot = nodes.repeat(batch_rep, 1, 1)
    batch_input = (input[0], input[1][0])
    batch_input = _repeat_batch(
        batch_input, batch_rep, batch_input[1].device, batch_input[1].size(0)
    )
    costs = []
    plans = []
    for _ in range(iter_rep):
        state = HCA.make_state(batch_input[0])
        _, plan, state = inner_func(batch_input + (state,))
        distance, delay = state.get_costs(batch_input[0], plan)[0]
        cost = objective_tensor(
            delay, distance, batch_input[0]["sor_loc"].size(1), batch_input[0]["v"]
        )
        vehicle_size = (
            batch_input[0]["sub_size"][0].item()
            + batch_input[0]["mom_size"][0].item()
        )
        sequence = HCA.process_sequences_torch(plan, vehicle_size)
        costs.append(cost.view(batch_rep, -1).t())
        plans.append(
            sequence.view(batch_rep, -1, sequence.size(1), sequence.size(-1))
            .transpose(0, 1)
        )
    max_length = max(plan.size(1) for plan in plans)
    plans = torch.cat(
        [F.pad(plan, (0, max_length - plan.size(1))) for plan in plans], dim=1
    )
    costs = torch.cat(costs, dim=1)
    best_costs, indices = costs.min(dim=-1)
    best_plans = plans[
        torch.arange(plans.size(0), out=indices.new()), indices
    ]
    report = [
        best_plans.tolist(), distance.tolist(), delay.tolist(),
        time.time() - started, delay[indices].item(), distance[indices].item(),
        delay.mean().item(), distance.mean().item(), costs.mean().item(),
        best_costs.item(),
    ]
    return best_plans.tolist(), best_costs.item(), report
