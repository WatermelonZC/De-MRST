"""Batched Torch implementation of the shared Marsupial transition."""

from typing import Tuple

import torch


@torch.jit.script
def update_state_tensors(
    mask: torch.Tensor,
    dor_completion: torch.Tensor,
    mbr_completion: torch.Tensor,
    vehicle_pos: torch.Tensor,
    vehicle_time: torch.Tensor,
    vehicle_coordinates: torch.Tensor,
    source: torch.Tensor,
    destination_with_depot: torch.Tensor,
    handling_time: torch.Tensor,
    pickup_time: torch.Tensor,
    system_distance: torch.Tensor,
    mbr_distance: torch.Tensor,
    dor_distance: torch.Tensor,
    mbr_wait: torch.Tensor,
    dor_wait: torch.Tensor,
    ids: torch.Tensor,
    task_selected: torch.Tensor,
    dor_selected: torch.Tensor,
    mbr_selected: torch.Tensor,
    dock_time: torch.Tensor,
    detach_time: torch.Tensor,
    speed: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply one task assignment to every active batch row."""
    batch = ids.squeeze(1)
    tasks = task_selected
    speeds = speed[batch].reshape(-1)
    if bool(torch.any(speeds <= 0)):
        raise ValueError("speed must be positive")

    new_mask = mask.clone()
    new_dor_completion = dor_completion.clone()
    new_mbr_completion = mbr_completion.clone()
    new_vehicle_time = vehicle_time.clone()
    new_vehicle_pos = vehicle_pos.clone()
    new_vehicle_coordinates = vehicle_coordinates.clone()

    new_mask[ids, dor_selected.unsqueeze(-1), tasks.unsqueeze(-1)] = True
    new_mask[ids, mbr_selected.unsqueeze(-1), tasks.unsqueeze(-1)] = True

    dor_position = vehicle_coordinates[batch, dor_selected]
    mbr_position = vehicle_coordinates[batch, mbr_selected]
    pickup_position = source[batch, tasks]
    delivery_position = destination_with_depot[batch, tasks]

    rendezvous_distance = torch.sum(torch.abs(dor_position - mbr_position), dim=1)
    mbr_arrival = (
        vehicle_time[batch, mbr_selected] + rendezvous_distance / speeds
    )
    synchronized = torch.maximum(vehicle_time[batch, dor_selected], mbr_arrival)
    dock_finish = synchronized + dock_time[batch].reshape(-1)

    rendezvous_to_source = torch.sum(
        torch.abs(pickup_position - dor_position), dim=1
    )
    source_to_destination = torch.sum(
        torch.abs(delivery_position - pickup_position), dim=1
    )
    destination_arrival = (
        dock_finish
        + rendezvous_to_source / speeds
        + pickup_time[batch, tasks].reshape(-1)
        + source_to_destination / speeds
    )
    mbr_release = destination_arrival + detach_time[batch].reshape(-1)
    dor_release = mbr_release + handling_time[batch, tasks].reshape(-1)

    new_dor_completion[batch, tasks] = dor_release
    new_mbr_completion[batch, tasks] = mbr_release
    new_vehicle_time[batch, dor_selected] = dor_release
    new_vehicle_time[batch, mbr_selected] = mbr_release
    new_vehicle_pos[batch, dor_selected] = tasks.unsqueeze(-1)
    new_vehicle_pos[batch, mbr_selected] = tasks.unsqueeze(-1)
    new_vehicle_coordinates[batch, dor_selected] = delivery_position
    new_vehicle_coordinates[batch, mbr_selected] = delivery_position

    combined_distance = rendezvous_to_source + source_to_destination
    step_system_distance = rendezvous_distance + combined_distance
    new_system_distance = system_distance + step_system_distance
    new_mbr_distance = mbr_distance + step_system_distance
    new_dor_distance = dor_distance + combined_distance
    new_mbr_wait = mbr_wait + torch.clamp(
        vehicle_time[batch, dor_selected] - mbr_arrival, min=0.0
    )
    new_dor_wait = dor_wait + torch.clamp(
        mbr_arrival - vehicle_time[batch, dor_selected], min=0.0
    )

    return (
        new_mask,
        new_dor_completion,
        new_mbr_completion,
        new_vehicle_time,
        new_vehicle_pos,
        new_system_distance,
        new_mbr_distance,
        new_dor_distance,
        new_mbr_wait,
        new_dor_wait,
        tasks.unsqueeze(-1),
        new_vehicle_coordinates,
    )
