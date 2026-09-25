"""Task state and action conversion for the centralized AM baseline."""

import torch

from problems.centralized_mrs.state import CentralizedMRSState


class CentralizedMRSProblem:
    NAME = "centralized_mrs"

    @staticmethod
    def get_costs(dataset, pi):
        state = CentralizedMRSState.initialize(dataset)
        for step in range(pi.size(1)):
            state = state.update(pi[:, step, 0], pi[:, step, 1], pi[:, step, 2])
        return state.get_costs(dataset, pi)

    @staticmethod
    def make_state(*args, **kwargs):
        return CentralizedMRSState.initialize(*args, **kwargs)

    @staticmethod
    def process_sequences_torch(sequences, vehicle_size=5):
        """
        sequences: Tensor of shape (batch_size, n_step)
        task_size: The number of tasks
        vehicle_size: The number of vehicles
        """
        batch_size, n_step, _ = sequences.shape
        # 1. 计算任务索引和车辆索引
        tasks = sequences[:, :, 0].repeat(1, 1, 2)  # 任务索引 (batch_size, n_step)
        vehicle_subs = sequences[:, :, 1]
        vehicle_moms = sequences[:, :, 2]
        # vehicles = sequences // task_size  # 车辆索引 (batch_size, n_step)

        # 2. 创建存放任务的张量，初始化为 -1
        result = torch.full((batch_size, vehicle_size, n_step), -1, dtype=torch.long, device=sequences.device)
        vehicle_task = torch.cat((vehicle_subs, vehicle_moms), 1)
        # 3. 计算 batch 维度索引
        n_step = n_step * 2
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_step)

        # 4. 计算 task_positions (每辆车已分配的任务索引)
        seq1 = vehicle_task.unsqueeze(2)  # (B, N, 1)
        seq2 = vehicle_task.unsqueeze(1)  # (B, 1, N)

        # 构建相等矩阵 (B, N, N)：对于每个 (b, i, j)，判断第 b 个 batch 的第 i 和第 j 个元素是否相等
        equal_matrix = (seq1 == seq2).long()

        # 仅保留下三角（包含对角线），确保只统计前面（含当前）位置的重复项
        tril_mask = torch.tril(torch.ones((n_step, n_step), device=vehicle_task.device)).unsqueeze(0)  # (1, N, N)
        tril_equal = equal_matrix * tril_mask  # (B, N, N)

        # 对每个时间步 i，统计有多少个 j≤i 满足 seq[i] == seq[j]，再 -1 得到编号
        vehicle_counts = tril_equal.sum(dim=2) - 1  # (B, N)

        result[batch_indices, vehicle_task, vehicle_counts.long()] = tasks

        return result
