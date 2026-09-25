"""Data conversion and generation for the centralized attention baseline."""

import pandas as pd
import torch

MAX_VALUES = {
    'location': 100.0,
    'time': 2500.0,
    'tau_p': 60.0,
    'tau_ad': 15.0,
    'velocity': 5.0,
    'tau_h': 100.0,
}

def normalize(tensor):
    time_min = tensor.min(dim=1, keepdim=True)[0]  # 每个 sample 的最小时间
    time_max = tensor.max(dim=1, keepdim=True)[0]
    time_norm = (tensor - time_min) / (time_max - time_min + 1e-8)
    return time_norm

def read_excel_to_tensor(input_file="tasks.xlsx", designated_driver=False):
    # 读取任务数据
    task_df = pd.read_excel(input_file, sheet_name='Tasks', engine='openpyxl')

    # 重建sor_loc和tar_loc
    sor_loc = torch.tensor(
        task_df[['source_x', 'source_y']].values,
        dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 2)

    tar_loc = torch.tensor(
        task_df[['destination_x', 'destination_y']].values,
        dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 2)

    designated_time = torch.tensor(
        task_df[['t_delivery']].values,
        dtype=torch.float32
    ).unsqueeze(0)

    # 读取车辆数据
    vehicle_df = pd.read_excel(input_file, sheet_name='Vehicles')

    if len(vehicle_df) < 2:
        raise ValueError("Vehicles sheet must contain MBR and DOR rows")
    count_column = vehicle_df.columns[1]
    mom_size = int(vehicle_df.iloc[0][count_column])
    sub_size = int(vehicle_df.iloc[1][count_column])

    tau_a = float(vehicle_df.iloc[0]['tau_a'])
    tau_d = float(vehicle_df.iloc[0]['tau_d'])
    tau_p = float(vehicle_df.iloc[0]['tau_p'])
    v = float(vehicle_df.iloc[0].get('objective_speed', vehicle_df.iloc[0]['v']))
    v_mbr = float(vehicle_df.iloc[0]['v'])
    v_dor_value = float(vehicle_df.iloc[1]['v'])
    v_dor = v_dor_value if torch.isfinite(torch.tensor(v_dor_value)) else 2.4
    params = torch.tensor([[tau_a, tau_d, tau_p]])
    tau_h_tasks = torch.tensor(
        task_df[['t_operation']].values, dtype=torch.float32
    ).unsqueeze(0)
    tau_p_tasks = (
        torch.tensor(task_df[['t_picking']].values, dtype=torch.float32).unsqueeze(0)
        if 't_picking' in task_df.columns
        else torch.full_like(tau_h_tasks, float(tau_p))
    )

    # 重建原始Tensor
    original_data = {
        'sor_loc': sor_loc,
        'tar_loc': tar_loc,
        'mom_size': mom_size,
        'sub_size': sub_size,
        'params': params,
        'designated_time': designated_time,
        'v': torch.tensor([[v]], dtype=torch.float32),
        'v_mbr': torch.tensor([[v_mbr]], dtype=torch.float32),
        'v_dor': torch.tensor([[v_dor]], dtype=torch.float32),
        'tau_h': tau_h_tasks,
        'tau_p': tau_p_tasks
    }

    try:
        positions_df = pd.read_excel(
            input_file, sheet_name='InitialPositions', engine='openpyxl'
        )
    except ValueError:
        positions_df = None
    if positions_df is not None:
        mbr_positions = positions_df[
            positions_df['role'].astype(str).str.upper() == 'MBR'
        ].sort_values('robot_id')[['x', 'y']].values
        dor_positions = positions_df[
            positions_df['role'].astype(str).str.upper() == 'DOR'
        ].sort_values('robot_id')[['x', 'y']].values
        if len(mbr_positions) != mom_size or len(dor_positions) != sub_size:
            raise ValueError('InitialPositions sheet does not match fleet sizes')
        original_data['mbr_initial_positions'] = torch.tensor(
            mbr_positions, dtype=torch.float32
        ).unsqueeze(0)
        original_data['dor_initial_positions'] = torch.tensor(
            dor_positions, dtype=torch.float32
        ).unsqueeze(0)

    return original_data

def collate_to_batch(list_of_dicts):
    """
    将一个字典列表（每个字典是一个样本）转换为一个批处理字典。
    """
    # 如果列表为空，返回空字典
    if not list_of_dicts:
        return {}

    # 获取第一个字典的所有键，作为批处理字典的键
    keys = list_of_dicts[0].keys()
    batched_dict = {}

    # 遍历所有键
    for key in keys:
        # 提取当前键在所有字典中的值，形成一个值的列表
        list_of_values = [d[key] for d in list_of_dicts]

        # 检查值是否为张量，如果不是（例如 'sub_size'），则先转换为张量
        if not isinstance(list_of_values[0], torch.Tensor):
            list_of_values = [torch.tensor(item) for item in list_of_values]

        # 使用 torch.stack 将值列表沿着新的维度（dim=0）堆叠起来
        # 这会创建批处理维度
        batched_dict[key] = torch.stack(list_of_values, dim=0)

    return batched_dict

def generate_hca_data_robustness(num_samples=1, size=20, mom_size=1, sub_size=1,
                      distribution_type='uniform', real_data_path=None, normalize=False, isEval=False):
    global normalization_tensor
    def _create_points_programmatically(dist_type):
        """
        Generates 2D points based on a specified distribution type,
        ensuring they are within a [0, 1] unit square before scaling.
        """
        num_total_points = num_samples * size
        points_unit_scale = None

        if dist_type == 'uniform':
            points_unit_scale = torch.rand(num_samples, size, 2)
            # Convert from unit scale [0, 1] to physical scale [0, 100] and round to nearest 5
            points_physical = points_unit_scale * 100.0
            return torch.round(points_physical / 5.0) * 5.0

        elif dist_type == 'triangle':
            u = torch.rand(num_samples, size, 2)
            x = torch.min(u, dim=-1).values.unsqueeze(-1)
            y = torch.max(u, dim=-1).values.unsqueeze(-1)
            points_unit_scale = torch.cat([x, y], dim=-1)

        elif dist_type == 'ellipse':
            h, k = 0.5, 0.5  # Center
            a, b = 0.5, 0.25  # Radii
            r = torch.sqrt(torch.rand(num_total_points))
            theta = 2 * torch.pi * torch.rand(num_total_points)
            x = h + a * r * torch.cos(theta)
            y = k + b * r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'semicircle':
            h, k = 0.5, 0.0  # Center at the bottom edge
            radius = 0.5
            r = radius * torch.sqrt(torch.rand(num_total_points))
            theta = torch.pi * torch.rand(num_total_points)  # Theta from 0 to pi
            x = h + r * torch.cos(theta)
            y = k + r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'annulus':
            h, k = 0.5, 0.5  # Center
            r_outer, r_inner = 0.5, 0.2
            theta = 2 * torch.pi * torch.rand(num_total_points)
            r = torch.sqrt(torch.rand(num_total_points) * (r_outer ** 2 - r_inner ** 2) + r_inner ** 2)
            x = h + r * torch.cos(theta)
            y = k + r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'gaussian_mixture':
            means = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
            stds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])

            # Generate all points at once
            cluster_assignments = torch.randint(0, 2, (num_total_points,))
            points = torch.normal(mean=means[cluster_assignments], std=stds[cluster_assignments])

            # Resample points that are out of bounds [0, 1]
            in_bounds = (points >= 0) & (points <= 1)
            all_in_bounds = torch.all(in_bounds, dim=1)

            while not torch.all(all_in_bounds):
                out_of_bounds_indices = torch.where(~all_in_bounds)[0]
                n_resample = len(out_of_bounds_indices)
                new_assignments = torch.randint(0, 2, (n_resample,))
                new_points = torch.normal(mean=means[new_assignments], std=stds[new_assignments])
                points[out_of_bounds_indices] = new_points

                # Recheck bounds
                in_bounds = (points >= 0) & (points <= 1)
                all_in_bounds = torch.all(in_bounds, dim=1)

            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'spiral':
            noise = 0.02
            theta = torch.sqrt(torch.rand(num_total_points)) * 2.5 * 2 * torch.pi
            r_norm = theta / (2.5 * 2 * torch.pi)
            x = 0.5 + (r_norm * torch.cos(theta)) * 0.45 + torch.randn(num_total_points) * noise
            y = 0.5 + (r_norm * torch.sin(theta)) * 0.45 + torch.randn(num_total_points) * noise
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        else:
            raise ValueError(f"无效的程序化 distribution_type: {dist_type}")

        # As a safety measure, clamp all points to be within [0, 1] before scaling
        points_unit_scale = torch.clamp(points_unit_scale, 0.0, 1.0)
        # Convert from unit scale [0, 1] to physical scale [0, 100] and round to nearest 5
        points_physical = points_unit_scale * 100.0
        return points_physical

    if distribution_type == 'real_data':
        if real_data_path is None:
            raise ValueError("使用 'real_data' 分布时必须提供 real_data_path。")
        try:
            df = pd.read_excel(real_data_path)
        except Exception as e:
            raise RuntimeError(f"读取真实数据文件时出错: {e}")

        sor_loc = torch.tensor(df[['source_x', 'source_y']].values, dtype=torch.float32).unsqueeze(0)
        tar_loc = torch.tensor(df[['destination_x', 'destination_y']].values, dtype=torch.float32).unsqueeze(0)

        # 2. Convert time columns to a tensor of shape (num_rows, 1)
        designated_time = torch.tensor(df['t_delivery'].values, dtype=torch.float32).reshape(1, -1, 1)
        tau_h = torch.tensor(df['t_operation'].values, dtype=torch.float32).reshape(1, -1, 1)
        tau_p = (
            torch.tensor(df['t_picking'].values, dtype=torch.float32).reshape(1, -1, 1)
            if 't_picking' in df.columns
            else torch.full_like(tau_h, 30.0)
        )

    else:
        # --- 程序化生成数据 ---
        sor_loc = _create_points_programmatically(distribution_type)
        tar_loc = _create_points_programmatically(distribution_type)

        ddl_base = torch.ones(num_samples, 1, 1) * 300
        task_indices = torch.arange(size, dtype=torch.float32).reshape(1, size, 1)
        time_noise = (torch.rand(num_samples, size, 1) * 2 - 1) * 40.0
        designated_time = torch.round(ddl_base + task_indices * 40.0 + time_noise)
        tau_h = 60.0 + 5.0 * torch.randint(0, 9, (num_samples, size, 1), dtype=torch.float32)
        tau_p = torch.full((num_samples, size, 1), 30.0, dtype=torch.float32)

    MAX_VALUES['time'] = 300 + size * 40.0 + 40.0
    normalization_tensor = torch.tensor([
        MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 sor_loc
        MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 tar_loc
        MAX_VALUES['time'],  # 对应 designated_time
        MAX_VALUES['tau_h'],  # 对应 tau_h
        MAX_VALUES['tau_p']  # 对应 tau_p
    ])

    tau_adp_fixed = torch.tensor([8.0, 8.0, 30.0])
    param_tensor = tau_adp_fixed.unsqueeze(0).repeat(num_samples, 1)

    v_fixed = torch.full((num_samples, 1), 1.8)

    # --- 组装成包含物理单位的数据集 ---
    physical_data = [
        {
            'sor_loc': sor_loc[i], 'tar_loc': tar_loc[i], 'designated_time': designated_time[i],
            'sub_size': sub_size, 'mom_size': mom_size,
            'params': param_tensor[i],  # 每个样本的params都是 [8, 8, 30]
            'v': v_fixed[i], 'tau_h': tau_h[i], 'tau_p': tau_p[i]
        }
        for i in range(num_samples)
    ]

    if not normalize:
        if isEval:
            physical_data = collate_to_batch(physical_data)
        return physical_data
    else:
        normalized_data = []
        max_tau = torch.tensor([MAX_VALUES['tau_ad'], MAX_VALUES['tau_ad'], MAX_VALUES['tau_p']])
        for sample in physical_data:
            normalized_sample = {
                'sor_loc': sample['sor_loc'] / MAX_VALUES['location'],
                'tar_loc': sample['tar_loc'] / MAX_VALUES['location'],
                'designated_time': sample['designated_time'] / MAX_VALUES['time'],
                'sub_size': sample['sub_size'], 'mom_size': sample['mom_size'],
                'params': sample['params'] / max_tau,
                'v': sample['v'] / MAX_VALUES['velocity'],
                'tau_h': sample['tau_h'] / MAX_VALUES['tau_h'],
                'tau_p': sample['tau_p'] / MAX_VALUES['tau_p']
            }
            normalized_data.append(normalized_sample)
        return collate_to_batch(normalized_data)
