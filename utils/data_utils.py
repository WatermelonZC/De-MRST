"""Tensor conversion for the centralized attention baseline."""

import pandas as pd
import torch

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
