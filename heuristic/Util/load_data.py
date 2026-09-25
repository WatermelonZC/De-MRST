import numpy as np
import pandas as pd

from heuristic.Util.Config import Config


class HeuristicInstance(list):
    """List-compatible task rows with immutable physical instance metadata."""

    def __init__(
        self,
        rows,
        mbr_count,
        dor_count,
        speed,
        dock_time,
        detach_time,
        pickup_time,
        mbr_speed=None,
        dor_speed=None,
        mbr_initial_positions=None,
        dor_initial_positions=None,
    ):
        super().__init__(rows)
        self.mbr_count = int(mbr_count)
        self.dor_count = int(dor_count)
        self.speed = float(speed)
        self.dock_time = float(dock_time)
        self.detach_time = float(detach_time)
        self.pickup_time = float(pickup_time)
        self.mbr_speed = float(speed if mbr_speed is None else mbr_speed)
        self.dor_speed = float(self.mbr_speed if dor_speed is None else dor_speed)
        self.mbr_initial_positions = tuple(mbr_initial_positions or ())
        self.dor_initial_positions = tuple(dor_initial_positions or ())


def read_excel(file_name):
    task_df = pd.read_excel(file_name, sheet_name="Tasks")
    vehicle_df = pd.read_excel(file_name, sheet_name="Vehicles")
    if len(vehicle_df) < 2:
        raise ValueError("Vehicles sheet must contain MBR and DOR rows")

    count_column = vehicle_df.columns[1]
    required = ["tau_a", "tau_d", "tau_p", "v"]
    missing = [column for column in required if column not in vehicle_df.columns]
    if missing:
        raise ValueError(f"Vehicles sheet is missing columns: {missing}")

    mbr_count = int(vehicle_df.iloc[0][count_column])
    dor_count = int(vehicle_df.iloc[1][count_column])
    dock_time = float(vehicle_df.iloc[0]["tau_a"])
    detach_time = float(vehicle_df.iloc[0]["tau_d"])
    pickup_time = float(vehicle_df.iloc[0]["tau_p"])
    speed = float(vehicle_df.iloc[0].get("objective_speed", vehicle_df.iloc[0]["v"]))
    mbr_speed = float(vehicle_df.iloc[0]["v"])
    dor_speed_value = float(vehicle_df.iloc[1]["v"])
    dor_speed = dor_speed_value if np.isfinite(dor_speed_value) else 2.4
    mbr_positions = []
    dor_positions = []
    try:
        positions_df = pd.read_excel(file_name, sheet_name="InitialPositions")
    except ValueError:
        positions_df = None
    if positions_df is not None:
        for _, row in positions_df.iterrows():
            position = (float(row["x"]), float(row["y"]))
            if str(row["role"]).upper() == "MBR":
                mbr_positions.append(position)
            elif str(row["role"]).upper() == "DOR":
                dor_positions.append(position)
    Config.configure(
        mbr_count,
        dor_count,
        speed,
        dock_time,
        detach_time,
        pickup_time,
        mbr_initial_positions=mbr_positions or None,
        dor_initial_positions=dor_positions or None,
        mbr_speed=mbr_speed,
    )
    Config.Graph_size = len(task_df)
    return HeuristicInstance(
        task_df.values.tolist(), mbr_count, dor_count, speed,
        dock_time, detach_time, pickup_time,
        mbr_speed=mbr_speed,
        dor_speed=dor_speed,
        mbr_initial_positions=mbr_positions or None,
        dor_initial_positions=dor_positions or None,
    )



def toTensor(data):
    coords = np.array([item[1:5] for item in data])  # shape: (20, 4)

    # 计算每列的最小值和最大值
    min_vals = coords.min(axis=0)  # [x_min, y_min, x2_min, y2_min]
    max_vals = coords.max(axis=0)  # [x_max, y_max, x2_max, y2_max]

    # 归一化函数
    def normalize(coords, min_vals, max_vals):
        return (coords - min_vals) / (max_vals - min_vals)

    # 应用归一化
    norm_coords = normalize(coords, min_vals, max_vals)

    # 将归一化后的坐标替换回原数据
    result = data.copy()
    for i in range(len(result)):
        result[i][1:5] = norm_coords[i]

    print(result)
