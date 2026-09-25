

class Config:
    M_parent = 4               # 母车数
    M_child = 8                # 子车数
    M = M_parent + M_child     # 子母车总数
    M_parent_list = list(range(1, M_parent + 1))
    M_child_list = list(range(M_parent + 1, M_parent + M_child + 1))
    V = None                      # 车速必须由实例输入设置，不再使用全局默认值
    objective_speed = None
    objective_mode = "v4"
    T_couple = 8.0            # 子车和母车耦合所需时间
    T_decouple = 8.0           # 子车和母车解耦所需时间
    T_load = 30.0             # 取货时间
    weight = 0.05
    Graph_size = 0

    # 初始位置
#     M_position_map = {1: [0.750784039, 0.964488745], 2: [0.793587446,0.339022696
# ], 3: [0.935249805,0.466492653
# ], 4: [0.729373157,0.595317423],
#                       5: [0.126288295, 0.626656532
# ], 6: [0.729373157,0.595317423], 7: [100, 25], 8: [100, 75], 9: [50, 15], 10: [0, 65], 11: [100, 75],
#                       12: [50, 20], 13: [0, 85], 14: [50, 10], 15: [0, 55], 16: [0, 10], 17: [100, 20], 18: [50, 65],
#                       19: [0, 10], 20: [50, 90]}

    M_position_map = {i : [0,0] for i in range(1, 20)}

    @classmethod
    def configure(
        cls,
        mbr_count,
        dor_count,
        speed,
        dock_time,
        detach_time,
        pickup_time,
        mbr_initial_positions=None,
        dor_initial_positions=None,
        mbr_speed=None,
        objective_mode=None,
    ):
        if speed <= 0:
            raise ValueError("Instance speed must be positive")
        cls.M_parent = int(mbr_count)
        cls.M_child = int(dor_count)
        cls.M = cls.M_parent + cls.M_child
        cls.M_parent_list = list(range(1, cls.M_parent + 1))
        cls.M_child_list = list(range(cls.M_parent + 1, cls.M + 1))
        cls.V = float(speed if mbr_speed is None else mbr_speed)
        cls.objective_speed = float(speed)
        if objective_mode is not None:
            objective_mode = str(objective_mode).lower()
            if objective_mode not in (
                "v4", "makespan", "engineering", "distance_tardiness",
                "makespan_distance",
            ):
                raise ValueError(
                    "objective_mode must be 'v4', 'makespan', 'engineering', or "
                    "'distance_tardiness', or 'makespan_distance'"
                )
            cls.objective_mode = objective_mode
        cls.T_couple = float(dock_time)
        cls.T_decouple = float(detach_time)
        cls.T_load = float(pickup_time)
        cls.M_position_map = {index: [0.0, 0.0] for index in range(1, cls.M + 1)}
        if mbr_initial_positions is not None:
            if len(mbr_initial_positions) != cls.M_parent:
                raise ValueError("MBR initial positions must match fleet size")
            for index, position in enumerate(mbr_initial_positions, start=1):
                cls.M_position_map[index] = list(position)
        if dor_initial_positions is not None:
            if len(dor_initial_positions) != cls.M_child:
                raise ValueError("DOR initial positions must match fleet size")
            for offset, position in enumerate(dor_initial_positions, start=cls.M_parent + 1):
                cls.M_position_map[offset] = list(position)
#     M_position_map = {1: [0, 5], 2: [100, 65], 3: [100, 5], 4: [0, 25],
#                       5: [0, 75], 6: [50, 25], 7: [100, 25], 8: [100, 75], 9: [50, 15], 10: [0, 65], 11: [100, 75],
#                       12: [50, 20], 13: [0, 85], 14: [50, 10], 15: [0, 55], 16: [0, 10], 17: [100, 20], 18: [50, 65],
#                       19: [0, 10], 20: [50, 90]}
