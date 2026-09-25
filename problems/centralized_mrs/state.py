import torch
from typing import NamedTuple

from torch import Tensor

from mrs.core import DEFAULT_DOR_SPEED, DEFAULT_MBR_SPEED
from mrs.torch_core import update_state_tensors
from utils.data_utils import normalize


class CentralizedMRSState(NamedTuple):
    # Fixed input
    # 任务起始点
    sor_loc: torch.Tensor
    # 任务终止点
    tar_loc: torch.Tensor
    # 时间速度等参数
    params: torch.Tensor
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    i: torch.Tensor  # Keeps track of step
    vehicle_size: int
    # lengths: torch.Tensor
    sequence: list
    mask: torch.Tensor
    vehicle_pos: torch.Tensor
    vehicle_coordinates: torch.Tensor
    # 上一次任务索引
    last_task_indices: torch.Tensor
    sub_vehicle_size: int
    mom_vehicle_size: int
    order_size: int
    vehicle_time: torch.Tensor
    ksi_s: torch.Tensor
    ksi_c: torch.Tensor
    T_s: torch.Tensor
    T_f_s: torch.Tensor
    T_f_c: torch.Tensor
    ksi_max: torch.Tensor
    tau_h: torch.Tensor
    tau_p: torch.Tensor
    speed: torch.Tensor
    mbr_speed: torch.Tensor
    dor_speed: torch.Tensor
    # cost
    travel_cost: torch.Tensor
    mbr_distance: torch.Tensor
    dor_distance: torch.Tensor
    mbr_wait: torch.Tensor
    dor_wait: torch.Tensor
    time_cost: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
        )

    @staticmethod
    def initialize(input):
        sor_loc = input['sor_loc']
        tar_loc = torch.cat((input['tar_loc'], torch.zeros(input['tar_loc'].size(0), 1, 2, dtype=torch.float32, device=input['tar_loc'].device)), dim=1)
        sub_vehicle_size = input['sub_size'][0].item() if isinstance(input['sub_size'], Tensor) else input['sub_size']
        mom_vehicle_size = input['mom_size'][0].item() if isinstance(input['mom_size'], Tensor) else input['mom_size']
        params = input['params']
        tau_h = input['tau_h']
        tau_p = input.get('tau_p')
        if tau_p is None:
            tau_p = params[:, 2].view(-1, 1, 1).expand_as(tau_h)
        speed = input.get('v')
        if speed is None:
            raise KeyError("Every CentralizedMRSProblem instance must provide input['v']")
        if speed.ndim == 1:
            speed = speed.unsqueeze(-1)
        mbr_speed = input.get(
            'v_mbr', torch.full_like(speed, DEFAULT_MBR_SPEED)
        )
        dor_speed = input.get(
            'v_dor', torch.full_like(speed, DEFAULT_DOR_SPEED)
        )
        if mbr_speed.ndim == 1:
            mbr_speed = mbr_speed.unsqueeze(-1)
        if dor_speed.ndim == 1:
            dor_speed = dor_speed.unsqueeze(-1)
        vehicle_size = sub_vehicle_size + mom_vehicle_size
        batch_size = tar_loc.size(0)
        n_loc = sor_loc.size(1)
        device = tar_loc.device
        # all_loc = torch.cat((sor_loc, tar_loc, torch.zeros(batch_size, 1, 2, dtype=torch.float32, device=device)), dim=1)
        # lock_matrix = torch.zeros(batch_size, n_loc, n_loc, dtype=torch.bool, device=device)
        last_task_indices = torch.full((batch_size, 1), -1, dtype=torch.long, device=device, requires_grad=False)
        ksi_s = torch.full((batch_size, n_loc), -1, dtype=torch.float, device=device, requires_grad=False)
        ksi_c = torch.full((batch_size, n_loc), -1, dtype=torch.float, device=device, requires_grad=False)
        T_s = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        T_f_s = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        T_f_c = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        ksi_max = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        vehicle_time = torch.full((batch_size, vehicle_size), 0, dtype=torch.float, device=device, requires_grad=False)
        last_task_indices.fill_(n_loc)
        mask = torch.zeros(batch_size, vehicle_size, n_loc, dtype=torch.bool, device=device, requires_grad=False)
        mbr_initial_positions = input.get(
            'mbr_initial_positions',
            torch.zeros(batch_size, mom_vehicle_size, 2, dtype=torch.float32, device=device),
        )
        dor_initial_positions = input.get(
            'dor_initial_positions',
            torch.zeros(batch_size, sub_vehicle_size, 2, dtype=torch.float32, device=device),
        )
        vehicle_coordinates = torch.cat(
            (dor_initial_positions, mbr_initial_positions), dim=1
        ).to(dtype=torch.float32)
        vehicle_pos = torch.empty(batch_size, vehicle_size, 1, dtype=torch.int64, device=device, requires_grad=False)
        if 'mbr_initial_positions' in input or 'dor_initial_positions' in input:
            base_index = n_loc + 1
            dor_indices = torch.arange(
                base_index, base_index + sub_vehicle_size, dtype=torch.int64, device=device
            )
            mbr_indices = torch.arange(
                base_index + sub_vehicle_size,
                base_index + sub_vehicle_size + mom_vehicle_size,
                dtype=torch.int64,
                device=device,
            )
            vehicle_pos[:, :sub_vehicle_size, 0] = dor_indices
            vehicle_pos[:, sub_vehicle_size:, 0] = mbr_indices
        else:
            vehicle_pos.fill_(n_loc)
        travel_cost = torch.zeros(batch_size, device=device, requires_grad=False)
        mbr_distance = torch.zeros(batch_size, device=device, requires_grad=False)
        dor_distance = torch.zeros(batch_size, device=device, requires_grad=False)
        mbr_wait = torch.zeros(batch_size, device=device, requires_grad=False)
        dor_wait = torch.zeros(batch_size, device=device, requires_grad=False)
        time_cost = torch.zeros(batch_size, device=device, requires_grad=False)
        # assert vehicle_size < 2 * sub_vehicle_size
        return CentralizedMRSState(
            sor_loc=sor_loc,
            tar_loc=tar_loc,
            params=params,
            ids=torch.arange(batch_size, dtype=torch.long, device=device)[:, None],  # (batch_size, 1),  # Add steps dimension
            # lengths=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(batch_size, 1, dtype=torch.int64, device=device),  # Vector with length num_steps
            vehicle_size=vehicle_size,
            sequence=[],
            # lock_matrix=lock_matrix,
            mask=mask,
            vehicle_pos=vehicle_pos,
            vehicle_coordinates=vehicle_coordinates,
            last_task_indices=last_task_indices,
            mom_vehicle_size=mom_vehicle_size,
            sub_vehicle_size=sub_vehicle_size,
            vehicle_time=vehicle_time,
            ksi_max=ksi_max,
            ksi_c=ksi_c,
            ksi_s=ksi_s,
            T_s=T_s,
            T_f_s=T_f_s,
            T_f_c=T_f_c,
            travel_cost=travel_cost,
            mbr_distance=mbr_distance,
            dor_distance=dor_distance,
            mbr_wait=mbr_wait,
            dor_wait=dor_wait,
            time_cost=time_cost,
            order_size=sor_loc.size(1),
            tau_h=tau_h,
            tau_p=tau_p,
            speed=speed,
            mbr_speed=mbr_speed,
            dor_speed=dor_speed,
        )

    def get_sub_state(self):
        return self.vehicle_pos[:, :self.sub_vehicle_size, :], (self.vehicle_time[:, :self.sub_vehicle_size]).unsqueeze(-1)

    def get_mom_state(self):
        return self.vehicle_pos[:, self.sub_vehicle_size:, :], (self.vehicle_time[:, self.sub_vehicle_size:]).unsqueeze(-1)

    def get_costs(self, dataset, pi):
        time_limit = dataset['designated_time']
        if time_limit.ndim != self.tar_loc.ndim:
            time_limit = time_limit.repeat(1, self.T_f_s.size(1))
        else:
            time_limit = time_limit.squeeze(-1)
        cost_delay = torch.sum((self.T_f_s - time_limit).clamp(min=0), dim=1)
        return (self.travel_cost.detach(), cost_delay.detach()), None

    # 这是您类中的方法
    def update(self, task_selected, sub_vehicle_selected, mom_vehicle_selected):
        (
            new_mask, new_T_f_s, new_T_f_c, new_vehicle_time, new_vehicle_pos,
            new_travel_cost, new_mbr_distance, new_dor_distance,
            new_mbr_wait, new_dor_wait, new_last_task_indices,
            new_vehicle_coordinates,
        ) = update_state_tensors(
            self.mask, self.T_f_s, self.T_f_c, self.vehicle_pos, self.vehicle_time,
            self.vehicle_coordinates,
            self.sor_loc, self.tar_loc, self.tau_h, self.tau_p,
            self.travel_cost, self.mbr_distance, self.dor_distance,
            self.mbr_wait, self.dor_wait, self.ids, task_selected,
            sub_vehicle_selected, mom_vehicle_selected, self.params[:, 0],
            self.params[:, 1], self.mbr_speed
        )

        # 2. 更新 sequence (这是一个Python列表，在JIT外部处理)
        new_sequence = self.sequence + [torch.stack([task_selected, sub_vehicle_selected, mom_vehicle_selected], dim=1)]

        # 3. 使用 _replace() 创建并返回一个新的实例
        return self._replace(
            i=self.i + 1,
            mask=new_mask,
            T_f_s=new_T_f_s,
            T_f_c=new_T_f_c,
            vehicle_pos=new_vehicle_pos,
            vehicle_coordinates=new_vehicle_coordinates,
            vehicle_time=new_vehicle_time,
            travel_cost=new_travel_cost,
            mbr_distance=new_mbr_distance,
            dor_distance=new_dor_distance,
            mbr_wait=new_mbr_wait,
            dor_wait=new_dor_wait,
            last_task_indices=new_last_task_indices,
            sequence=new_sequence
        )


    # def update(self, task_selected, sub_vehicle_selected, mom_vehicle_selected):
    #     """
    #     状态更新函数，精确实现“母车先接子车，再共同执行任务”的逻辑。
    #     """
    #     sub_vehicles = sub_vehicle_selected
    #     mom_vehicles = mom_vehicle_selected
    #     order_indices = task_selected
    #
    #     idx = self.ids.squeeze(dim=1)
    #     n_orders = self.order_size
    #
    #     # --- 1. 更新 Mask (与之前版本相同) ---
    #     self.mask[self.ids, sub_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True
    #     self.mask[self.ids, mom_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True
    #
    #     # --- 2. 获取所有相关位置的坐标 (与之前版本相同) ---
    #     pre_sub_loc = self.tar_loc[idx, self.vehicle_pos[idx, sub_vehicles].squeeze()]
    #     pre_mom_loc = self.tar_loc[idx, self.vehicle_pos[idx, mom_vehicles].squeeze()]
    #
    #     pickup_loc = self.sor_loc[idx, order_indices]
    #     delivery_loc = self.tar_loc[idx, order_indices]
    #
    #     # --- 3. 模拟完整的任务流程 (全新逻辑) ---
    #     # 解包参数
    #     tau_a, tau_d, tau_p, tau_h, v = self.params[:, 0], self.params[:, 1], self.params[:, 2], self.params[:, 3], self.params[:, 4]
    #
    #     # == 阶段一: 集合 (Rendezvous) 在子车的旧位置 ==
    #     dist_mom_to_sub = torch.sum(torch.abs(pre_sub_loc - pre_mom_loc), dim=1)
    #     time_mom_to_sub = dist_mom_to_sub / v
    #
    #     # 计算母车到达子车位置的时间点
    #     rendezvous_arrival_time = self.vehicle_time[idx, mom_vehicles] + time_mom_to_sub
    #
    #     # 同步点: 团队的出发时间，取决于“子车何时空闲”和“母车何时到达”，取最晚的那个
    #     departure_time_from_rendezvous = torch.max(self.vehicle_time[idx, sub_vehicles], rendezvous_arrival_time)
    #
    #     # == 阶段二: 团队一起前往取货点 ==
    #     dist_rendezvous_to_pickup = torch.sum(torch.abs(pickup_loc - pre_sub_loc), dim=1)
    #     time_rendezvous_to_pickup = dist_rendezvous_to_pickup / v
    #
    #     arrival_time_at_pickup = departure_time_from_rendezvous + time_rendezvous_to_pickup
    #
    #     # == 阶段三: 在取货点进行服务 ==
    #     # 假设取货服务在到达后立即开始
    #     finish_time_at_pickup = arrival_time_at_pickup + tau_p
    #
    #     # == 阶段四: 团队一起前往送货点 ==
    #     dist_pickup_to_delivery = torch.sum(torch.abs(delivery_loc - pickup_loc), dim=1)
    #     time_pickup_to_delivery = dist_pickup_to_delivery / v
    #
    #     arrival_time_at_delivery = finish_time_at_pickup + time_pickup_to_delivery
    #
    #     # == 阶段五: 在送货点进行服务，计算最终完成时间 ==
    #     start_time_at_delivery = arrival_time_at_delivery + tau_a
    #     final_finish_time_sub = start_time_at_delivery + tau_d + tau_h
    #     final_finish_time_mom = start_time_at_delivery + tau_d
    #
    #     # --- 4. 更新车辆的最终状态 ---
    #     self.vehicle_time[idx, sub_vehicles] = final_finish_time_sub
    #     self.vehicle_time[idx, mom_vehicles] = final_finish_time_mom
    #     self.vehicle_pos[idx, sub_vehicles] = order_indices.unsqueeze(-1)
    #     self.vehicle_pos[idx, mom_vehicles] = order_indices.unsqueeze(-1)
    #
    #     # --- 5. 更新统计数据并返回新状态 ---
    #     # 总行程包含三段：母车->子车，团队->取货点，团队->送货点
    #     self.T_f_s[idx, order_indices] = final_finish_time_sub
    #     total_dist_this_step = dist_mom_to_sub + dist_rendezvous_to_pickup + dist_pickup_to_delivery
    #     dist = self.travel_cost + total_dist_this_step
    #     self.sequence.append(torch.stack([order_indices, sub_vehicle_selected, mom_vehicle_selected], dim=1))
    #     update_i = self.i[:, :] + 1
    #     return self._replace(i=update_i, last_task_indices=order_indices.unsqueeze(-1), travel_cost=dist)


    def all_finished(self):
        # Exactly n steps
        return self.i[0].item() >= self.order_size

    def get_current_node(self):
        return self.vehicle_pos, normalize(self.vehicle_time).unsqueeze(-1), self.last_task_indices

    def get_mom_vehicle_mask(self):
        """
        【新版逻辑 - 选项A】
        在新模型下，任务是原子性的，子车完成任务后即为可用。
        因此，不再需要基于上一步任务类型的特殊mask。
        此函数返回一个全为False的mask，表示所有子车均可选。
        具体的可用性由解码器根据 vehicle_time 判断。
        """
        batch_size = self.mask.size(0)
        device = self.mask.device

        # 始终返回一个没有车辆被禁用的mask
        return torch.zeros(batch_size, 1, self.mom_vehicle_size, dtype=torch.bool, device=device)

    def get_sub_vehicle_mask(self):
        """
        【新版逻辑 - 选项A】
        在新模型下，任务是原子性的，子车完成任务后即为可用。
        因此，不再需要基于上一步任务类型的特殊mask。
        此函数返回一个全为False的mask，表示所有子车均可选。
        具体的可用性由解码器根据 vehicle_time 判断。
        """
        batch_size = self.mask.size(0)
        sub_size = self.sub_vehicle_size
        device = self.mask.device

        # 始终返回一个没有车辆被禁用的mask
        return torch.zeros(batch_size, 1, sub_size, dtype=torch.bool, device=device)

    def get_task_mask(self):
        """
        【新版逻辑】
        生成可执行任务的掩码。
        在新模型下，任务是原子性的(P-D一体)。因此，mask的逻辑非常简单：
        任何已经被完成的订单/任务，都应该被禁用。

        此函数依赖于 self.mask 的正确更新。
        self.mask 的维度应为 (batch, num_vehicles, num_orders)。
        """
        # self.mask 记录了 (B, V, N) 的完成状态
        # 我们需要一个 (B, 1, N) 的mask，表示哪些任务被任何车辆完成过
        # .any(dim=1) 会在 vehicle 维度上进行“或”操作。
        # 只要有一个车辆完成了这个任务，结果就为 True。
        # keepdim=True 保持维度为 (B, 1, N) 以匹配输出格式。

        task_done_mask = self.mask.any(dim=1, keepdim=True)

        return task_done_mask
