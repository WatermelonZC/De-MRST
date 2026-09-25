import math

from heuristic.Util.Config import Config
from heuristic.Util.util import (
    copy_dict_int_dict,
    copy_dict_int_int,
    copy_dict_int_list,
)
from mrs.core import (
    TaskSpec,
    deadlock_penalty,
    evaluate_centralized_plan,
    manhattan_distance,
    objective_value_for_mode,
)


class Solution:
    def __init__(self, instance, sequence_map, path_init_task_map):
        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map
        self.objective_mode = getattr(Config, 'objective_mode', 'v4')
        if self.objective_mode not in (
            'v4', 'makespan', 'engineering', 'distance_tardiness',
            'makespan_distance',
        ):
            raise ValueError(
                "objective_mode must be 'v4', 'makespan', 'engineering', or "
                "'distance_tardiness', or 'makespan_distance'"
            )

        self.distance = None
        self.tardiness = None
        self.fitness = None
        self.feasible = None
        self.path_map = None
        self.metrics = None

        self.task_num = len([task for task in self.sequence_map.keys()])  # 任务数

        self.info_map = {task: dict() for task in sequence_map.keys()}  # 存储某个任务的开始时间，结束时间等信息
        self.code = self.get_code()

        self.hash_key = hash(tuple(self.code[0] + self.code[1]))

    def get_path_map(self):
        if self.path_map is not None:
            return copy_dict_int_list(self.path_map)
        path_map = {}
        for path_index, init_task in self.path_init_task_map.items():
            if init_task == 0:
                # 该path为空
                path_map[path_index] = []
                continue
            if path_index in Config.M_parent_list:
                path = [init_task]
                parent_next_task = self.sequence_map[init_task]['parent_next_task']
                while parent_next_task != 0:
                    path.append(parent_next_task)
                    parent_next_task = self.sequence_map[parent_next_task]['parent_next_task']
                path_map[path_index] = path
            else:
                path = [init_task]
                parent_next_task = self.sequence_map[init_task]['child_next_task']
                while parent_next_task != 0:
                    path.append(parent_next_task)
                    parent_next_task = self.sequence_map[parent_next_task]['child_next_task']
                path_map[path_index] = path
        self.path_map = path_map
        return path_map

    def get_code(self):
        code_parent = [0]
        code_child = [0]
        for path_index in range(1, Config.M + 1):
            init_task = self.path_init_task_map[path_index]
        # for path_index, init_task in self.path_init_task_map.items():
            if path_index in Config.M_parent_list:
                chain = "parent"
                code_list = code_parent
            else:
                chain = "child"
                code_list = code_child
            if init_task == 0:
                if path_index != Config.M_parent and path_index != Config.M_child + Config.M_parent:
                    code_list.append(0)
            else:
                path = [init_task]
                next_task = self.sequence_map[init_task][chain + '_next_task']
                while next_task != 0:
                    path.append(next_task)
                    next_task = self.sequence_map[next_task][chain + '_next_task']
                code_list += path
                if path_index != Config.M_parent and path_index != Config.M_child + Config.M_parent:
                    code_list.append(0)
        return [code_parent, code_child]

    def get_fitness(self):
        if self.fitness is not None:
            return self.fitness
        all_tasks = self._task_specs()
        selected_task_numbers = sorted(self.sequence_map)
        tasks = [all_tasks[task_number - 1] for task_number in selected_task_numbers]
        dense_task_id = {
            task_number: index
            for index, task_number in enumerate(selected_task_numbers)
        }
        order = self._execution_order()
        if order is None:
            self.feasible = False
            self.distance = 0.0
            self.tardiness = 0.0
            self.fitness = (
                100.0 * max(1, len(tasks))
                if self.objective_mode in ('makespan', 'engineering')
                else deadlock_penalty(tasks)
            )
            return self.fitness

        assignments = []
        for task_number in order:
            task_data = self.sequence_map[task_number]
            mbr_id = task_data['parent'] - 1
            dor_id = task_data['child'] - Config.M_parent - 1
            assignments.append((dense_task_id[task_number], mbr_id, dor_id))

        self.metrics = evaluate_centralized_plan(
            tasks,
            assignments,
            self.instance.mbr_count,
            self.instance.dor_count,
            self.instance.speed,
            self.instance.dock_time,
            self.instance.detach_time,
            mbr_speed=getattr(self.instance, 'mbr_speed', self.instance.speed),
            dor_speed=getattr(self.instance, 'dor_speed', self.instance.speed),
            mbr_initial_positions=getattr(self.instance, 'mbr_initial_positions', None) or None,
            dor_initial_positions=getattr(self.instance, 'dor_initial_positions', None) or None,
            objective_mode=self.objective_mode,
        )
        for execution in self.metrics.executions:
            task_number = selected_task_numbers[execution.task_id]
            task = tasks[execution.task_id]
            result = execution.result
            rendezvous_distance = manhattan_distance(
                execution.mbr_before.position, execution.dor_before.position
            )
            rendezvous_travel = rendezvous_distance / getattr(
                self.instance, 'mbr_speed', self.instance.speed
            )
            mbr_arrival = execution.mbr_before.available_time + rendezvous_travel
            synchronized = max(execution.dor_before.available_time, mbr_arrival)
            task_tardiness = max(0.0, result.task_completion_time - task.due_time)
            info = self.info_map[task_number]
            info.update({
                'parent_pre_d_time': execution.mbr_before.available_time,
                'child_pre_e_time': execution.dor_before.available_time,
                'parent_start_time': max(
                    execution.mbr_before.available_time,
                    execution.dor_before.available_time - rendezvous_travel,
                ),
                'attach_time': synchronized,
                'parent_decouple_time': result.mbr_release_time,
                'child_end_time': result.dor_release_time,
                'distance': result.system_distance,
                'mbr_distance': result.mbr_distance,
                'dor_distance': result.dor_distance,
                'mbr_wait': result.mbr_wait,
                'dor_wait': result.dor_wait,
                'parent_pre_position': list(execution.mbr_before.position),
                'child_pre_position': list(execution.dor_before.position),
                'source2destination_distance': manhattan_distance(task.source, task.destination),
                'parent2source_distance': manhattan_distance(execution.mbr_before.position, task.source),
                'child2source_distance': manhattan_distance(execution.dor_before.position, task.source),
                'child': self.sequence_map[task_number]['child'],
                'parent': self.sequence_map[task_number]['parent'],
                'tardiness': task_tardiness,
                'cost': (
                    result.task_completion_time
                    if self.objective_mode == 'makespan'
                    else objective_value_for_mode(
                        self.objective_mode,
                        task_tardiness,
                        result.system_distance,
                        len(tasks),
                        self.instance.speed,
                        makespan=result.task_completion_time,
                        mbr_speed=getattr(
                            self.instance, 'mbr_speed', self.instance.speed
                        ),
                    )
                ),
                'misalignment': abs(execution.dor_before.available_time - mbr_arrival),
            })

        self.fitness = (
            self.metrics.makespan
            if self.objective_mode == 'makespan'
            else self.metrics.objective
        )
        self.distance = self.metrics.system_distance
        self.tardiness = self.metrics.total_tardiness
        self.feasible = True
        return self.fitness

    def _task_specs(self):
        tasks = []
        for row in self.instance:
            pickup_time = self.instance.pickup_time
            if len(row) > 7 and not math.isnan(float(row[7])):
                pickup_time = float(row[7])
            tasks.append(TaskSpec(
                source=(float(row[1]), float(row[2])),
                destination=(float(row[3]), float(row[4])),
                due_time=float(row[5]),
                pickup_time=pickup_time,
                handling_time=float(row[6]),
            ))
        return tasks

    def _execution_order(self):
        parent_ready = set()
        child_ready = set()
        for path_index, first_task in self.path_init_task_map.items():
            if first_task == 0:
                continue
            if path_index in Config.M_parent_list:
                parent_ready.add(first_task)
            else:
                child_ready.add(first_task)

        order = []
        while len(order) < self.task_num:
            enabled = parent_ready & child_ready
            if not enabled:
                return None
            task = min(enabled)
            order.append(task)
            parent_ready.remove(task)
            child_ready.remove(task)
            parent_next = self.sequence_map[task]['parent_next_task']
            child_next = self.sequence_map[task]['child_next_task']
            if parent_next != 0:
                parent_ready.add(parent_next)
            if child_next != 0:
                child_ready.add(child_next)
        return order

    def get_path_init_task_map(self):
        """
        深拷贝输出
        :return:
        """
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self):
        """
        深拷贝输出
        :return:
        """
        return copy_dict_int_dict(self.sequence_map)
