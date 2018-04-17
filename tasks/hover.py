import math
import numpy as np
from .task import Task

class Hover(Task):
    """Hover task."""
    def __init__(self, target_z=150., start_z=100., hover_duration=3.0, simplified=False):
        super().__init__(
            target_pos=np.array([0., 0., target_z]),
            init_pose=np.array([0., 0., start_z, 0., 0., 0.]),
            runtime=10.0,
            simplified=simplified
        )

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = target_z
        self.hover_duration = hover_duration

        # Reward task-specific variables
        self.last_time = 0.0
        self.time_hover = 0.0

        self.task_name = 'hover'

    def is_task_finished(self):
        return False   # self.time_hover >= self.hover_duration

    def get_reward(self):
        reward = 20.0

        def apply_lower_threshold(x, threshold):
            return max(0.0, x - threshold)

        def apply_higher_threshold(x, threshold):
            return max(0.0, threshold - x)

        # Compute reward for reaching target height
        target_z_distance = abs(self.target_z - self.position[2])
        height_penalty = target_z_distance ** 0.7

        # Penalize for shifting in horizontal plane
        xy_shift_penalty = math.sqrt(self.position[0]**2 + self.position[1]**2)

        # Penalize for going to far away
        too_far_penalty = apply_lower_threshold(target_z_distance, 120) ** 2

        # Penalise for twist angles
        twist_penalty = apply_lower_threshold((1. - np.cos(self.orientation)).sum(), 0.1)

        # Penalise for rotating
        rotation_penalty = apply_lower_threshold(np.abs(self.sim.angular_v).sum(), 0.5)

        # Orientation reward
        reward -= height_penalty * 0.3 + xy_shift_penalty * 0.01 + too_far_penalty * 0.1 + twist_penalty * 3.0 + rotation_penalty * 0.5

        # Target closeness reward
        reward += apply_higher_threshold(target_z_distance, 5.0) * 10

        if reward < 0:
            reward = 3

        return reward

    def reset_vars(self):
        self.last_time = self.sim.time
        self.time_hover = 0.0
