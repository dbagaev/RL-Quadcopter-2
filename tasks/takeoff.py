"""Takeoff task."""

import math
from pathlib import Path
import numpy as np
from .task import Task

class Takeoff(Task):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self, target_z=20., start_z=5., simplified=False):
        self.task_name = 'takeoff'
        super().__init__(
            target_pos=np.array([0., 0., target_z]),
            init_pose=np.array([0., 0., start_z, 0., 0., 0.]),
            simplified=simplified
        )
        print("Starting takeoff task")

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = target_z

    def is_task_finished(self):
        return self.position[2] >= self.target_z

    def get_reward(self):
        reward = 30.0

        # Compute reward for reaching target height
        target_z_distance = abs(self.target_z - self.position[2])
        height_penalty = target_z_distance

        # Penalize for shifting in horizontal plane
        xy_shift_penalty = math.sqrt(self.position[0]**2 + self.position[1]**2)

        def apply_threshold(x, threshold):
            return max(0.0, x - threshold)

        # Penalise for twist angles
        twist_penalty = apply_threshold((1. - np.cos(self.orientation)).sum(), 0.1)

        # Penalise for rotating
        rotation_penalty = apply_threshold(np.abs(self.sim.angular_v).sum(), 0.5)
        # reward -= rotate_penalty * 1.0

        # Orientation reward
        reward -= height_penalty * 0.05 + xy_shift_penalty * 0.01 + twist_penalty * 2.0 + rotation_penalty * 0.5

        if self.is_task_finished():  # agent has crossed the target height
            reward += 50.0 # bonus reward for reaching the target
        #elif self.timeout:  # agent has run out of time
        #    reward -= 500.0  # extra penalty for timeout or going out of emulated space

        return reward
