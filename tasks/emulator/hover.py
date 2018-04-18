"""Hover task."""

import math
from pathlib import Path
import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from .quadrocopter_task import QuadrocopterTask
from .emulator import Emulator

from ..task import Task

class Hover(Task):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self, start_z=5, target_z=15, simplified=False):
        self.task_name = 'emulator_hover'
        super().__init__(
            target_pos=np.array([0., 0., target_z]),
            init_pose=np.array([0., 0., start_z, 0., 0., 0.]),
            runtime=10.0,
            simplified=simplified
        )

        self.action_repeat = 1
        self.state_size = self.action_repeat * len(self.state)

        self.action_low = -25
        self.action_high = 25

        self.simplified = simplified

        if self.simplified:
            self.action_size = 1
        else:
            self.action_size = 3

        # Task-specific parameters
        self.hover_duration = 3.0

        # Reward task-specific variables
        self.last_time = 0.0
        self.time_hover = 0.0


        self.time_hover = 0
        self.last_time_stamp = 0.0

    def create_emulator(self, init_pose, init_velocities, init_angle_velocities, runtime):
        return Emulator(init_pose, runtime)

    def reset_vars(self):
        self.last_time = self.sim.time
        self.time_hover = 0.0

    def get_reward(self):
        timestamp = self.sim.time
        pose = self.sim.pose

        # Compute reward / penalty and check if this episode is complete
        reward = 0.0

        height_penalty = min(abs(self.target_pos[2] - pose[2]), 20.0)
        reward -= height_penalty

        dist_penalty = abs(pose[0]) + abs(pose[1])
        reward -= dist_penalty * 0.1

        # Is it close to hover position
        if abs(pose[2] - self.target_pos[2]) < 0.5:
            self.time_hover += timestamp - self.last_time_stamp
            reward += 2.0 * self.time_hover
        if self.is_task_finished():  # agent has crossed the target height
            reward += 50.0  # bonus reward
        elif self.timeout:  # agent has run out of time
            reward -= 50.0  # extra penalty

        self.last_time_stamp = timestamp

        return reward

    def is_task_finished(self):
        return self.time_hover >= 3.0

    def simplified_action_to_full(self, action):
        return np.array([0.0, 0.0, action[0]])
