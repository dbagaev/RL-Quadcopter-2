from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench

import numpy as np


def limit(t, mn, mx):
    if t < mn:
        return mn
    elif t > mx:
        return mx
    else:
        return t

class Emulator:
    def __init__(self, init_pose=None, init_v=None, runtime=10.):
        self.init_pose = init_pose
        self.init_velocities = init_v
        self.runtime = runtime
        self.reset()

    def reset(self):
        self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]) if self.init_pose is None else np.copy(self.init_pose)
        self.v = np.array([0.0, 0.0, 0.0])
        self.angular_v = np.array([0.0, 0.0, 0.0])

        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.prop_wind_speed = np.array([0., 0., 0., 0.])
        self.done = False

        self.time = 0.0
        self.time_delta = 0.1
        self.gravity = np.array([0.0, 0.0, 9.8])

    def next_timestep(self, w):
        w[0] = limit(w[0], -25.0, 25.0)
        w[1] = limit(w[1], -25.0, 25.0)
        w[2] = limit(w[2], -25.0, 25.0)

        self.pose[0:3] = self.pose[0:3] + self.v * self.time_delta + 0.5 * (w - self.gravity) * self.time_delta * self.time_delta
        self.v += (w - self.gravity) * self.time_delta

        # At the ground?
        if self.pose[2] < 0:
            self.pose[2] = 0
            self.v[2] = 0

        self.time += self.time_delta

        if self.time > self.runtime:
            self.done = True

