from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench

def limit(t, mn, mx):
    if t < mn:
        return mn
    elif t > mx:
        return mx
    else:
        return t

class Emulator:
    def __init__(self):
        self.pose = Pose(
            position=Point(0.0, 0.0, 0.0),  # drop off from a slight random height
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0))
        self.velocity = Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

        self.time = 0.0
        self.time_delta = 0.1
        self.gravity = 10.0

    def reset(self, pose, velocity):
        self.pose = pose
        self.velocity = velocity
        self.time = 0.0

    def process(self, w):
        w.force.x = limit(w.force.x, -25.0, 25.0)
        w.force.y = limit(w.force.y, -25.0, 25.0)
        w.force.z = limit(w.force.z, -25.0, 25.0)

        self.velocity.linear.x += w.force.x * self.time_delta
        self.velocity.linear.y += w.force.y * self.time_delta
        self.velocity.linear.z += (w.force.z - self.gravity) * self.time_delta

        self.pose.position.x += self.velocity.linear.x * self.time_delta
        self.pose.position.y += self.velocity.linear.y * self.time_delta
        self.pose.position.z += self.velocity.linear.z * self.time_delta

        if(self.pose.position.z < 0):
            self.pose.position.z = 0
            self.velocity.linear.z = 0

        self.time += self.time_delta

    def dump(self):
        print('#{:4.1f}: [{:9.3f}, {:9.3f}, {:9.3f}]'.format(self.time, self.pose.position.x, self.pose.position.y, self.pose.position.z), end=' ')
        pass

