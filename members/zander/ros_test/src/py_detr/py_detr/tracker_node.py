import time

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node


class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.output_limit = output_limit

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0

        dt = now - self.prev_time
        self.prev_time = now

        # Integral and derivative
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 1e-3 else 0.0
        self.prev_error = error

        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limit:
            output = max(min(output, self.output_limit), -self.output_limit)
        return output


class Tracker(Node):
    def __init__(self):
        super().__init__("tracker_node")

        # === Parameters ===
        self.image_w = 320
        self.image_h = 240
        self.target_sub = self.create_subscription(Point, "/target/center", self.callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # PID for horizontal and vertical control
        self.pid_x = PIDController(kp=0.005, ki=0.0, kd=0.001, output_limit=0.5)
        self.pid_y = PIDController(kp=0.005, ki=0.0, kd=0.001, output_limit=0.5)

        # EMA for smoothing the incoming detections
        self.smooth_cx = None
        self.smooth_cy = None
        self.alpha = 0.3  # smoothing factor

        self.get_logger().info("✅ Tracker initialised")

    def callback(self, msg: Point):
        cx, cy, conf = msg.x, msg.y, msg.z
        if conf < 0.4:  # skip low-confidence detections
            return

        # Smooth detection
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = cx, cy
        else:
            self.smooth_cx = self.alpha * cx + (1 - self.alpha) * self.smooth_cx
            self.smooth_cy = self.alpha * cy + (1 - self.alpha) * self.smooth_cy

        # Compute normalized error
        error_x = (self.image_w / 2 - self.smooth_cx)
        error_y = (self.image_h / 2 - self.smooth_cy)

        # Compute control
        vx = self.pid_x.update(error_x)
        vy = self.pid_y.update(error_y)

        # Publish Twist command (example for differential drive robot)
        cmd = Twist()
        cmd.angular.z = vx  # rotate to reduce horizontal error
        cmd.linear.x = vy * -1  # move forward/backward to reduce vertical error
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Target ({cx:.0f},{cy:.0f}) err=({error_x:.0f},{error_y:.0f}) "
            f"cmd=({vx:.3f},{vy:.3f})"
        )

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Tracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
