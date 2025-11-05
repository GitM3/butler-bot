import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Float64


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


class PitchTracker(Node):
    def __init__(self):
        super().__init__("pitch_tracker")

        # Camera params
        self.image_h = 480
        self.alpha = 0.3  # smoothing

        # Servo limits
        self.pitch_min = 0.0
        self.pitch_max = 90.0
        self.pitch_angle = 0.0  # starting position

        # PID for pitch
        self.pid_pitch = PIDController(kp=0.02, ki=0.0, kd=0.0, output_limit=5.0)

        # ROS interfaces
        self.target_sub = self.create_subscription(Point, "/target/center", self.callback, 10)
        self.servo_pub = self.create_publisher(Float64, "/set_position", 10)

        self.smooth_cy = None
        self.get_logger().info("✅ Pitch tracker initialized")

    def callback(self, msg: Point):
        cx, cy, conf = msg.x, msg.y, msg.z
        if conf < 0.4:
            return

        # Smooth detection
        if self.smooth_cy is None:
            self.smooth_cy = cy
        else:
            self.smooth_cy = self.alpha * cy + (1 - self.alpha) * self.smooth_cy

        # Compute error (positive if target below center)
        error_y = (self.image_h / 2 - self.smooth_cy)

        # Convert pixel error to angle correction
        correction = self.pid_pitch.update(error_y)

        # Update servo pitch
        self.pitch_angle += correction
        self.pitch_angle = max(min(self.pitch_angle, self.pitch_max), self.pitch_min)

        # Publish new servo angle
        msg_out = Float64()
        msg_out.data = self.pitch_angle
        self.servo_pub.publish(msg_out)

        self.get_logger().info(
            f"cy={cy:.1f}, err={error_y:.1f}, corr={correction:.2f}, pitch={self.pitch_angle:.1f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PitchTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
