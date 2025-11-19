import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

COCO_CLASSES = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
}


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

        self.declare_parameter("target_class", "bottle")
        target_class_name = (
            self.get_parameter("target_class").get_parameter_value().string_value
        )
        self.target_class_id = next(
            (cid for cid, name in COCO_CLASSES.items() if name == target_class_name),
            None,
        )
        if self.target_class_id is None:
            raise ValueError(f"Unknown TARGET_CLASS '{target_class_name}' for tracker")
        self.get_logger().info(
            f"Pitch tracker locked to TARGET_CLASS='{target_class_name}' (id={self.target_class_id})"
        )

        # ROS interfaces
        self.target_sub = self.create_subscription(
            Float64MultiArray, "/target/center", self.callback, 10
        )
        self.servo_pub = self.create_publisher(Float64, "/set_position", 10)

        self.smooth_cy = None
        self.get_logger().info("✅ Pitch tracker initialized")

    def callback(self, msg: Float64MultiArray):
        detections = self._parse_detections(msg.data)
        if not detections:
            return

        target_det = self._select_detection(detections)
        if target_det is None:
            return

        cx, cy = target_det["cx"], target_det["cy"]
        depth = target_det["depth"]

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
            f"cy={cy:.1f}, depth={depth if depth is not None else float('nan'):.2f}, "
            f"err={error_y:.1f}, corr={correction:.2f}, pitch={self.pitch_angle:.1f}"
        )

    def _parse_detections(self, data):
        detections = []
        if len(data) % 4 != 0:
            self.get_logger().warning("Malformed detection payload received")
            return detections

        for i in range(0, len(data), 4):
            class_id, cx, cy, depth = data[i : i + 4]
            detections.append(
                {
                    "class_id": int(round(class_id)),
                    "cx": float(cx),
                    "cy": float(cy),
                    "depth": None if math.isnan(depth) else float(depth),
                }
            )
        return detections

    def _select_detection(self, detections):
        # Prefer the requested target class; fallback to closest (smallest depth)
        target = next(
            (det for det in detections if det["class_id"] == self.target_class_id),
            None,
        )
        if target is not None:
            return target

        candidates = [
            det for det in detections if det["depth"] is not None
        ]
        if candidates:
            return min(candidates, key=lambda det: det["depth"])
        return detections[0] if detections else None


def main(args=None):
    rclpy.init(args=args)
    node = PitchTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
