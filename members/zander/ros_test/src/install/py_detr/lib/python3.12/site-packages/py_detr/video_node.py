import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_node")

        # Parameters
        self.declare_parameter("video_path", "input.mkv")
        self.declare_parameter("frame_rate", 30.0)  # Hz

        self.video_path = self.get_parameter("video_path").get_parameter_value().string_value
        self.frame_rate = self.get_parameter("frame_rate").get_parameter_value().double_value

        # Try to open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {self.video_path}")
            raise RuntimeError("Could not open video file")

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        # Create timer to publish frames at given frame rate
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)

        self.get_logger().info(f"Streaming video '{self.video_path}' at {self.frame_rate:.2f} Hz")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Loop video when it ends
            self.get_logger().info("End of video reached, looping...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(msg)
        else:
            self.get_logger().warn("Failed to read frame after looping")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
