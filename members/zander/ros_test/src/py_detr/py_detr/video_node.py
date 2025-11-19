import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional dependency
    rs = None


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_node")

        # Parameters
        self.declare_parameter("video_path", "input.mkv")
        self.declare_parameter("frame_rate", 30.0)  # Hz
        self.declare_parameter("bag_path", "")

        raw_video_path = self.get_parameter("video_path").get_parameter_value().string_value
        raw_bag_path = self.get_parameter("bag_path").get_parameter_value().string_value
        self.video_path = self._expand_path(raw_video_path)
        self.frame_rate = self.get_parameter("frame_rate").get_parameter_value().double_value
        self.bag_path = self._expand_path(raw_bag_path)

        self.bridge = CvBridge()
        self.mode = None
        self.timer = None

        # Video state
        self.cap = None
        self.publisher = None

        # ROS bag state
        self.reader = None
        self.bag_topics = {}
        self.storage_options = None

        # RealSense bag state
        self.rs_pipeline = None
        self.rs_playback = None
        self.rs_color_pub = None
        self.rs_depth_pub = None

        if self.bag_path:
            if self._is_rosbag2_dir(self.bag_path):
                self.mode = "rosbag2"
                self._setup_bag_reader()
                period = 1.0 / max(1e-3, self.frame_rate)
                self.timer = self.create_timer(period, self.publish_from_bag)
                self.get_logger().info(
                    f"Streaming ROS bag '{self.bag_path}' at {self.frame_rate:.2f} Hz (looping)"
                )
            elif os.path.isfile(self.bag_path) and self.bag_path.endswith(".bag"):
                self.mode = "realsense"
                self._setup_realsense_bag()
                period = 1.0 / max(1e-3, self.frame_rate)
                self.timer = self.create_timer(period, self.publish_from_realsense_bag)
                self.get_logger().info(
                    f"Streaming RealSense bag '{self.bag_path}' at {self.frame_rate:.2f} Hz (looping)"
                )
            else:
                raise RuntimeError(
                    f"Bag path '{self.bag_path}' is neither a rosbag2 directory nor a .bag file"
                )
        else:
            self.mode = "video"
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open video file: {self.video_path}")
                raise RuntimeError("Could not open video file")
            self.publisher = self.create_publisher(Image, "/camera/camera/color/image_raw", 10)
            self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)
            self.get_logger().info(
                f"Streaming video '{self.video_path}' at {self.frame_rate:.2f} Hz"
            )

    def _expand_path(self, path_str: str) -> str:
        path_str = path_str or ""
        path_str = path_str.strip()
        if not path_str:
            return ""
        return os.path.abspath(os.path.expanduser(path_str))

    def _is_rosbag2_dir(self, path_str: str) -> bool:
        if not path_str:
            return False
        bag_path = Path(path_str)
        metadata_file = bag_path / "metadata.yaml"
        return bag_path.is_dir() and metadata_file.exists()

    # ---------------------------- Video playback ---------------------------- #

    def publish_frame(self):
        if self.mode != "video":
            return
        ret, frame = self.cap.read()
        if not ret:
            # Loop video when it ends
            self.get_logger().info("End of video reached, looping...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            self.publisher.publish(msg)
        else:
            self.get_logger().warn("Failed to read frame after looping")

    # ---------------------------- Bag playback ----------------------------- #

    def _setup_bag_reader(self):
        self.reader = SequentialReader()
        self.storage_options = StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        converter_options = ConverterOptions(input_serialization_format="", output_serialization_format="")
        self.reader.open(self.storage_options, converter_options)

        self.bag_topics = {}
        for topic_meta in self.reader.get_all_topics_and_types():
            try:
                msg_type = get_message(topic_meta.type)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                self.get_logger().warning(f"Skipping topic {topic_meta.name}: {exc}")
                continue
            publisher = self.create_publisher(msg_type, topic_meta.name, 10)
            self.bag_topics[topic_meta.name] = {
                "type": msg_type,
                "publisher": publisher,
            }
        if not self.bag_topics:
            raise RuntimeError(f"No playable topics found in bag '{self.bag_path}'")
        self.get_logger().info(
            f"Bag topics: {', '.join(self.bag_topics.keys())}"
        )

    def _reset_bag_reader(self):
        self.reader = SequentialReader()
        converter_options = ConverterOptions(input_serialization_format="", output_serialization_format="")
        self.reader.open(self.storage_options, converter_options)

    def publish_from_bag(self):
        if self.mode != "rosbag2":
            return

        if not self.reader.has_next():
            self.get_logger().info("Reached end of bag, looping...")
            self._reset_bag_reader()
            if not self.reader.has_next():
                self.get_logger().error("Bag appears to contain no messages")
                return

        topic, data, _ = self.reader.read_next()
        topic_info = self.bag_topics.get(topic)
        if topic_info is None:
            return  # Topic skipped
        msg = deserialize_message(data, topic_info["type"])
        topic_info["publisher"].publish(msg)

    # -------------------------- RealSense bag playback --------------------- #

    def _setup_realsense_bag(self):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required for RealSense .bag playback but is not installed"
            )
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        try:
            config.enable_device_from_file(self.bag_path, repeat_playback=False)
        except RuntimeError as exc:  # pragma: no cover - device file errors
            raise RuntimeError(f"Failed to open RealSense bag '{self.bag_path}': {exc}") from exc
        self.rs_pipeline.start(config)
        device = self.rs_pipeline.get_active_profile().get_device()
        self.rs_playback = device.as_playback()
        try:
            self.rs_playback.set_real_time(False)
        except RuntimeError:
            pass  # some firmware versions omit playback control

        self.rs_color_pub = self.create_publisher(Image, "/camera/camera/color/image_raw", 10)
        self.rs_depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)

    def _rewind_realsense_bag(self):
        if not self.rs_playback:
            return
        try:
            self.rs_playback.pause()
            self.rs_playback.seek(rs.time_point(0))
            self.rs_playback.resume()
            self.get_logger().info("Looping RealSense bag to beginning")
        except RuntimeError as exc:
            self.get_logger().error(f"Failed to rewind RealSense bag: {exc}")

    def publish_from_realsense_bag(self):
        if self.mode != "realsense" or self.rs_pipeline is None:
            return

        frames = self.rs_pipeline.poll_for_frames()
        if frames is None:
            self._rewind_realsense_bag()
            return

        stamp = self.get_clock().now().to_msg()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame and self.rs_color_pub is not None:
            color_image = np.asanyarray(color_frame.get_data())
            fmt = color_frame.get_profile().format()
            encoding = "rgb8"
            if fmt == rs.format.bgr8 or fmt == rs.format.bgra8:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            elif fmt == rs.format.yuyv:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2RGB_YUYV)
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding=encoding)
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = "camera_color_optical_frame"
            self.rs_color_pub.publish(color_msg)

        if depth_frame and self.rs_depth_pub is not None:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.rs_depth_pub.publish(depth_msg)

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except RuntimeError:
                pass
            self.rs_pipeline = None
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
