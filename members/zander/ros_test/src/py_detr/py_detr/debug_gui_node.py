import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from PIL import ImageTk
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image
import tkinter as tk
from tkinter import ttk


class DebugPlaybackNode(Node):
    """ROS node responsible for playing bags and tracking topic previews."""

    def __init__(self):
        super().__init__("debug_gui_node")

        self.declare_parameter("bag_path", "")
        self.declare_parameter("frame_rate", 10.0)
        self.declare_parameter(
            "display_topics",
            ["camera/color/image_raw", "camera/depth/image_rect_raw"],
        )

        raw_bag_path = self.get_parameter("bag_path").get_parameter_value().string_value
        self.bag_path = self._expand_path(raw_bag_path)
        self.frame_rate = (
            self.get_parameter("frame_rate").get_parameter_value().double_value or 1.0
        )
        display_topics_param = (
            self.get_parameter("display_topics").get_parameter_value().string_array_value
        )
        self.display_topics = list(display_topics_param) or [
            "camera/color/image_raw",
            "camera/depth/image_rect_raw",
        ]

        if not self.bag_path:
            raise RuntimeError("The 'bag_path' parameter is required for debug_gui_node")

        # Bag/RealSense playback state
        self.mode: Optional[str] = None
        self.reader: Optional[SequentialReader] = None
        self.bag_topics: Dict[str, Dict[str, object]] = {}
        self.rs_pipeline = None
        self.rs_playback = None
        self.rs_color_pub = None
        self.rs_depth_pub = None

        self.bridge = CvBridge()
        self.bag_lock = threading.Lock()
        self._latest_images: Dict[str, PILImage.Image] = {}
        self._image_lock = threading.Lock()

        # Playback thread coordination
        self._play_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

        # Subscriptions for preview grid
        self._subscriptions = []
        for topic in self.display_topics:
            sub = self.create_subscription(
                Image, topic, self._make_image_callback(topic), 10
            )
            self._subscriptions.append(sub)

        # Prepare bag handling components
        if self._is_rosbag2_dir(self.bag_path):
            self.mode = "rosbag2"
            self._setup_bag_reader()
            self.get_logger().info(
                f"Debug GUI using rosbag2 directory '{self.bag_path}'"
            )
        elif os.path.isfile(self.bag_path) and self.bag_path.endswith(".bag"):
            self.mode = "realsense"
            self._setup_realsense_bag()
            self.get_logger().info(
                f"Debug GUI using RealSense bag '{self.bag_path}'"
            )
        else:
            raise RuntimeError(
                f"Bag path '{self.bag_path}' is neither a rosbag2 directory nor a .bag file"
            )

    # ------------------------- Parameter helpers ------------------------- #

    def _expand_path(self, path_str: str) -> str:
        path_str = path_str.strip() if path_str else ""
        if not path_str:
            return ""
        return os.path.abspath(os.path.expanduser(path_str))

    def _is_rosbag2_dir(self, path_str: str) -> bool:
        bag_path = Path(path_str)
        metadata_file = bag_path / "metadata.yaml"
        return bag_path.is_dir() and metadata_file.exists()

    # -------------------------- Image callbacks -------------------------- #

    def _make_image_callback(self, topic: str):
        def callback(msg: Image):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except Exception as exc:  # pragma: no cover - cv_bridge specific
                self.get_logger().warning(f"Failed to convert image on {topic}: {exc}")
                return

            if cv_image.ndim == 2:
                normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
                rgb_image = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                fmt = msg.encoding.lower()
                if fmt in ("rgb8", "rgba8"):
                    rgb_image = cv_image[..., :3]
                else:
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            pil_image = PILImage.fromarray(rgb_image)
            with self._image_lock:
                self._latest_images[topic] = pil_image

        return callback

    def get_latest_image(self, topic: str) -> Optional[PILImage.Image]:
        with self._image_lock:
            img = self._latest_images.get(topic)
            if img is None:
                return None
            return img.copy()

    # ---------------------------- Bag playback --------------------------- #

    def _open_bag_reader(self):
        reader = SequentialReader()
        storage_options = StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="", output_serialization_format=""
        )
        reader.open(storage_options, converter_options)
        return reader

    def _setup_bag_reader(self):
        self.reader = self._open_bag_reader()
        self.bag_topics = {}
        for topic_meta in self.reader.get_all_topics_and_types():
            try:
                msg_type = get_message(topic_meta.type)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                self.get_logger().warning(f"Skipping topic {topic_meta.name}: {exc}")
                continue
            publisher = self.create_publisher(msg_type, topic_meta.name, 10)
            self.bag_topics[topic_meta.name] = {"type": msg_type, "publisher": publisher}
        if not self.bag_topics:
            raise RuntimeError(f"No playable topics found within '{self.bag_path}'")

    def _reset_bag_reader(self):
        try:
            self.reader = self._open_bag_reader()
            self.get_logger().info("Restarted rosbag2 playback from start")
        except RuntimeError as exc:
            self.get_logger().error(f"Failed to restart rosbag2: {exc}")
            self.reader = None

    def _publish_next_rosbag_sample(self) -> bool:
        if self.reader is None:
            return False

        with self.bag_lock:
            topic = data = None
            while self.reader is not None:
                try:
                    topic, data, _ = self.reader.read_next()
                    break
                except RuntimeError:
                    self.get_logger().info("Reached end of rosbag2, rewinding")
                    self._reset_bag_reader()
                    if self.reader is None:
                        return False

            if topic is None or data is None:
                return False

            topic_info = self.bag_topics.get(topic)
            if topic_info is None:
                return False
            msg = deserialize_message(data, topic_info["type"])
            topic_info["publisher"].publish(msg)
            return True

    # ------------------------ RealSense bag playback --------------------- #

    def _setup_realsense_bag(self):
        self._start_realsense_pipeline()
        self.rs_color_pub = self.create_publisher(Image, "camera/color/image_raw", 10)
        self.rs_depth_pub = self.create_publisher(
            Image, "camera/depth/image_rect_raw", 10
        )

    def _start_realsense_pipeline(self):
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except RuntimeError:
                pass
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        try:
            config.enable_device_from_file(self.bag_path, repeat_playback=False)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to open RealSense bag '{self.bag_path}': {exc}")
        self.rs_pipeline.start(config)
        device = self.rs_pipeline.get_active_profile().get_device()
        self.rs_playback = device.as_playback()
        try:
            self.rs_playback.set_real_time(False)
        except RuntimeError:
            pass

    def _rewind_realsense_bag(self):
        try:
            self._start_realsense_pipeline()
            self.get_logger().info("Restarted RealSense playback from start")
        except RuntimeError as exc:
            self.get_logger().error(f"Failed to restart RealSense bag: {exc}")

    def _publish_next_realsense_sample(self) -> bool:
        if self.rs_pipeline is None:
            return False

        frames = self.rs_pipeline.poll_for_frames()
        if frames is None:
            self._rewind_realsense_bag()
            return False

        stamp = self.get_clock().now().to_msg()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        published = False

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
            published = True

        if depth_frame and self.rs_depth_pub is not None:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.rs_depth_pub.publish(depth_msg)
            published = True

        return published

    # ----------------------------- Playback API -------------------------- #

    def publish_next_frame(self) -> bool:
        if self.mode == "rosbag2":
            return self._publish_next_rosbag_sample()
        if self.mode == "realsense":
            return self._publish_next_realsense_sample()
        return False

    def start_playback(self):
        self._play_event.set()

    def pause_playback(self):
        self._play_event.clear()

    def is_playing(self) -> bool:
        return self._play_event.is_set()

    def describe_playback_state(self) -> str:
        mode = self.mode or "unknown"
        status = "playing" if self.is_playing() else "paused"
        return f"{mode} - {status}"

    def _playback_loop(self):
        while not self._shutdown_event.is_set():
            if not self._play_event.wait(timeout=0.1):
                continue
            start = time.time()
            self.publish_next_frame()
            elapsed = time.time() - start
            delay = max(0.0, (1.0 / max(self.frame_rate, 1e-3)) - elapsed)
            if delay:
                time.sleep(delay)

    def shutdown(self):
        self._shutdown_event.set()
        self._play_event.set()
        if self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
        if self.reader is not None:
            self.reader = None
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except RuntimeError:
                pass
            self.rs_pipeline = None


class DebugGuiApp:
    """Tkinter UI that drives the debug playback node."""

    def __init__(self, node: DebugPlaybackNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title("py_detr debug GUI")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._photo_refs: Dict[str, ImageTk.PhotoImage] = {}

        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(controls, text="Step", command=self._step_once).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Play", command=self.node.start_playback).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(controls, text="Pause", command=self.node.pause_playback).pack(
            side=tk.LEFT, padx=2
        )

        self.status_var = tk.StringVar(value=self.node.describe_playback_state())
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main")

        self.image_labels: Dict[str, tk.Label] = {}
        max_cols = 2
        for idx, topic in enumerate(self.node.display_topics):
            row = idx // max_cols
            col = idx % max_cols
            frame = ttk.Frame(main_tab, padding=4, relief=tk.GROOVE)
            frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            main_tab.grid_columnconfigure(col, weight=1)
            ttk.Label(frame, text=topic).pack(anchor=tk.W)
            label = tk.Label(frame, text="Waiting for image", bg="#111111", fg="#bbbbbb")
            label.pack(fill=tk.BOTH, expand=True)
            self.image_labels[topic] = label

        notebook.enable_traversal()
        self.root.after(200, self._refresh_images)
        self.root.after(200, self._refresh_status)

    def _step_once(self):
        self.node.publish_next_frame()

    def _refresh_status(self):
        self.status_var.set(self.node.describe_playback_state())
        self.root.after(500, self._refresh_status)

    def _refresh_images(self):
        for topic, label in self.image_labels.items():
            pil_image = self.node.get_latest_image(topic)
            if pil_image is None:
                continue
            display = self._resize_image(pil_image, max_width=480, max_height=360)
            photo = ImageTk.PhotoImage(display)
            self._photo_refs[topic] = photo
            label.configure(image=photo, text="")
        self.root.after(200, self._refresh_images)

    def _resize_image(self, image: PILImage.Image, max_width: int, max_height: int):
        width, height = image.size
        scale = min(max_width / max(1, width), max_height / max(1, height))
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            return image.resize(new_size, PILImage.BILINEAR)
        return image

    def _on_close(self):
        self.node.pause_playback()
        self.root.quit()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = DebugPlaybackNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    app = DebugGuiApp(node)
    try:
        app.run()
    finally:
        node.shutdown()
        executor.shutdown()
        executor.remove_node(node)
        executor_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
