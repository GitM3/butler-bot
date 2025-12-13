import gc
import math
import time
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import rclpy
import tf2_geometry_msgs  # noqa: F401
import tf_transformations
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray, String
from tf2_ros import (Buffer, StaticTransformBroadcaster, TransformBroadcaster,
                     TransformListener)

COL_WHITE = (255, 255, 255)
COL_TARGET = (0, 200, 255)
COL_CONTOUR = (0, 255, 0)
COL_SEARCH = (255, 165, 0)

# Check for updates: https://github.com/roboflow/rf-detr/blob/main/rfdetr/util/coco_classes.py
COCO_CLASSES = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
}

class State(Enum):
    DETECT = 0
    SEARCH = 1
    TRACK = 2
    FINISH = 3
STATES = ["DETECT","SEARCH","TRACK","FINISH"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

class RFDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path):
        try:
            # Load the ONNX model and initialize the ONNX Runtime session
            self.ort_session = ort.InferenceSession(onnx_model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            # Get input shape
            input_info = self.ort_session.get_inputs()[0]
            self.input_name = input_info.name
            shape = list(input_info.shape)  # [N,C,H,W]
            self.fixed_h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) and shape[2] > 0 else None
            self.fixed_w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) and shape[3] > 0 else None
            print("Providers:", self.ort_session.get_providers())
            print(f"Model input: NCHW={shape}, fixed_size=({self.fixed_h},{self.fixed_w})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{onnx_model_path}'. "
                f"Ensure the path is correct and the model is a valid ONNX file."
            ) from e

    def _preprocess(self, frame_rgb:np.ndarray)->np.ndarray:
        """
        Preprocesses to correct format for inferrence.
        frame_rgb: HxWx{3 or 4} uint8, RGB(A)
        Returns: 1xCxHxW float32
        """
        
        # TODO: if not needed remove, add a check maybe
        # Resize the image to the model's input size
        # Drop alpha if present (RGBA -> RGB) without copy when possible
        if frame_rgb.shape[2] == 4:
            frame_rgb = frame_rgb[:, :, :3]

        h_in = self.fixed_h or frame_rgb.shape[0]
        w_in = self.fixed_w or frame_rgb.shape[1]

        if frame_rgb.shape[0] != h_in or frame_rgb.shape[1] != w_in:
            frame_rgb = cv2.resize(frame_rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

        x = frame_rgb.astype(np.float32) / 255.0
        x = (x - self.MEANS) / self.STDS     # HWC
        x = np.transpose(x, (2, 0, 1))       # CHW
        x = np.expand_dims(x, 0)             # NCHW
        return x.astype(np.float32, copy=False)

    def _post_process(
        self,
        outputs,
        origin_h: int,
        origin_w: int,
        confidence_threshold: float,
        max_number_boxes: int,
        allowed_cids: set | None = None,
        ):
        """
        Expects outputs like:
          outputs[0]: boxes (N, num_queries, 4) in normalized cx,cy,w,h
          outputs[1]: logits (N, num_queries, num_classes) BEFORE sigmoid
          (optional) outputs[2]: masks ...
        Returns: scores, labels, boxes_xyxy_abs, masks_or_None
        """
        boxes_pred = outputs[0].squeeze(0)           # (Q,4)
        logits     = outputs[1].squeeze(0)           # (Q,C)
        masks      = outputs[2].squeeze(0) if len(outputs) >= 3 else None  # (Q,H,W) or None

        probs  = sigmoid(logits)                     
        scores = np.max(probs, axis=1)               
        labels = np.argmax(probs, axis=1)            

        order = np.argsort(scores)[::-1]
        order = order[:max_number_boxes]
        scores = scores[order]
        labels = labels[order]
        boxes  = boxes_pred[order]
        if masks is not None:
            masks = masks[order]

        # Filter by threshold
        keep = scores > confidence_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes  = boxes[keep]
        if masks is not None:
            masks = masks[keep]

        if allowed_cids is not None and len(scores) > 0:
            mask_keep = np.array([int(c) in allowed_cids for c in labels], dtype=bool)
            scores = scores[mask_keep]
            labels = labels[mask_keep]
            boxes  = boxes[mask_keep]
            if masks is not None:
                masks = masks[mask_keep]

        boxes = box_cxcywh_to_xyxy(boxes)            # normalized
        boxes[:, [0, 2]] *= float(origin_w)
        boxes[:, [1, 3]] *= float(origin_h)

        return scores, labels, boxes, masks

    def predict(
        self,
        frame_rgb: np.ndarray,
        confidence_threshold: float = 0.4,
        max_number_boxes: int = 50,
        allowed_cids: set | None = None,
    ):
        """
        frame_rgb: HxWx{3 or 4} uint8 in RGB(A)
        Returns: scores (K,), labels (K,), boxes_xyxy_abs (K,4), masks or None
        """
        origin_h, origin_w = frame_rgb.shape[:2]
        input_image = self._preprocess(frame_rgb)
        outputs = self.ort_session.run(None, {self.input_name: input_image})
        return self._post_process(outputs, origin_h, origin_w, confidence_threshold, max_number_boxes, allowed_cids)


class RFDetrNode(Node):
    def __init__(self):
        super().__init__("rfdetr_node")
        self.declare_parameter("model_path", "base.onnx")
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/camera/annotated", 10)
        self.target_pub = self.create_publisher(Float64MultiArray, "/target/center", 10)
        self.servo_pub = self.create_publisher(Float64, "/set_position", 10)
        self.yaw_pub = self.create_publisher(Float64, "/set_yaw", 10)
        self.state_pub = self.create_publisher(String, "/detector/state", 10)
        # Transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_frame = "camera_link"
        self.camera_optical_frame = "camera_optical_frame"
        self.base_frame = "base_link"

        self.point_cam_pub = self.create_publisher(PointStamped, "target_point_cam", 10)
        self.point_base_pub = self.create_publisher(PointStamped, "target_point_base", 10)

        filter_labels = ["cup", "bottle", "wine glass"]
        self.target_labels = [i for i, n in COCO_CLASSES.items() if n in filter_labels]
        self.get_logger().info(f"Target labels: {self.target_labels} ({filter_labels})")
        self.declare_parameter("target_class", "bottle")
        self.target_class_name = (
            self.get_parameter("target_class").get_parameter_value().string_value
        )
        try:
            self.target_class_id = next(
                cid for cid, name in COCO_CLASSES.items() if name == self.target_class_name
            )
        except StopIteration as exc:
            raise ValueError(
                f"Unknown TARGET_CLASS '{self.target_class_name}' in COCO subset"
            ) from exc
        self.get_logger().info(
            f"Tracking TARGET_CLASS='{self.target_class_name}' (id={self.target_class_id})"
        )
        self.declare_parameter("depth_scale", 1.0)
        self.depth_scale = (
            self.get_parameter("depth_scale").get_parameter_value().double_value
        )

        self.model = RFDETR_ONNX(self.model_path)
        self.state = State.DETECT
        # tracking parameters
        self.prev_t = time.time()
        self.track_stable_start = None
        self.prev_contour_center = None
        self.detect_yes = 0
        self.detect_no = 0
        self.DETECT_YES_THRESH = 3   # 3 stable frames = object confirmed
        self.DETECT_NO_THRESH  = 5
        self.contour_yes = 0
        self.contour_no = 0
        self.CONTOUR_YES_THRESH = 3
        self.CONTOUR_NO_THRESH  = 5
        self.frame_counter = 0
        # Control parameters
        self.pitch_min = 0.0
        self.pitch_max = 90.0
        self.pitch_angle = 30.0
        self.pitch_angle_prev = 0.0
        self.yaw_angle = 0.0
        self.yaw_angle_prev = 0.0
        self.image_h = 480
        self.image_w = 640
        self.image_cx= int(self.image_w / 2)
        self.image_cy = int(self.image_h / 2)
        self.x_max = int(0.3 * self.image_w)
        self.y_max = int(0.3 * self.image_h)
        self.publish_camera_tf()
        self.publish_static_camera_optical_tf()

        # Depth contour tuning
        self.declare_parameter("depth_mask_threshold_mm", 700.0)
        self.declare_parameter("depth_mask_margin_mm", 20.0)
        self.declare_parameter("depth_mask_percentile", 2.0)
        self.declare_parameter("depth_kernel_size", 9)
        self.declare_parameter("depth_min_contour_area", 500.0)
        self.declare_parameter("track_window_px", 120.0)
        self.declare_parameter("search_bbox_margin", 0.1)
        self.declare_parameter("depth_threshold", 600.0)
        self.declare_parameter("bag_path", "")
        self.declare_parameter("frame_rate", 20.0)
        self.declare_parameter("finish_time", 15.0)
        self.declare_parameter("annotate", False)
        self.declare_parameter("pitch_step", 5.0 )

        self.annotate = self.get_parameter("annotate").get_parameter_value().bool_value
        self.draw = self.annotate
        self.finish_time = float(self.get_parameter("finish_time").get_parameter_value().double_value)
        self.depth_mask_threshold = float(self.get_parameter("depth_mask_threshold_mm").get_parameter_value().double_value)
        self.depth_mask_margin = float(self.get_parameter("depth_mask_margin_mm").get_parameter_value().double_value)
        self.depth_mask_percentile = float(self.get_parameter("depth_mask_percentile").get_parameter_value().double_value)
        self.depth_kernel_size = int(self.get_parameter("depth_kernel_size").get_parameter_value().integer_value)
        if self.depth_kernel_size < 3:
            self.depth_kernel_size = 3
        if self.depth_kernel_size % 2 == 0:
            self.depth_kernel_size += 1
        self.depth_min_contour_area = float(self.get_parameter("depth_min_contour_area").get_parameter_value().double_value)
        self.track_window_px = int(self.get_parameter("track_window_px").get_parameter_value().double_value)
        if self.track_window_px < 20:
            self.track_window_px = 20
        self.search_bbox_margin = float(self.get_parameter("search_bbox_margin").get_parameter_value().double_value)
        self.pitch_step = float(self.get_parameter("pitch_step").get_parameter_value().double_value)
        self.depth_threshold = float(self.get_parameter("depth_threshold").get_parameter_value().double_value)
        self.bag_path = self.get_parameter("bag_path").get_parameter_value().string_value
        self.frame_rate = float(self.get_parameter("frame_rate").get_parameter_value().double_value)
        self.timer = self.create_timer(1/self.frame_rate, self.callback)  
        self.prev_state = State.FINISH
        if self.depth_kernel_size % 2 == 0:
            self.depth_kernel_size += 1
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.depth_kernel_size, self.depth_kernel_size)
        )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.bag_path != "":
            self.config.enable_device_from_file(self.bag_path)
        else:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        if self.bag_path != "":
            self.device = self.pipeline.get_active_profile().get_device()
            self.device.as_playback().set_real_time(False)
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile() 
        self.color_intrinsics = color_stream.get_intrinsics()
        # RealSense post-processing filters
        # self.spatial_filter = rs.spatial_filter()
        # self.temporal_filter = rs.temporal_filter()
        # self.hole_filling_filter = rs.hole_filling_filter()
        # self.to_disparity = rs.disparity_transform(True)
        # self.from_disparity = rs.disparity_transform(False)

        # self.spatial_filter.set_option(rs.option.filter_magnitude, 1)
        # self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        # self.spatial_filter.set_option(rs.option.filter_smooth_delta, 10)
        # self.spatial_filter.set_option(rs.option.holes_fill, 0)
        # self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        # self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.get_logger().info("✅ BB Detector ready")

    def publish_static_camera_optical_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.camera_optical_frame

        #  optical frame origin == camera center
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # ROS optical frame convention
        q = tf_transformations.quaternion_from_euler(-math.pi/2, 0.0, -math.pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.static_tf_broadcaster.sendTransform(t)

    def publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.camera_frame

        # Mounting offset
        t.transform.translation.x = 0.25
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.30

        # Pitch rotation about Y-axis
        pitch_angle_rad = np.pi * (self.pitch_angle /180.0)
        q_pitch = tf_transformations.quaternion_from_euler(
            0.0,
            pitch_angle_rad,
            0.0
        )
        q_mount = tf_transformations.quaternion_from_euler(
            0.0,
            -math.pi / 2,
            0.0
        )
        q = tf_transformations.quaternion_multiply(q_mount, q_pitch)


        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def deproject_pixel_to_3d(self, u, v, depth_m):
        intr = self.color_intrinsics
        fx = intr.fx
        fy = intr.fy
        cx = intr.ppx
        cy = intr.ppy

        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m
        return X, Y, Z

    def post_process_depth_frame(self, depth_frame):
        frame = depth_frame
        # frame = self.to_disparity.process(frame)
        # frame = self.spatial_filter.process(frame)
        # frame = self.temporal_filter.process(frame)
        # frame = self.from_disparity.process(frame)
        # frame = self.hole_filling_filter.process(frame)
        return frame

    def build_depth_mask(self, depth_image, roi_bbox=None):
        if depth_image is None or depth_image.size == 0:
            return None
        h, w = depth_image.shape
        base_mask = ((depth_image > 0) &
                     (depth_image <= self.depth_mask_threshold)).astype(np.uint8) * 255

        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            if x2 > x1 and y2 > y1:
                x2_idx = min(x2 + 1, w)
                y2_idx = min(y2 + 1, h)

                depth_roi = depth_image[y1:y2_idx, x1:x2_idx]
                roi_valid = depth_roi[depth_roi > 0]

                if roi_valid.size > 0:
                    percentile = np.clip(self.depth_mask_percentile, 0.0, 100.0)
                    d_min = float(np.percentile(roi_valid, percentile))
                    d_min = min(d_min, self.depth_mask_threshold)

                    margin = max(0.0, self.depth_mask_margin)
                    dynamic_mask_roi = ((depth_roi >= d_min) &
                                        (depth_roi <= d_min + margin)).astype(np.uint8) * 255

                    dynamic_mask_roi = cv2.morphologyEx(
                        dynamic_mask_roi, cv2.MORPH_CLOSE, self.kernel
                    )
                    mask_roi = cv2.dilate(dynamic_mask_roi, self.kernel, iterations=1)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)

                    mask = np.zeros_like(depth_image, dtype=np.uint8)
                    mask[y1:y2_idx, x1:x2_idx] = mask_roi
                    return mask


        # Fallback
        roi_center = depth_image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        roi_valid_center = roi_center[roi_center > 0]
        if roi_valid_center.size == 0:
            roi_valid_center = depth_image[depth_image > 0]
        if roi_valid_center.size == 0:
            return base_mask

        percentile = np.clip(self.depth_mask_percentile, 0.0, 100.0)
        d_min = float(np.percentile(roi_valid_center, percentile))
        d_min = min(d_min, self.depth_mask_threshold)

        margin = max(5.0, self.depth_mask_margin)
        dynamic_mask = ((depth_image >= d_min) &
                        (depth_image <= d_min + margin)).astype(np.uint8) * 255

        dynamic_mask = cv2.morphologyEx(dynamic_mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(dynamic_mask, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    def expand_bbox(self, bbox, margin_ratio, image_shape):
        if bbox is None:
            return None
        margin_ratio = max(0.0, margin_ratio)
        h, w = image_shape
        if hasattr(bbox, "astype"):
            x1, y1, x2, y2 = bbox.astype(float)
        else:
            x1, y1, x2, y2 = map(float, bbox)
        width = x2 - x1
        height = y2 - y1
        mx = width * margin_ratio
        my = height * margin_ratio
        x1 = int(max(0, x1 - mx))
        y1 = int(max(0, y1 - my))
        x2 = int(min(w - 1, x2 + mx))
        y2 = int(min(h - 1, y2 + my))
        return (x1, y1, x2, y2)

    def bbox_from_center(self, center, half_size, image_shape):
        if center is None:
            return None
        h, w = image_shape
        cx, cy = center
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(w - 1, cx + half_size))
        y2 = int(min(h - 1, cy + half_size))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def find_depth_contour(self, depth_mask, bbox=None):
        if depth_mask is None:
            return None

        h, w = depth_mask.shape

        if bbox is None:
            search_mask = depth_mask.copy()
            x_offset = 0
            y_offset = 0
        else:
            x1, y1, x2, y2 = bbox
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            if x2 <= x1 or y2 <= y1:
                return None

            x2_idx = min(x2 + 1, w)
            y2_idx = min(y2 + 1, h)

            search_mask = depth_mask[y1:y2_idx, x1:x2_idx].copy()
            x_offset = x1
            y_offset = y1

        contours, _ = cv2.findContours(
            search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.depth_min_contour_area:
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        cx_local = int(M["m10"] / M["m00"])
        cy_local = int(M["m01"] / M["m00"])

        cx = cx_local + x_offset
        cy = cy_local + y_offset

        contour[:, 0, 0] += x_offset  # x
        contour[:, 0, 1] += y_offset  # y

        return {
            "center": (cx, cy),
            "contour": contour,
            "mask": search_mask,
        }

    def callback(self):
        self.draw = True if self.annotate and self.frame_rate % 15 else False
        t0 = time.perf_counter()
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        t1 = time.perf_counter()

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            self.get_logger().warn("No frames received!")
            return

        depth_frame = self.post_process_depth_frame(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())      # uint16 depth in mm
        frame = np.asanyarray(color_frame.get_data())            # RGB aligned to depth

        t2 = time.perf_counter()

        detections, boxes, _, _ = self.detect_objects(frame, depth_image)
        target_detection = next((d for d in detections if d["class_id"] == self.target_class_id), None)

        object_detected_raw = target_detection is not None

        t3 = time.perf_counter()

        if object_detected_raw:
            self.detect_yes += 1
            self.detect_no = 0
        else:
            self.detect_no += 1
            self.detect_yes = 0

        object_detected_stable = (self.detect_yes >= self.DETECT_YES_THRESH)
        object_lost_stable     = (self.detect_no  >= self.DETECT_NO_THRESH)
        # DEBUG: CLASSES
        for det in detections:
            self.get_logger().info(
                f"det: class_id={det['class_id']} "
                f"name={det['class_name']} "
                f"score={det['score']:.2f} "
                f"center={det['center']}"
            )
        det_bbox = det_center = det_depth = None
        if target_detection:
            det_center = target_detection["center"]
            det_depth  = target_detection["depth"]
            det_bbox   = target_detection["bbox"]

            if self.draw:
                cv2.circle(frame, (int(det_center[0]), int(det_center[1])), 4, COL_TARGET, -1)

        if self.draw:
            for box in boxes:
                x1,y1,x2,y2 = box.astype(int)
                cv2.rectangle(frame, (x1,y1), (x2,y2), COL_WHITE, 2)

        contour_info = None
        depth_mask = None
        mask_bbox_draw = None

        # STATE: DETECT
        if self.state == State.DETECT:

            if object_detected_stable and (det_depth is not None) and det_depth <= self.depth_threshold:
                self.state = State.SEARCH
                self.get_logger().info("STATE → SEARCH")
                self.prev_contour_center = None
                self.contour_yes = self.contour_no = 0
                self.track_stable_start = None

        # STATE: SEARCH
        elif self.state == State.SEARCH:

            search_bbox = None
            if det_bbox is not None:
                search_bbox = self.expand_bbox(det_bbox, self.search_bbox_margin, depth_image.shape)
            if search_bbox is None:
                search_bbox = (0, 0, self.image_w * 0.8  , self.image_h - 1)

            depth_mask = self.build_depth_mask(depth_image, search_bbox)
            mask_bbox_draw = search_bbox
            contour_info = self.find_depth_contour(depth_mask, search_bbox)

            if contour_info is not None:
                self.prev_contour_center = contour_info["center"]
                self.contour_yes += 1
                self.contour_no = 0
            else:
                self.contour_no += 1
                self.contour_yes = 0

            contour_found_stable = (self.contour_yes >= self.CONTOUR_YES_THRESH)

            # Losing object BEFORE contour is confirmed → go back
            if object_lost_stable and not contour_found_stable:
                self.get_logger().info("STATE → DETECT (lost before confirming contour)")
                self.state = State.DETECT
                self.prev_contour_center = None
                self.contour_yes = self.contour_no = 0

            # Contour confirmed AND object lost → go to TRACK
            elif contour_found_stable and object_lost_stable:
                self.get_logger().info("STATE → TRACK (contour confirmed)")
                self.state = State.TRACK
                self.contour_no = 0
                self.track_stable_start = None

        # =====================================================
        # STATE: TRACK
        # =====================================================
        elif self.state == State.TRACK:

            if object_detected_stable:
                self.get_logger().info("STATE → DETECT (object back in view)")
                self.state = State.DETECT
                self.prev_contour_center = None
                self.contour_yes = self.contour_no = 0
                self.track_stable_start = None
            else:
                track_bbox = None
                if det_bbox is not None:
                    track_bbox = self.expand_bbox(det_bbox, self.search_bbox_margin, depth_image.shape)
                elif self.prev_contour_center is not None:
                    track_bbox = self.bbox_from_center(self.prev_contour_center, self.track_window_px, depth_image.shape)
                if track_bbox is None:
                    track_bbox = self.bbox_from_center((self.image_cx, self.image_cy), self.track_window_px, depth_image.shape)

                depth_mask = self.build_depth_mask(depth_image, track_bbox)
                mask_bbox_draw = track_bbox
                contour_info = self.find_depth_contour(depth_mask, track_bbox)

                if contour_info is not None:
                    self.prev_contour_center = contour_info["center"]
                    self.contour_yes += 1
                    self.contour_no = 0
                    if self.track_stable_start is None:
                        self.track_stable_start = time.time()
                else:
                    self.contour_no += 1
                    self.contour_yes = 0
                    self.track_stable_start = None

                contour_lost_stable = (self.contour_no >= self.CONTOUR_NO_THRESH)
                if contour_lost_stable:
                    self.get_logger().info("STATE → DETECT (contour lost)")
                    self.state = State.DETECT
                    self.prev_contour_center = None
                    self.contour_yes = self.contour_no = 0
                    self.track_stable_start = None
                elif contour_info is not None and self.track_stable_start is not None:
                    if (time.time() - self.track_stable_start) >= self.finish_time:
                        self.get_logger().info(f"STATE → FINISH (contour stable > {self.finish_time})")
                        self.state = State.FINISH
                        self.track_stable_start = None
                        self.contour_yes = self.contour_no = 0

        elif self.state == State.FINISH:
            if object_detected_stable:
                self.get_logger().info("STATE → DETECT (object detected during FINISH)")
                self.state = State.DETECT
                self.prev_contour_center = None
                self.contour_yes = self.contour_no = 0
                self.track_stable_start = None

        t4 = time.perf_counter()
        # Draw the ROI used for depth mask calculations
        if mask_bbox_draw is not None and self.draw:
            x1, y1, x2, y2 = mask_bbox_draw
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COL_SEARCH, 2)

        # =====================================================
        # SERVO_CONTROL
        # =====================================================
        target_point = None
        if contour_info is not None:
            target_point = contour_info["center"]
            if self.draw:
                cv2.drawContours(frame, [contour_info["contour"]], -1, COL_CONTOUR, 2)
                cv2.circle(frame, (int(target_point[0]), int(target_point[1])), 4, COL_CONTOUR, -1)
        elif det_center is not None:
            target_point = det_center

        if target_point is not None:
            cx, cy = target_point
            cx = int(cx)
            cy = int(cy)
            y_err = self.image_cy - cy 
            x_err = self.image_cx - cx
            COL = (0,0,255)
            if abs(y_err) > self.y_max:
                self.pitch_angle += self.pitch_step * (y_err)/self.image_h
                self.pitch_angle = np.clip(self.pitch_angle, self.pitch_min, self.pitch_max)
                COL = (255,0,0)
            if abs(x_err) > self.x_max:
                self.yaw_angle += 2 * (x_err)/self.image_w
                COL = (255,0,0)

            if self.draw:
                cv2.line(frame, (self.image_cx,cy), (cx,cy), COL, 1) # horizontal
                cv2.line(frame, (self.image_cx,self.image_cy), (self.image_cx,cy), COL, 1)

        if self.pitch_angle != self.pitch_angle_prev:
            self.pitch_angle_prev = self.pitch_angle
            msg_out = Float64()
            msg_out.data = self.pitch_angle
            self.servo_pub.publish(msg_out)
            self.publish_camera_tf()

        if self.yaw_angle != self.yaw_angle_prev:
            self.yaw_angle_prev = self.yaw_angle
            msg_out = Float64()
            msg_out.data = self.yaw_angle
            self.yaw_pub.publish(msg_out)

        if self.state != self.prev_state:
          msg = String()
          msg.data = self.state.name
          self.state_pub.publish(msg)
          self.prev_state = self.state
        now = time.time()
        fps = 1.0/(now - self.prev_t)
        self.prev_t = now

        if self.draw:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)
            cv2.putText(frame, "S: "+STATES[self.state.value], (120,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)

        # TODO: only publish some frames
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        self.publish_detections(detections, contour_info, depth_image)
        t5 = time.perf_counter()
        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            self.frame_counter = 0
            self.get_logger().info(
                f"timings: capture+align={(t1-t0)*1000:.1f}ms, "
                f"depth_filter={(t2-t1)*1000:.1f}ms, "
                f"detect={(t3-t2)*1000:.1f}ms, "
                f"state_machine={(t4-t3)*1000:.1f}ms"
                f"rest={(t5-t4)*1000:.1f}ms"
            )


    def detect_objects(self, frame, depth_array):
        """
        Run RF-DETR to obtain detections and enrich them with centers/depth.
        Returns (detections_list, boxes, scores, labels)
        """
        scores, labels, boxes, _ = self.model.predict(
            frame, allowed_cids=set(self.target_labels)
        )
        detections = []
        for idx in range(len(scores)):
            bbox = boxes[idx]
            cx = float((bbox[0] + bbox[2]) * 0.5)
            cy = float((bbox[1] + bbox[3]) * 0.5)
            depth = self.get_depth_at(depth_array, cx, cy)
            detections.append(
                {
                    "class_id": int(labels[idx]),
                    "class_name": COCO_CLASSES.get(int(labels[idx]), "unknown"),
                    "score": float(scores[idx]),
                    "bbox": bbox,
                    "center": (cx, cy),
                    "depth": depth,
                }
            )
        return detections, boxes, scores, labels

    def publish_detections(self, detections, contour_info=None, depth_array=None):
        target_cx = None
        target_cy = None
        target_depth = None

        if contour_info is not None and depth_array is not None:
            cx, cy = contour_info["center"]
            d = self.get_depth_at(depth_array, cx, cy)
            if d is not None and not math.isnan(d):
                target_cx, target_cy, target_depth = cx, cy, d

        # Otherwise fall back to first detection with valid depth
        if target_cx is None and detections:
            for det in detections:
                if det["depth"] is not None and not math.isnan(det["depth"]):
                    target_cx = det["center"][0]
                    target_cy = det["center"][1]
                    target_depth = det["depth"]
                    break

        if target_cx is None or target_depth is None:
            return

        X_cam, Y_cam, Z_cam = self.deproject_pixel_to_3d(
            target_cx,
            target_cy,
            target_depth,
        )

        point_cam = PointStamped()
        point_cam.header.stamp = self.get_clock().now().to_msg()
        point_cam.header.frame_id = self.camera_optical_frame
        point_cam.point.x = float(X_cam)/ 1000.0
        point_cam.point.y = float(Y_cam)/ 1000.0
        point_cam.point.z = float(Z_cam)/ 1000.0
        self.point_cam_pub.publish(point_cam)



    def get_depth_at(self, depth_array, cx, cy):
        """
        Look up depth (in meters) at floating pixel coordinates.
        """
        if depth_array is None or cx is None or cy is None:
            return None

        h, w = depth_array.shape[:2]
        x = int(np.clip(round(cx), 0, w - 1))
        y = int(np.clip(round(cy), 0, h - 1))

        depth_value = depth_array[y, x]
        if isinstance(depth_value, np.ndarray):
            depth_value = depth_value.item()

        if depth_value == 0 or (isinstance(depth_value, float) and math.isnan(depth_value)):
            return None

        if depth_array.dtype == np.uint16:
            return float(depth_value) * self.depth_scale
        return float(depth_value)

    def destroy_node(self):
        del self.model
        gc.collect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RFDetrNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
