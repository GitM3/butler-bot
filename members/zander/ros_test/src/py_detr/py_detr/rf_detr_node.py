import gc
import math
import time
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

COL_WHITE = (255, 255, 255)
COL_TARGET = (0, 200, 255)
COL_ELLIPSE = (0, 255, 0)

# Check for updates: https://github.com/roboflow/rf-detr/blob/main/rfdetr/util/coco_classes.py
COCO_CLASSES = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
}

class State(Enum):
    DETECT = 1
    ELLIPSE_SEARCH = 2
    ELLIPSE_TRACK = 3

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
        self.rgb_sub   = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/depth/image_rect_raw")  
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.callback)

        self.image_pub = self.create_publisher(Image, "/camera/annotated", 10)
        self.debug_image_pub = self.create_publisher(Image, "/camera/annotated_debug", 10)
        self.target_pub = self.create_publisher(Float64MultiArray, "/target/center", 10)
        self.enable_ellipse_debug_viz = True
        self.default_ellipse_params = {
            "bottom_ratio": 0.5,
            "min_contour_area": 1000.0,
            "canny_thresh1": 10.0,
            "canny_thresh2": 80.0,
            "gaussian_kernel": 5.0,
            "max_axis_scale": 1.5,
            "min_axis_length": 5.0,
            "axis_ratio_min": 0.55,
        }
        self.ellipse_param_overrides = None
        self.ellipse_param_name_order = [
            "bottom_ratio",
            "min_contour_area",
            "canny_thresh1",
            "canny_thresh2",
            "gaussian_kernel",
            "max_axis_scale",
            "min_axis_length",
            "axis_ratio_min",
        ]
        self.ellipse_debug_param_sub = self.create_subscription(
            Float64MultiArray,
            "/ellipse/params",
            self._ellipse_debug_param_callback,
            10,
        )

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
        self.declare_parameter("depth_scale", 0.001)
        self.depth_scale = (
            self.get_parameter("depth_scale").get_parameter_value().double_value
        )
        self.model = RFDETR_ONNX(self.model_path)
        self.get_logger().info("✅ RF-DETR ready")
        self.prev_t = time.time()
        self.state = State.DETECT
        self.depth_threshold = 2   # meters — adjust as needed
        self.last_bbox = None
        self.last_ellipse_bbox = None
        # Simple constant-velocity tracking state (pixels/sec)
        self.ev_center = None  # last ellipse center (x, y)
        self.ev_vel = np.array([0.0, 0.0], dtype=float)
        self.ev_time = None
        # Debug annotations for tracking
        self._dbg_track_mode = None
        self._dbg_pred_roi = None
        self._dbg_fallback_roi = None
        self._dbg_last_center = None
        self._dbg_proj_center = None
        self._dbg_velocity = None
        # Tracking tunables
        self.declare_parameter("track_roi_scale", 2.0)  # scale of last ellipse bbox
        self.declare_parameter("track_roi_margin_px", 40.0)  # extra pixels added to each side
        self.declare_parameter("track_vel_alpha", 0.8)  # EMA for velocity smoothing
        self.declare_parameter("track_max_age", 1.0)  # seconds; fall back if older
        self.track_roi_scale = float(self.get_parameter("track_roi_scale").get_parameter_value().double_value)
        self.track_roi_margin_px = float(self.get_parameter("track_roi_margin_px").get_parameter_value().double_value)
        self.track_vel_alpha = float(self.get_parameter("track_vel_alpha").get_parameter_value().double_value)
        self.track_max_age = float(self.get_parameter("track_max_age").get_parameter_value().double_value)
        now = time.time()
        self.last_seen_object_time = now
        self.last_seen_ellipse_time = now
        self.depth_array = None
        self.depth_encoding = ""

    def callback(self, rgb_msg, depth_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        self.depth_img = depth_msg  # Store for depth lookup
        try:
            self.depth_array = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().warning(f"Failed to convert depth image: {exc}")
            self.depth_array = None
        self.depth_encoding = depth_msg.encoding
        # Clear per-frame debug annotations
        self._dbg_track_mode = None
        self._dbg_pred_roi = None
        self._dbg_fallback_roi = None
        self._dbg_last_center = None
        self._dbg_proj_center = None
        self._dbg_velocity = None

        annotated = frame.copy()
        current_time = time.time()

        detections, boxes, scores, labels = self.detect_objects(frame, self.depth_array)
        target_detection = next(
            (det for det in detections if det["class_id"] == self.target_class_id),
            None,
        )
        object_detected = target_detection is not None

        det_cx = det_cy = det_depth = None
        det_bbox = None
        if target_detection:
            det_cx, det_cy = target_detection["center"]
            det_depth = target_detection["depth"]
            det_bbox = target_detection["bbox"]
        tracked_ellipse_detection = None

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_WHITE, 2)

        if det_cx is not None and det_cy is not None:
            cv2.circle(annotated, (int(det_cx), int(det_cy)), 4, COL_TARGET, -1)

        DETECT_LOST_THRESHOLD = 0.50
        ELLIPSE_LOST_THRESHOLD = 0.50

        if self.state == State.DETECT:
            if object_detected:
                self.last_seen_object_time = current_time
                self.last_bbox = det_bbox
                #self.get_logger().info(f"Object found at [{det_cx,det_cy,det_depth}]")
                if det_depth is not None and det_depth < self.depth_threshold:
                    self.state = State.ELLIPSE_SEARCH
                    self.get_logger().info("STATE → ELLIPSE_SEARCH")
        elif self.state == State.ELLIPSE_SEARCH:
            if object_detected:
                self.last_seen_object_time = current_time
                self.last_bbox = det_bbox

                ellipse_found = False
                e_cx = e_cy = None
                ellipse_params = None
                if det_bbox is not None:
                    ellipse_found, e_cx, e_cy, ellipse_params = self.find_ellipse(
                        frame, det_bbox
                    )
                if ellipse_found:
                    self.last_seen_ellipse_time = current_time
                    if ellipse_params is not None:
                        self._update_last_ellipse_bbox(frame.shape, ellipse_params)
                        self._update_ellipse_motion(ellipse_params)
                        center, axes, angle = ellipse_params
                        cv2.ellipse(
                            annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2
                        )

            else:
                if self.last_seen_object_time is None:
                    self.last_seen_object_time = current_time
                time_missing_obj = current_time - self.last_seen_object_time

                ellipse_found = False
                e_cx = e_cy = None
                ellipse_params = None
                if self.last_bbox is not None:
                    ellipse_found, e_cx, e_cy, ellipse_params = self.find_ellipse(
                        frame, self.last_bbox
                    )
                    if ellipse_found:
                        self.last_seen_ellipse_time = current_time
                        if ellipse_params is not None:
                            self._update_last_ellipse_bbox(frame.shape, ellipse_params)
                            self._update_ellipse_motion(ellipse_params)
                            center, axes, angle = ellipse_params
                            cv2.ellipse(
                                annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2
                            )

                ellipse_lost_time = (
                    current_time - self.last_seen_ellipse_time
                    if self.last_seen_ellipse_time is not None
                    else float("inf")
                )

                if ellipse_found and time_missing_obj > DETECT_LOST_THRESHOLD:
                    self.state = State.ELLIPSE_TRACK
                    self.get_logger().info("STATE → ELLIPSE_TRACK")
                elif not ellipse_found and ellipse_lost_time > ELLIPSE_LOST_THRESHOLD:
                    self.state = State.DETECT
                    self.last_ellipse_bbox = None
                    self.get_logger().info("STATE → DETECT")

        elif self.state == State.ELLIPSE_TRACK:
            if object_detected:
                    self.state = State.ELLIPSE_SEARCH
                    self.last_ellipse_bbox = None
                    self.get_logger().info("STATE → ELLIPSE_SEARCH")

            ellipse_found, e_cx, e_cy, ellipse_params = self.track_ellipse(frame)
            if ellipse_found:
                self.last_seen_ellipse_time = current_time

                depth_at_e = self.get_depth_at(self.depth_array, e_cx, e_cy)
                if ellipse_params is not None:
                    center, axes, angle = ellipse_params
                    cv2.ellipse(annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2)
                    self._update_last_ellipse_bbox(frame.shape, ellipse_params)
                    self._update_ellipse_motion(ellipse_params)
                else:
                    cv2.circle(
                        annotated,
                        (int(e_cx), int(e_cy)),
                        5,
                        COL_ELLIPSE,
                        -1,
                    )
                tracked_ellipse_detection = {
                    "class_id": self.target_class_id,
                    "class_name": f"{self.target_class_name}_ellipse",
                    "score": 1.0,
                    "bbox": np.array([e_cx, e_cy, e_cx, e_cy], dtype=float),
                    "center": (float(e_cx), float(e_cy)),
                    "depth": depth_at_e,
                }

            else:
                ellipse_lost_time = (
                    current_time - self.last_seen_ellipse_time
                    if self.last_seen_ellipse_time is not None
                    else float("inf")
                )
                # Lost ellipse for too long → back to DETECT
                if ellipse_lost_time > ELLIPSE_LOST_THRESHOLD:
                    self.state = State.DETECT
                    self.last_ellipse_bbox = None
                    self.get_logger().info("STATE → DETECT")

        now = time.time()
        fps = 1.0 / (now - self.prev_t)
        self.prev_t = now

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COL_WHITE,
            2,
        )

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "rgb8"))
        publish_detections = list(detections)
        if tracked_ellipse_detection is not None:
            publish_detections.insert(0, tracked_ellipse_detection)
        self.publish_detections(publish_detections)
    
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

    def find_ellipse(self, frame, det_bbox, bottom_ratio=0.3):
        """
        Depth-based bottom finder.
        Returns ellipse_found, center_x, center_y, ellipse_params
        ellipse_params: (center, axes, angle) OR None
        """

        if det_bbox is None or self.depth_array is None:
            return False, None, None, None

        x1, y1, x2, y2 = map(int, det_bbox)
        h_frame, w_frame = frame.shape[:2]

        # Expand ROI slightly (like original code)
        margin_y = int((y2 - y1) * 0.1)
        margin_x = int((x2 - x1) * 0.1)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w_frame - 1, x2 + margin_x)
        y2 = min(h_frame - 1, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return False, None, None, None

        # Bottom region of bounding box
        h = y2 - y1
        roi_y1 = y1 + int(h * (1.0 - bottom_ratio))

        depth_roi = self.depth_array[roi_y1:y2, x1:x2].astype(np.float32)

        if depth_roi.size == 0:
            return False, None, None, None

        # Remove invalid depths
        valid = np.isfinite(depth_roi) & (depth_roi > 0)
        if not np.any(valid):
            return False, None, None, None

        depth_valid = depth_roi.copy()
        depth_valid[~valid] = np.nan

        # Smooth to reduce noise
        depth_blur = cv2.GaussianBlur(depth_valid, (5, 5), 0)

        # Compute mask of near-minimum depth.
        # Bottle bottom is typically the shallowest surface.
        # Use 10th percentile to avoid outliers.
        valid_depths = depth_blur[valid]
        depth_thresh = np.nanpercentile(valid_depths, 10)
        mask = depth_blur <= depth_thresh

        if np.sum(mask) < 20:  # too small?
            return False, None, None, None

        ys, xs = np.where(mask)

        # Compute pixel centroid in ROI
        cx_roi = float(np.mean(xs))
        cy_roi = float(np.mean(ys))

        # Convert back to full image coords
        world_cx = int(x1 + cx_roi)
        world_cy = int(roi_y1 + cy_roi)

        # Optional: approximate ellipse parameters
        # (Using std dev of cluster as axes)
        ax = float(np.std(xs)) * 1.5
        ay = float(np.std(ys)) * 1.5
        ax = max(1.0, ax)
        ay = max(1.0, ay)

        ellipse_center = (world_cx, world_cy)
        ellipse_axes = (int(ax), int(ay))
        ellipse_angle = 0.0  # We don't estimate rotation

        ellipse_params = (ellipse_center, ellipse_axes, ellipse_angle)

        return True, world_cx, world_cy, ellipse_params

    def _ellipse_debug_param_callback(self, msg: Float64MultiArray):
        values = list(msg.data)
        overrides = {}
        for idx, name in enumerate(self.ellipse_param_name_order):
            if idx < len(values):
                val = values[idx]
                try:
                    if math.isnan(val):
                        continue
                except TypeError:
                    pass
                overrides[name] = float(val)
        if overrides:
            if self.ellipse_param_overrides is None:
                self.ellipse_param_overrides = {}
            self.ellipse_param_overrides.update(overrides)
            self.get_logger().info(
                "Updated ellipse debug params: "
                + ", ".join(f"{k}={v:.2f}" for k, v in overrides.items())
            )
        else:
            self.ellipse_param_overrides = None
            self.get_logger().info("Cleared ellipse debug params; falling back to defaults")

    def _get_ellipse_params(self):
        params = dict(self.default_ellipse_params)
        if self.ellipse_param_overrides:
            params.update(self.ellipse_param_overrides)
        return params

    def _publish_ellipse_debug(
        self,
        frame,
        roi,
        blurred,
        edges,
        candidate_ellipses,
        best_ellipse,
        bbox,
        roi_y1,
        canny_thresh,
        state_name,
    ):
        if not self.enable_ellipse_debug_viz or self.debug_image_pub is None:
            return
        if roi is None or roi.size == 0:
            return

        x1, y1, x2, y2 = bbox
        font = cv2.FONT_HERSHEY_SIMPLEX

        def _to_rgb(image):
            if image is None:
                return None
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image.copy()

        roi_color = _to_rgb(roi)
        blurred_rgb = _to_rgb(blurred)
        edges_rgb = _to_rgb(edges)
        roi_candidates = roi_color.copy() if roi_color is not None else None
        frame_overlay = frame.copy()
        cv2.rectangle(frame_overlay, (x1, roi_y1), (x2, y2), (255, 0, 0), 1)

        # Tracking overlays (projected ROI, last bbox, motion arrow)
        try:
            # Last ellipse bbox
            if self.last_ellipse_bbox is not None:
                lx1, ly1, lx2, ly2 = map(int, self.last_ellipse_bbox)
                cv2.rectangle(frame_overlay, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1)
                cv2.putText(frame_overlay, "last ellipse bbox", (lx1, max(0, ly1 - 6)), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            # Predicted ROI
            if self._dbg_pred_roi is not None:
                px1, py1, px2, py2 = map(int, self._dbg_pred_roi)
                cv2.rectangle(frame_overlay, (px1, py1), (px2, py2), (255, 0, 255), 2)
                cv2.putText(frame_overlay, "pred ROI", (px1, max(0, py1 - 6)), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            # Fallback ROI
            if self._dbg_fallback_roi is not None:
                fx1, fy1, fx2, fy2 = map(int, self._dbg_fallback_roi)
                cv2.rectangle(frame_overlay, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                cv2.putText(frame_overlay, "fb ROI", (fx1, max(0, fy1 - 6)), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # Motion vectors
            if self.ev_center is not None:
                lc = (int(round(self.ev_center[0])), int(round(self.ev_center[1])))
                cv2.circle(frame_overlay, lc, 3, (0, 255, 0), -1)
            if self._dbg_proj_center is not None and self._dbg_last_center is not None:
                p = (int(round(self._dbg_proj_center[0])), int(round(self._dbg_proj_center[1])))
                l = (int(round(self._dbg_last_center[0])), int(round(self._dbg_last_center[1])))
                cv2.arrowedLine(frame_overlay, l, p, (255, 0, 255), 2, tipLength=0.2)
                cv2.circle(frame_overlay, p, 3, (255, 0, 255), -1)
                if self._dbg_velocity is not None:
                    vx, vy = self._dbg_velocity
                    cv2.putText(frame_overlay, f"v=({vx:.1f},{vy:.1f})", (p[0] + 6, p[1] - 6), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            # Mode label
            if self._dbg_track_mode:
                cv2.putText(frame_overlay, f"track: {self._dbg_track_mode}", (10, 50), font, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

        sorted_candidates = sorted(candidate_ellipses, key=lambda e: e["area"], reverse=True)
        for idx, cand in enumerate(sorted_candidates):
            ellipse = cand["ellipse"]
            center_local = (int(round(ellipse[0][0])), int(round(ellipse[0][1])))
            axes = (
                max(1, int(round(ellipse[1][0] * 0.5))),
                max(1, int(round(ellipse[1][1] * 0.5))),
            )
            color = (0, 255, 0) if best_ellipse is not None and ellipse == best_ellipse else (0, 215, 255)
            thickness = 2 if best_ellipse is not None and ellipse == best_ellipse else 1
            label = f"#{idx+1}"

            if roi_candidates is not None:
                cv2.ellipse(roi_candidates, center_local, axes, ellipse[2], 0, 360, color, thickness)
                cv2.putText(
                    roi_candidates,
                    label,
                    (center_local[0] + 4, center_local[1] - 4),
                    font,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            world_center = (
                int(round(x1 + center_local[0])),
                int(round(roi_y1 + center_local[1])),
            )
            cv2.ellipse(frame_overlay, world_center, axes, ellipse[2], 0, 360, color, thickness)
            cv2.putText(
                frame_overlay,
                label,
                (world_center[0] + 4, world_center[1] - 4),
                font,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        tiles = []
        captions = []
        if roi_color is not None:
            tiles.append(roi_color)
            captions.append("ROI (color)")
        if blurred_rgb is not None:
            tiles.append(blurred_rgb)
            captions.append("Blurred gray")
        if edges_rgb is not None:
            tiles.append(edges_rgb)
            captions.append(
                f"Canny ({canny_thresh[0]}, {canny_thresh[1]})"
            )
        if roi_candidates is not None:
            tiles.append(roi_candidates)
            captions.append(
                f"Candidates: {len(sorted_candidates)}"
            )
        tiles.append(frame_overlay)
        captions.append("Frame overlay")

        tile_w, tile_h = 320, 240
        prepared_tiles = []
        for img, caption in zip(tiles, captions):
            resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                resized,
                caption,
                (8, 22),
                font,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            prepared_tiles.append(resized)

        if not prepared_tiles:
            return

        cols = 3
        rows = int(math.ceil(len(prepared_tiles) / cols))
        blank_tile = np.zeros((tile_h, tile_w, 3), dtype=prepared_tiles[0].dtype)
        mosaic_rows = []
        tile_idx = 0
        for _ in range(rows):
            row_tiles = []
            for _ in range(cols):
                if tile_idx < len(prepared_tiles):
                    row_tiles.append(prepared_tiles[tile_idx])
                    tile_idx += 1
                else:
                    row_tiles.append(blank_tile.copy())
            mosaic_rows.append(np.hstack(row_tiles))
        debug_canvas = np.vstack(mosaic_rows)

        if state_name:
            cv2.putText(
                debug_canvas,
                f"State: {state_name}",
                (10, 30),
                font,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        try:
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_canvas, "rgb8")
            )
        except Exception:
            pass

    def _update_last_ellipse_bbox(self, frame_shape, ellipse_params):
        if ellipse_params is None:
            return
        center, axes, _ = ellipse_params
        h, w = frame_shape[:2]
        x1 = int(np.clip(center[0] - axes[0], 0, w - 1))
        y1 = int(np.clip(center[1] - axes[1], 0, h - 1))
        x2 = int(np.clip(center[0] + axes[0], 0, w - 1))
        y2 = int(np.clip(center[1] + axes[1], 0, h - 1))
        self.last_ellipse_bbox = (x1, y1, x2, y2)

    def publish_detections(self, detections):
        """
        Publish detections as a flattened Float64 array:
        [class_id, cx, cy, depth, class_id, ...]
        """
        msg = Float64MultiArray()
        payload = []
        for det in detections:
            depth_val = det["depth"] if det["depth"] is not None else math.nan
            payload.extend(
                [
                    float(det["class_id"]),
                    float(det["center"][0]),
                    float(det["center"][1]),
                    float(depth_val),
                ]
            )
        msg.data = payload
        self.target_pub.publish(msg)



    def track_ellipse(self, frame):
        """
        Constant-velocity ROI projection around last ellipse, with fallback.
        """
        h, w = frame.shape[:2]
        now = time.time()

        # If we have a recent ellipse and bbox, project center forward
        use_pred = self.last_ellipse_bbox is not None and self.ev_center is not None and self.ev_time is not None and (now - self.ev_time) <= self.track_max_age
        if use_pred:
            dt = max(0.0, now - self.ev_time)
            proj = np.array(self.ev_center, dtype=float) + self.ev_vel * dt
            # ROI size based on last ellipse bbox scaled and with margin
            x1_l, y1_l, x2_l, y2_l = self.last_ellipse_bbox
            bw = max(4, x2_l - x1_l)
            bh = max(4, y2_l - y1_l)
            scale = max(1.0, float(self.track_roi_scale))
            margin = max(0.0, float(self.track_roi_margin_px))
            roi_w = int(round(bw * scale + 2 * margin))
            roi_h = int(round(bh * scale + 2 * margin))
            cx = float(np.clip(proj[0], 0, w - 1))
            cy = float(np.clip(proj[1], 0, h - 1))
            rx1 = int(np.clip(round(cx - roi_w * 0.5), 0, w - 1))
            ry1 = int(np.clip(round(cy - roi_h * 0.5), 0, h - 1))
            rx2 = int(np.clip(round(cx + roi_w * 0.5), 0, w))
            ry2 = int(np.clip(round(cy + roi_h * 0.5), 0, h))
            # Debug annotate
            self._dbg_track_mode = "predicted"
            self._dbg_pred_roi = (rx1, ry1, rx2, ry2)
            self._dbg_last_center = tuple(self.ev_center) if self.ev_center is not None else None
            self._dbg_proj_center = (cx, cy)
            self._dbg_velocity = tuple(self.ev_vel) if self.ev_vel is not None else None
            if rx2 <= rx1 or ry2 <= ry1:
                use_pred = False
            else:
                found, ecx, ecy, eparams = self.find_ellipse(frame, (rx1, ry1, rx2, ry2), bottom_ratio=1.0)
                if found:
                    return True, ecx, ecy, eparams

        # Fallback: search around last bbox location (expanded)
        if self.last_ellipse_bbox is not None:
            x1, y1, x2, y2 = self.last_ellipse_bbox
            bw = max(4, x2 - x1)
            bh = max(4, y2 - y1)
            scale = max(1.5, float(self.track_roi_scale))
            margin = max(20.0, float(self.track_roi_margin_px))
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            roi_w = int(round(bw * scale + 2 * margin))
            roi_h = int(round(bh * scale + 2 * margin))
            rx1 = int(np.clip(round(cx - roi_w * 0.5), 0, w - 1))
            ry1 = int(np.clip(round(cy - roi_h * 0.5), 0, h - 1))
            rx2 = int(np.clip(round(cx + roi_w * 0.5), 0, w))
            ry2 = int(np.clip(round(cy + roi_h * 0.5), 0, h))
            self._dbg_track_mode = self._dbg_track_mode or "expanded"
            self._dbg_fallback_roi = (rx1, ry1, rx2, ry2)
            if rx2 > rx1 and ry2 > ry1:
                found, ecx, ecy, eparams = self.find_ellipse(frame, (rx1, ry1, rx2, ry2), bottom_ratio=1.0)
                if found:
                    return True, ecx, ecy, eparams

        # Last resort: brute-force whole frame
        self._dbg_track_mode = self._dbg_track_mode or "full"
        return self.find_ellipse(frame, (0, 0, w, h), bottom_ratio=1.0)

    def _update_ellipse_motion(self, ellipse_params):
        try:
            center, _, _ = ellipse_params
            cx, cy = float(center[0]), float(center[1])
        except Exception:
            return
        now = time.time()
        if self.ev_center is not None and self.ev_time is not None:
            dt = now - self.ev_time
            if dt > 1e-3:
                inst_v = (np.array([cx, cy], dtype=float) - np.array(self.ev_center, dtype=float)) / dt
                self.ev_vel = self.track_vel_alpha * self.ev_vel + (1.0 - self.track_vel_alpha) * inst_v
        self.ev_center = (cx, cy)
        self.ev_time = now

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
