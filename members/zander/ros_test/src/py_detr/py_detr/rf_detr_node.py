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
    DETECT = 0
    ELLIPSE_SEARCH = 1
    ELLIPSE_TRACK = 2
STATES = ["DETECT","E_SEARCH","E_TRACK"]

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
        self.target_pub = self.create_publisher(Float64MultiArray, "/target/center", 10)
        self.find_debug_pub = self.create_publisher(Image, "/ellipse/find_debug", 1)
        self.find_mask_pub = self.create_publisher(Image, "/ellipse/find_mask", 1)
        self.track_debug_pub = self.create_publisher(Image, "/ellipse/track_debug", 1)

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
        now = time.time()
        self.last_seen_object_time = now
        self.last_seen_ellipse_time = now
        self.depth_array = None
        self.prev_gray = None
        self.target_ellipse_keyframes = 0 # Keyframes of good target ellipse pairs.


    def callback(self, rgb_msg, depth_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        try:
            self.depth_array = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().warning(f"Failed to convert depth image: {exc}")
            self.depth_array = None

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

        KF_THRESHOLD = 5
        ELLIPSE_LOST_THRESHOLD = 0.50

        if object_detected:
            self.last_seen_object_time = current_time
            self.last_bbox = det_bbox
        

        ellipse_found = False
        e_cx = e_cy = None
        ellipse_params = None

        if self.state == State.DETECT:
                #self.get_logger().info(f"Object found at [{det_cx,det_cy,det_depth}]")
                if object_detected and det_depth is not None and det_depth < self.depth_threshold:
                    self.state = State.ELLIPSE_SEARCH
                    self.get_logger().info("STATE → ELLIPSE_SEARCH")
                    self.target_ellipse_keyframes = 0
        elif self.state == State.ELLIPSE_SEARCH:
            if object_detected:
                if det_bbox is not None:
                    ellipse_found, e_cx, e_cy, ellipse_params = self.find_ellipse(
                        frame, det_bbox
                    )
                if ellipse_found:
                    self.last_seen_ellipse_time = current_time
                    self.target_ellipse_keyframes += 1
                    if ellipse_params is not None:
                        self._update_last_ellipse_bbox(frame.shape, ellipse_params)
                        center, axes, angle = ellipse_params
                        cv2.ellipse(
                            annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2
                        )
            else:
                if self.target_ellipse_keyframes < KF_THRESHOLD:
                    self.state = State.DETECT
                    self.get_logger().info("STATE → DETECT (bad init)")
                elif self.last_bbox is not None:
                    ellipse_found, e_cx, e_cy, ellipse_params = self.find_ellipse(
                        frame, self.last_bbox
                    )
                    if ellipse_found:
                        self.last_seen_ellipse_time = current_time
                        if ellipse_params is not None:
                            self._update_last_ellipse_bbox(frame.shape, ellipse_params)
                            center, axes, angle = ellipse_params
                            cv2.ellipse(
                                annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2
                            )
                        self.state = State.ELLIPSE_TRACK
                        self.get_logger().info("STATE → ELLIPSE_TRACK (stable init)")
                    else:
                        self.state = State.DETECT
                        self.last_ellipse_bbox = None
                        self.get_logger().info("STATE → DETECT (lost both)")
        elif self.state == State.ELLIPSE_TRACK:
            if object_detected:
                    self.state = State.DETECT
                    self.last_ellipse_bbox = None
                    self.get_logger().info("STATE → DETECT")

            ellipse_found, e_cx, e_cy, ellipse_params = self.track_ellipse(frame)
            if ellipse_found:
                self.last_seen_ellipse_time = current_time

                depth_at_e = self.get_depth_at(self.depth_array, e_cx, e_cy)
                if ellipse_params is not None:
                    center, axes, angle = ellipse_params
                    cv2.ellipse(annotated, center, axes, angle, 0, 360, COL_ELLIPSE, 2)
                    self._update_last_ellipse_bbox(frame.shape, ellipse_params)
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
        cv2.putText(
            annotated,
            "S: " +STATES[self.state.value],
            (120, 25),
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

    def find_ellipse(self, frame, det_bbox, bottom_ratio=1):
        """
        Depth-based bottom finder.
        Returns ellipse_found, center_x, center_y, ellipse_params
        ellipse_params: (center, axes, angle) OR None
        """

        if det_bbox is None or self.depth_array is None:
            return False, None, None, None

        x1, y1, x2, y2 = map(int, det_bbox)
        h_frame, w_frame = frame.shape[:2]

        # Expand ROI 
        margin_y = int((y2 - y1) * 0.3)
        margin_x = int((x2 - x1) * 0.3)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w_frame - 1, x2 + margin_x)
        y2 = min(h_frame - 1, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return False, None, None, None

        roi_color = frame[y1:y2, x1:x2]
        roi_copy = roi_color.copy() if roi_color.size else None
        mask_img = None
        depth_vis_img = None
        depth_thresh_val = None
        mask_count = 0
        ellipse_params = None
        world_cx = None
        world_cy = None
        found = False

        def finish():
            self._publish_find_debug(
                roi_copy,
                mask_img,
                depth_vis_img,
                ellipse_params,
                (x1, y1, x2, y2),
                depth_thresh_val,
                mask_count,
                found,
            )
            return found, world_cx, world_cy, ellipse_params

        # Bottom region of bounding box
        h = y2 - y1
        roi_y1 = y1 + int(h * (1.0 - bottom_ratio))
        if roi_y1 >= y2:
            return finish()

        depth_slice = self.depth_array[roi_y1:y2, x1:x2]
        if depth_slice.size == 0:
            return finish()
        depth_roi = depth_slice.astype(np.float32)

        # Remove invalid depths
        valid = np.isfinite(depth_roi) & (depth_roi > 0)
        if not np.any(valid):
            return finish()

        depth_valid = depth_roi.copy()
        depth_valid[~valid] = np.nan

        # Smooth to reduce noise
        depth_blur = cv2.GaussianBlur(depth_valid, (5, 5), 0)

        # Compute mask of near-minimum depth.
        valid_depths = depth_blur[valid]
        if valid_depths.size == 0:
            return finish()
        depth_thresh = np.nanpercentile(valid_depths, 10)
        depth_thresh_val = float(depth_thresh)
        mask = depth_blur <= depth_thresh
        mask_count = int(np.sum(mask))

        # Prepare visualization buffers for debugging
        mask_full = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        mask_full_offset = max(0, roi_y1 - y1)
        mask_full[mask_full_offset:, :] = (mask.astype(np.uint8) * 255)
        mask_img = mask_full

        depth_full = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        finite_depths = depth_roi[valid]
        if finite_depths.size > 0:
            d_min = float(np.min(finite_depths))
            d_max = float(np.max(finite_depths))
            span = max(1e-6, d_max - d_min)
            depth_norm = np.clip((depth_roi - d_min) / span, 0.0, 1.0)
            depth_norm[~np.isfinite(depth_norm)] = 0.0
            depth_full[mask_full_offset:, :] = (depth_norm * 255).astype(np.uint8)
        depth_vis_img = depth_full

        if mask_count < 20:  # too small?
            return finish()

        ys, xs = np.where(mask)

        # Compute pixel centroid in ROI
        cx_roi = float(np.mean(xs))
        cy_roi = float(np.mean(ys))

        # Convert back to full image coords
        world_cx = int(x1 + cx_roi)
        world_cy = int(roi_y1 + cy_roi)

        # Approximate ellipse parameters
        ax = float(np.std(xs)) * 1.5
        ay = float(np.std(ys)) * 1.5
        ax = max(1.0, ax)
        ay = max(1.0, ay)

        ellipse_center = (world_cx, world_cy)
        ellipse_axes = (int(ax), int(ay))
        ellipse_angle = 0.0

        ellipse_params = (ellipse_center, ellipse_axes, ellipse_angle)
        found = True
        return finish()

    def _publish_find_debug(
        self,
        roi,
        mask_img,
        depth_img,
        ellipse_params,
        bbox,
        depth_thresh,
        mask_pixels,
        found,
    ):
        if self.find_mask_pub is not None and mask_img is not None:
            try:
                mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
                self.find_mask_pub.publish(mask_msg)
            except Exception:
                pass

        if self.find_debug_pub is None or roi is None:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        overlay = roi.copy()
        x1, y1, _, _ = bbox
        if ellipse_params is not None:
            center, axes, angle = ellipse_params
            local_center = (int(center[0] - x1), int(center[1] - y1))
            cv2.ellipse(
                overlay,
                local_center,
                (max(1, axes[0]), max(1, axes[1])),
                angle,
                0,
                360,
                COL_ELLIPSE,
                2,
            )
            cv2.circle(overlay, local_center, 3, COL_TARGET, -1)

        status = "FOUND" if found else "SEARCH"
        caption = f"{status} | mask={mask_pixels}"
        if depth_thresh is not None:
            caption += f" thr={depth_thresh:.3f}"
        cv2.putText(
            overlay,
            caption,
            (10, max(20, overlay.shape[0] - 10)),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        tiles = []
        tiles.append((overlay, "ROI overlay"))
        if mask_img is not None:
            mask_rgb = cv2.applyColorMap(mask_img, cv2.COLORMAP_TURBO)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            tiles.append((mask_rgb, "Depth mask"))
        if depth_img is not None and np.any(depth_img):
            depth_rgb = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)
            depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
            tiles.append((depth_rgb, "Depth slice"))

        prepared_tiles = []
        tile_w, tile_h = 320, 240
        for img, label in tiles:
            if img is None:
                continue
            rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                resized,
                label,
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

        debug_canvas = np.hstack(prepared_tiles)
        try:
            self.find_debug_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_canvas, encoding="rgb8")
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
        KLT optical flow tracking for ellipse center with light RANSAC.
        Falls back to depth-based ellipse detection if needed.
        Returns: (found, cx, cy, ellipse_params)
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = frame.shape[:2]
        debug_rois = []
        if getattr(self, "last_ellipse_bbox", None) is not None:
            debug_rois.append(
                {"rect": self.last_ellipse_bbox, "color": (0, 255, 255), "label": "last ellipse"}
            )

        # ============================================================
        # 1) TRY OPTICAL FLOW (KLT)
        # ============================================================
        klt_ready = (
            getattr(self, "prev_gray", None) is not None and
            getattr(self, "klt_points", None) is not None and
            self.klt_points is not None and
            len(self.klt_points) > 0
        )

        if klt_ready:
            try:
                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray,
                    self.klt_points, None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
            except Exception:
                new_pts, status = None, None

            if (
                new_pts is not None and
                status is not None and
                len(new_pts) > 0 and
                len(status) > 0
            ):
                status = status.reshape(-1)
                tracked = new_pts[status == 1]

                # ----- SAFELY RESHAPE -----
                tracked = np.asarray(tracked).reshape(-1, 2)
                tracked = tracked[np.isfinite(tracked).all(axis=1)]

                # ============================================================
                # 1A) RANSAC FILTERING
                # ============================================================
                if tracked.shape[0] >= 5:
                    # Fit center with RANSAC: find the densest region
                    med = np.median(tracked, axis=0)  # initial center guess
                    distances = np.linalg.norm(tracked - med, axis=1)

                    # Threshold based on MAD (median abs deviation)
                    mad = np.median(np.abs(distances - np.median(distances)))
                    r_inlier = max(5.0, 2.5 * mad)  # radius threshold

                    inliers = tracked[distances < r_inlier]

                    if inliers.shape[0] >= 3:
                        tracked = inliers

                # If enough left after RANSAC, accept
                if tracked.shape[0] >= 3:
                    cx = int(np.median(tracked[:, 0]))
                    cy = int(np.median(tracked[:, 1]))

                    ax = max(6, int(np.std(tracked[:, 0]) * 1.5))
                    ay = max(6, int(np.std(tracked[:, 1]) * 1.5))

                    ellipse_params = ((cx, cy), (ax, ay), 0.0)

                    # Update bbox
                    self.last_ellipse_bbox = (
                        max(0, cx - ax), max(0, cy - ay),
                        min(w - 1, cx + ax), min(h - 1, cy + ay)
                    )

                    # Update KLT state
                    self.klt_points = tracked.reshape(-1, 1, 2)
                    self.prev_gray = gray
                    self._publish_track_debug(
                        frame,
                        points=tracked,
                        ellipse_params=ellipse_params,
                        rois=debug_rois,
                        mode="KLT",
                        extra_text=f"{tracked.shape[0]} pts",
                    )
                    return True, cx, cy, ellipse_params

            # If KLT failed, reset it to allow fallback
            self.klt_points = None
            self.prev_gray = gray

        # ============================================================
        # 2) FALLBACK SEARCH (TIGHT AROUND LAST ELLIPSE)
        # ============================================================
        if getattr(self, "last_ellipse_bbox", None) is not None:
            x1, y1, x2, y2 = self.last_ellipse_bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # Tight ROI
            bw = max(10, x2 - x1)
            bh = max(10, y2 - y1)
            roi_w = int(bw * 1.3)
            roi_h = int(bh * 1.3)

            rx1 = int(np.clip(cx - roi_w / 2, 0, w - 1))
            ry1 = int(np.clip(cy - roi_h / 2, 0, h - 1))
            rx2 = int(np.clip(cx + roi_w / 2, 0, w))
            ry2 = int(np.clip(cy + roi_h / 2, 0, h))
            tight_roi = (rx1, ry1, rx2, ry2)
            debug_rois.append(
                {"rect": tight_roi, "color": (255, 0, 255), "label": "tight ROI"}
            )

            found, cx, cy, eparams = self.find_ellipse(
                frame, (rx1, ry1, rx2, ry2), bottom_ratio=1.0
            )
            if found:
                self._init_klt_points(gray, rx1, ry1, rx2, ry2)
                self._publish_track_debug(
                    frame,
                    ellipse_params=eparams,
                    rois=debug_rois,
                    mode="tight ROI",
                )
                return True, cx, cy, eparams

        # ============================================================
        # 3) FALLBACK #2 — EXPANDED ROI
        # ============================================================
        if getattr(self, "last_ellipse_bbox", None) is not None:
            x1, y1, x2, y2 = self.last_ellipse_bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            bw = max(10, x2 - x1)
            bh = max(10, y2 - y1)
            roi_w = int(bw * 2.0)
            roi_h = int(bh * 2.0)

            rx1 = int(np.clip(cx - roi_w / 2, 0, w - 1))
            ry1 = int(np.clip(cy - roi_h / 2, 0, h - 1))
            rx2 = int(np.clip(cx + roi_w / 2, 0, w))
            ry2 = int(np.clip(cy + roi_h / 2, 0, h))
            expanded_roi = (rx1, ry1, rx2, ry2)
            debug_rois.append(
                {"rect": expanded_roi, "color": (0, 165, 255), "label": "expanded ROI"}
            )

            found, cx, cy, eparams = self.find_ellipse(
                frame, (rx1, ry1, rx2, ry2), bottom_ratio=1.0
            )
            if found:
                self._init_klt_points(gray, rx1, ry1, rx2, ry2)
                self._publish_track_debug(
                    frame,
                    ellipse_params=eparams,
                    rois=debug_rois,
                    mode="expanded ROI",
                )
                return True, cx, cy, eparams

        # ============================================================
        # 4) FINAL FALLBACK — WHOLE FRAME
        # ============================================================
        found, cx, cy, eparams = self.find_ellipse(frame, (0, 0, w, h), bottom_ratio=1.0)
        if found:
            self._init_klt_points(gray, 0, 0, w, h)
            debug_rois.append({"rect": (0, 0, w, h), "color": (255, 255, 0), "label": "full frame"})
            self._publish_track_debug(
                frame,
                ellipse_params=eparams,
                rois=debug_rois,
                mode="full frame",
            )
            return True, cx, cy, eparams

        # Nothing found
        self.klt_points = None
        self.prev_gray = gray
        self._publish_track_debug(frame, rois=debug_rois, mode="lost")
        return False, None, None, None

    def _publish_track_debug(self, frame, *, points=None, ellipse_params=None, rois=None, mode="", extra_text=""):
        if self.track_debug_pub is None:
            return

        canvas = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if rois:
            for roi in rois:
                rect = roi.get("rect")
                if rect is None:
                    continue
                color = tuple(int(c) for c in roi.get("color", (0, 255, 255)))
                label = roi.get("label", "")
                x1, y1, x2, y2 = map(int, rect)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                if label:
                    cv2.putText(
                        canvas,
                        label,
                        (x1, max(0, y1 - 6)),
                        font,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        if points is not None and len(points) > 0:
            for pt in points:
                px, py = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        if ellipse_params is not None:
            center, axes, angle = ellipse_params
            cv2.ellipse(
                canvas,
                (int(center[0]), int(center[1])),
                (max(1, int(axes[0])), max(1, int(axes[1]))),
                angle,
                0,
                360,
                COL_ELLIPSE,
                2,
            )
            cv2.circle(canvas, (int(center[0]), int(center[1])), 3, COL_TARGET, -1)

        if mode:
            cv2.putText(
                canvas,
                f"track: {mode}",
                (10, 30),
                font,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        if extra_text:
            cv2.putText(
                canvas,
                extra_text,
                (10, 60),
                font,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        try:
            self.track_debug_pub.publish(
                self.bridge.cv2_to_imgmsg(canvas, encoding="rgb8")
            )
        except Exception:
            pass

    def _init_klt_points(self, gray, x1, y1, x2, y2):
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            self.klt_points = None
            return

        # Detect up to 50 good features
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7
        )
        if pts is None:
            self.klt_points = None
            return

        # Shift from ROI coords to full-frame coords
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1

        self.klt_points = pts
        self.prev_gray = gray
    
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
