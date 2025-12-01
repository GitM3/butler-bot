import gc
import math
import time
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
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
        self.image_pub = self.create_publisher(Image, "/camera/annotated", 10)
        self.target_pub = self.create_publisher(Float64MultiArray, "/target/center", 10)
        self.find_mask_pub = self.create_publisher(Image, "/ellipse/find_mask", 1)
        self.find_debug_pub   = self.create_publisher(Image, "/ellipse/find_debug", 1)
        self.find_debug_edges_pub   = self.create_publisher(Image, "/ellipse/find_debug_edges", 1)
        self.find_debug_bdetect_pub = self.create_publisher(Image, "/ellipse/find_debug_bdetect", 1)
        self.find_debug_masked_pub  = self.create_publisher(Image, "/ellipse/find_debug_masked", 1)

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
        
        self.prev_t = time.time()
        self.prev_ellipse = None
        self.prev_gray = None
        self.good_detect_kf = 0
        self.good_ellipse_kf = 0
        self.detect_yes = 0
        self.detect_no = 0
        self.DETECT_YES_THRESH = 3   # 3 stable frames = object confirmed
        self.DETECT_NO_THRESH  = 5
        self.ellipse_yes = 0
        self.ellipse_no = 0
        self.ELLIPSE_YES_THRESH = 3
        self.ELLIPSE_NO_THRESH  = 5
        # Ellipse search tuning
        self.declare_parameter("ellipse_bottom_ratio", 0.4)
        self.declare_parameter("ellipse_depth_margin", 0.04)
        self.declare_parameter("ellipse_canny_low", 25.0)
        self.declare_parameter("ellipse_canny_high", 100.0)
        self.declare_parameter("ellipse_min_contour_area", 150.0)
        self.declare_parameter("ellipse_min_mask_pixels", 80.0)
        self.declare_parameter("mask_kernel_size", 50)
        self.declare_parameter("depth_threshold", 1500.0)
        self.declare_parameter("bag_path", "")
        self.declare_parameter("frame_rate", 20.0)

        self.ellipse_bottom_ratio = float(self.get_parameter("ellipse_bottom_ratio").get_parameter_value().double_value)
        self.ellipse_depth_margin = float(self.get_parameter("ellipse_depth_margin").get_parameter_value().double_value)
        self.ellipse_canny_low = float(self.get_parameter("ellipse_canny_low").get_parameter_value().double_value)
        self.ellipse_canny_high = float(self.get_parameter("ellipse_canny_high").get_parameter_value().double_value)
        self.ellipse_min_contour_area = float(self.get_parameter("ellipse_min_contour_area").get_parameter_value().double_value)
        self.ellipse_min_mask_pixels = float(self.get_parameter("ellipse_min_mask_pixels").get_parameter_value().double_value)
        self.depth_threshold = float(self.get_parameter("depth_threshold").get_parameter_value().double_value)
        self.mask_kernel_size = int(self.get_parameter("mask_kernel_size").get_parameter_value().integer_value)
        self.bag_path = self.get_parameter("bag_path").get_parameter_value().string_value
        self.frame_rate = float(self.get_parameter("frame_rate").get_parameter_value().double_value)
        self.target_ellipse_keyframes = 0 # Keyframes of good target ellipse pairs.
        self.timer = self.create_timer(1/self.frame_rate, self.callback)  

        self.kf = cv2.KalmanFilter(5*2,5) # cx,cy,ma,mi,an
        dt = 1.0/self.frame_rate
        mat_i  = np.eye(5, dtype=np.float32)
        mat_dt = mat_i * dt

        A = np.zeros((10, 10), dtype=np.float32)
        A[0:5, 0:5] = mat_i
        A[0:5, 5:10] = mat_dt
        A[5:10, 5:10] = mat_i
        self.kf.transitionMatrix = A

        H = np.zeros((5,10), dtype=np.float32)
        H[0:5,0:5] = mat_i
        self.kf.measurementMatrix = H
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(5, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(10, dtype=np.float32) * 1

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
        self.get_logger().info("✅ BB Detector ready")

    def kf_reset(self):
        self.kf.errorCovPost = np.eye(10, dtype=np.float32) * 1

    def kf_predict(self):
        pred = self.kf.predict()
        px, py          = float(pred[0]), float(pred[1])
        p_major         = float(pred[2])
        p_minor         = float(pred[3])
        p_angle_deg     = float(pred[4])

        return ((px, py), (p_major, p_minor), p_angle_deg)
    def kf_step(self,ellipse):
        tracked_ellipse = self.kf_predict()
        if ellipse is not None:
            (cx, cy), (major, minor), angle_deg = ellipse
            meas = np.array(
                [[cx], [cy], [major], [minor], [angle_deg]],
                dtype=np.float32
            )
            self.kf.correct(meas)
        return tracked_ellipse

    def callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            self.get_logger().warn("No frames received!")
            return

        depth_image = np.asanyarray(depth_frame.get_data())      # uint16 depth in mm
        frame = np.asanyarray(color_frame.get_data())            # RGB aligned to depth

        annotated = frame.copy()

        detections, boxes, scores, labels = self.detect_objects(frame, depth_image)
        target_detection = next((d for d in detections if d["class_id"] == self.target_class_id), None)

        object_detected_raw = target_detection is not None

        if object_detected_raw:
            self.detect_yes += 1
            self.detect_no = 0
        else:
            self.detect_no += 1
            self.detect_yes = 0

        object_detected_stable = (self.detect_yes >= self.DETECT_YES_THRESH)
        object_lost_stable     = (self.detect_no  >= self.DETECT_NO_THRESH)

        det_bbox = det_center = det_depth = None
        if target_detection:
            det_center = target_detection["center"]
            det_depth  = target_detection["depth"]
            det_bbox   = target_detection["bbox"]

            cx, cy = det_center
            cv2.circle(annotated, (int(cx),int(cy)), 4, COL_TARGET, -1)

        for box in boxes:
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), COL_WHITE, 2)


        # Ellipse stuff
        init_depth_mask = ((depth_image > 0) & (depth_image <= self.depth_threshold)).astype(np.uint8)*255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.mask_kernel_size,self.mask_kernel_size))
        expanded_mask = cv2.dilate(init_depth_mask, kernel, iterations=1)

        ellipse = self.find_ellipse(expanded_mask, frame, det_bbox)
        ellipse_found_raw = ellipse is not None

        # Debounce ellipse
        if ellipse_found_raw:
            self.ellipse_yes += 1
            self.ellipse_no = 0
        else:
            self.ellipse_no += 1
            self.ellipse_yes = 0

        ellipse_found_stable = (self.ellipse_yes >= self.ELLIPSE_YES_THRESH)
        ellipse_lost_stable  = (self.ellipse_no  >= self.ELLIPSE_NO_THRESH)


        # Debug publish find mask + overlay
        find_mask_rgb = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2RGB)
        if ellipse is not None:
            cv2.ellipse(find_mask_rgb, ellipse, (0,255,0), 2)
        self.find_mask_pub.publish(self.bridge.cv2_to_imgmsg(find_mask_rgb, "rgb8"))


        # =====================================================
        # STATE: DETECT
        # =====================================================
        if self.state == State.DETECT:

            if object_detected_stable and (det_depth is not None) and det_depth <= self.depth_threshold:
                self.state = State.ELLIPSE_SEARCH
                self.get_logger().info("STATE → ELLIPSE_SEARCH")
                self.kf_reset()
                self.prev_ellipse = None
                self.ellipse_yes = self.ellipse_no = 0

        # =====================================================
        # STATE: ELLIPSE_SEARCH
        # =====================================================
        elif self.state == State.ELLIPSE_SEARCH:

            # Draw current ellipse (not stable yet)
            if ellipse is not None:
                annotated = self.draw_ellipse(annotated, ellipse, color=(255,0,0))

            # Losing object BEFORE ellipse is confirmed → go back
            if object_lost_stable and not ellipse_found_stable:
                self.get_logger().info("STATE → DETECT (lost before confirming ellipse)")
                self.state = State.DETECT

            # Ellipse confirmed AND object lost → go to TRACK
            elif ellipse_found_stable and object_lost_stable:
                self.prev_ellipse = ellipse
                self.get_logger().info("STATE → ELLIPSE_TRACK (ellipse confirmed)")
                self.state = State.ELLIPSE_TRACK

            # If ellipse is found, correct KF for stabilizing angles etc.
            if ellipse_found_raw:
                self.prev_ellipse = ellipse
                tracked_ellipse = self.kf_step(ellipse)
                annotated = self.draw_ellipse(annotated, tracked_ellipse, color=(0,0,255))

        # =====================================================
        # STATE: ELLIPSE_TRACK
        # =====================================================
        elif self.state == State.ELLIPSE_TRACK:

            # If object resurfaces, abandon ellipse tracking
            if object_detected_stable:
                self.get_logger().info("STATE → DETECT (object back in view)")
                self.state = State.DETECT

            # Step 1: Try direct ellipse detection
            tracked_ellipse = self.kf_predict()
            if ellipse is None:
                # Try fallback: grow region around prev + predicted ellipse
                ellipse = self.fallback_ellipse_search(frame, tracked_ellipse)

            # If ellipse still not found → tracking failed
            if ellipse is None:
                if ellipse_lost_stable:
                    self.get_logger().info("STATE → DETECT (ellipse lost)")
                    self.state = State.DETECT
            else:
                # Update and draw
                self.prev_ellipse = ellipse
                tracked_ellipse = self.kf_step(ellipse)
                annotated = self.draw_ellipse(annotated, ellipse, color=(255,0,0))
                annotated = self.draw_ellipse(annotated, tracked_ellipse, color=(0,0,255))

            track_dbg = self.draw_ellipse(frame, tracked_ellipse, (0,255,255))
            self.track_debug_pub.publish(self.bridge.cv2_to_imgmsg(track_dbg,"rgb8"))

        now = time.time()
        fps = 1.0/(now - self.prev_t)
        self.prev_t = now

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)
        cv2.putText(annotated, "S: "+STATES[self.state.value], (120,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "rgb8"))
        self.publish_detections(detections)

    def draw_ellipse(self,img, ellipse, color=(0,255,0), thickness=2):
        if ellipse is None:
            return img
        output = img.copy()
        cv2.ellipse(output, ellipse, color, thickness)
        return output

    def fallback_ellipse_search(self, frame, predicted_ellipse):
        if self.prev_ellipse is None:
            return None

        h, w, _ = frame.shape

        for scale in [1, 2, 3, 4]:
            k = int(self.mask_kernel_size * scale)
            mask = np.zeros((h, w), dtype=np.uint8)

            cv2.ellipse(mask, self.prev_ellipse, 255, 2)
            cv2.ellipse(mask, predicted_ellipse, 255, 2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

            ellipse = self.find_ellipse(mask, frame)
            if ellipse is not None:
                return ellipse

        return None

    def find_ellipse(self, mask, rgb, det_bbox=None):
        """
        Finds best ellipse; also publishes debug images so we can diagnose failure.
        """

        # ------------------------------
        # 1. Build masked image
        # ------------------------------
        masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)

        # ------------------------------
        # 2. Canny edges
        # ------------------------------
        gray = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, self.ellipse_canny_low, self.ellipse_canny_high)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)

        # Publish Canny edges debug
        edges_rgb = cv2.cvtColor(edges_masked, cv2.COLOR_GRAY2RGB)
        self.find_debug_edges_pub.publish(self.bridge.cv2_to_imgmsg(edges_rgb, "rgb8"))

        # ------------------------------
        # 3. Optional bounding-box gate
        # ------------------------------
        if det_bbox is not None:
            x1, y1, x2, y2 = det_bbox.astype(float)
            w = x2 - x1
            h = y2 - y1

            margin_x = 0.10 * w
            margin_y = 0.10 * h

            bx1 = int(max(0, x1 - margin_x))
            by1 = int(max(0, y1 - margin_y))
            bx2 = int(min(rgb.shape[1]-1, x2 + margin_x))
            by2 = int(min(rgb.shape[0]-1, y2 + margin_y))

            # Create bounding-gate mask
            bdetect = np.zeros_like(edges_masked, dtype=np.uint8)
            cv2.rectangle(bdetect, (bx1, by1), (bx2, by2), 255, -1)
        else:
            # Allow entire area if no detection
            bdetect = np.ones_like(edges_masked, dtype=np.uint8) * 255

        # Publish bounding box gating mask
        bdetect_rgb = cv2.cvtColor(bdetect, cv2.COLOR_GRAY2RGB)
        self.find_debug_bdetect_pub.publish(self.bridge.cv2_to_imgmsg(bdetect_rgb, "rgb8"))

        # ------------------------------
        # 4. Combine masks for debugging
        # ------------------------------
        combined_debug = cv2.bitwise_and(edges_rgb, edges_rgb, mask=bdetect)
        self.find_debug_masked_pub.publish(self.bridge.cv2_to_imgmsg(combined_debug, "rgb8"))

        # ------------------------------
        # 5. Contour detection
        # ------------------------------
        contours, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ellipse = None
        best_area = 0

        dbg = rgb.copy()  # draw contours for debugging

        for c in contours:
            # Draw contour in debug view
            cv2.drawContours(dbg, [c], -1, (0, 0, 255), 1)

            if len(c) < 5:
                continue

            area = cv2.contourArea(c)
            if area < 200:
                continue

            ellipse = cv2.fitEllipse(c)
            (cx, cy), (A, B), ang = ellipse

            # Enforce bounding box gating
            if det_bbox is not None:
                if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                    cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                    continue

            # Previous ellipse stability check
            if self.prev_ellipse is not None:
                (px, py), (pA, pB), pAng = self.prev_ellipse
                d_prev = np.hypot(px - cx, py - cy)

                if d_prev > 50:
                    continue
                if abs(A - pA) > 0.4 * pA:
                    continue
                if abs(B - pB) > 0.4 * pB:
                    continue
                if abs(ang - pAng) > 30:
                    continue

            # If we reached here → accept candidate
            if area > best_area:
                best_area = area
                best_ellipse = ellipse
                # Draw accepted ellipse in green
                cv2.ellipse(dbg, ellipse, (0, 255, 0), 2)

        # Publish debugging overlay
        self.find_debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "rgb8"))

        return best_ellipse

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
