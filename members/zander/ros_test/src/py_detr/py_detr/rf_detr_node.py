import gc
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image

# Check for updates: https://github.com/roboflow/rf-detr/blob/main/rfdetr/util/coco_classes.py
COCO_CLASSES = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
}

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
            self.ort_session = ort.InferenceSession(onnx_model_path)

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
            # Resize in RGB space
            frame_rgb = cv2.resize(frame_rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0,1], then (x - mean) / std in RGB order
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

        # RF-DETR exports typically use sigmoid (multi-label style). Keep as requested.
        probs  = sigmoid(logits)                      # (Q,C)
        scores = np.max(probs, axis=1)               # (Q,)
        labels = np.argmax(probs, axis=1)            # (Q,)

        # Sort & clip to max
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

        # Filter by allowed_cids if provided
        if allowed_cids is not None and len(scores) > 0:
            mask_keep = np.array([int(c) in allowed_cids for c in labels], dtype=bool)
            scores = scores[mask_keep]
            labels = labels[mask_keep]
            boxes  = boxes[mask_keep]
            if masks is not None:
                masks = masks[mask_keep]

        # Convert boxes to absolute xyxy
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
        # Parameters
        self.declare_parameter("model_path", "base.onnx")
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/camera/camera/color/image_raw",
                                            self.callback, 10)
        self.image_pub = self.create_publisher(Image, "/camera/annotated", 10)
        self.target_pub = self.create_publisher(Point, "/target/center", 10)

        filter_labels = ["cup","bottle","wine glass"]
        self.target_labels = [
                i for i, n in COCO_CLASSES.items() if n in filter_labels
        ]
        self.get_logger().info(f"Target labels: {self.target_labels} ({filter_labels})")
        # Model prep
        self.model = RFDETR_ONNX(self.model_path)
        self.get_logger().info("✅ RF-DETR ready")
        self.prev_t = time.time()


    def callback(self, msg: Image):
        # If your camera publishes RGBA, use "rgba8" and we'll drop alpha.
        # For minimal conversions, set your camera or upstream node to publish rgb8.
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")  # HxWx3, uint8, RGB

        # ONNX inference (returns filtered to target_labels already)
        scores, labels, boxes, _ = self.model.predict(
            frame,
            allowed_cids=self.target_labels,
        )

        # Build publish_points and draw (minimal: cv2 only)
        publish_points = []
        annotated = frame.copy()  # still RGB

        for i in range(len(scores)):
            cid   = int(labels[i])
            conf  = float(scores[i])
            x1, y1, x2, y2 = boxes[i].astype(np.int32)
            # center
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            publish_points.append((cx, cy, conf, COCO_CLASSES[cid]))

            # simple annotation
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            txt = f"{COCO_CLASSES[cid]} {conf:.2f}"
            cv2.putText(annotated, txt, (x1 + 4, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        now = time.time()
        fps = 1.0 / max(1e-6, (now - self.prev_t))
        self.prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Publish annotated image (convert RGB -> BGR only if you need "bgr8")
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="rgb8")
        self.image_pub.publish(out_msg)

        # Publish target center as average across filtered detections (same as your previous logic)
        if publish_points:
            n = len(publish_points)
            cx = sum(p[0] for p in publish_points) / n
            cy = sum(p[1] for p in publish_points) / n
            conf = sum(p[2] for p in publish_points) / n
            self.target_pub.publish(Point(x=float(cx), y=float(cy), z=float(conf)))
            self.get_logger().info(f"Found:{cx},{cy}")

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
