import gc
import time
import warnings

import cv2
import rclpy
import supervision as sv
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from sensor_msgs.msg import Image


class RFDetrNode(Node):
    def __init__(self):
        super().__init__("rfdetr_node")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/camera/image_raw",
                                            self.callback, 10)
        self.image_pub = self.create_publisher(Image, "/camera/annotated", 10)
        self.target_pub = self.create_publisher(Point, "/target/center", 10)

        self.score_thr = 0.5
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

        self.get_logger().info("Loading RF-DETR model...")
        self.model = RFDETRMedium()
        torch.set_num_threads(4)
        self.model.optimize_for_inference()
        #self.tracker = SORTTracker()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.prev_t = time.time()
        self.get_logger().info("✅ RF-DETR ready")
        filter_labels = ["cup","bottle","wine glass"]
        self.target_labels = [
                i for i, n in COCO_CLASSES.items() if n in filter_labels
        ]
        self.get_logger().info(f"Target labels: {self.target_labels} ({filter_labels})")

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8") #TODO: No need to annotate other objects, but probably good for debugging for now.
        # frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (320, 240))
        detections = self.model.predict(frame, threshold=self.score_thr)

        id_labels = []
        publish_points = []
        for i in range(len(detections.xyxy)):
            cid = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            label = COCO_CLASSES[cid]
            id_labels.append(f"{label} {conf:.2f}")
            if cid in self.target_labels:
                x1, y1, x2, y2 = detections.xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                publish_points.append((cx, cy, conf, label))

        annotated = frame.copy()
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=id_labels)

        now = time.time()
        fps = 1.0 / (now - self.prev_t)
        self.prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        out_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        self.image_pub.publish(out_msg)
        found = len(publish_points)
        if publish_points:
            cx = sum(p[0] for p in publish_points) / found
            cy = sum(p[1] for p in publish_points) /  found
            conf = sum(p[2] for p in publish_points) /  found
            msg = Point(x=float(cx), y=float(cy), z=float(conf))
            self.target_pub.publish(msg)
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
