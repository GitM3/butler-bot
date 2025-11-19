import gc
import time

import cv2
import supervision as sv
import torch
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

# --- Settings ---
SCORE_THR = 0.5
CAM_INDEX = 0
WINDOW_NAME = "RF-DETR Webcam"

# Optional: prevent MKL/OMP thread deadlocks on CPU
torch.set_num_threads(1)

def find_bottom_ellipse(frame, xyxy):
    """
    frame: full BGR image
    xyxy: [x1, y1, x2, y2] box coordinates (ints)
    returns: (center, found)
    """
    x1, y1, x2, y2 = map(int, xyxy)
    w, h = x2 - x1, y2 - y1

    # focus on bottom ~30% of ROI
    roi_y1 = y1 + int(h * 0.7)
    roi = frame[roi_y1:y2, x1:x2]
    if roi.size == 0:
        return ((x1 + w // 2, y2 - 5), False)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse, best_area = None, 0
    for c in contours:
        if len(c) >= 5:
            area = cv2.contourArea(c)
            if area > 50:  # ignore noise
                ellipse = cv2.fitEllipse(c)
                if area > best_area:
                    best_area, best_ellipse = area, ellipse

    if best_ellipse is not None:
        (cx, cy), (MA, ma), angle = best_ellipse
        # offset to full image coordinates
        return (int(x1 + cx), int(roi_y1 + cy)), True

    # fallback: bottom center of ROI
    return (x1 + w // 2, y2 - 5), False

# --- Initialize model ---
print("Loading RF-DETR model...")
model = RFDETRMedium()
model.optimize_for_inference()
print("✅ Model ready")

# --- Open webcam ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
print("Opened Webcam")
try:
    prev_t = time.time()
    print("Starting detection loop. Press 'q' or ESC to quit.")
    while True:
        ok, frame= cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        detections = model.predict(frame, threshold=SCORE_THR)
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
        # Ellipse
        for i, class_id in enumerate(detections.class_id):
            if COCO_CLASSES[class_id] in ["bottle","cup"]:
                xyxy = detections.xyxy[i]
                center, found = find_bottom_ellipse(frame, xyxy)
                color = (0, 255, 0) if found else (0, 0, 255)
                cv2.circle(annotated_frame, center, 5, color, -1)
        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    del model
    gc.collect()
    print("✅ Done")

