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
    torch._C._reset_parallel_info()
    print("✅ Done")

