import gc
import threading
import time
import warnings

import cv2
import supervision as sv
import torch
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from trackers import SORTTracker


def heartbeat():
    while True:
        print("[heartbeat]", time.strftime("%H:%M:%S"))
        time.sleep(5)

threading.Thread(target=heartbeat, daemon=True).start()
# --- Settings ---
SCORE_THR = 0.5
CAM_INDEX = 0
WINDOW_NAME = "RF-DETR + SORT Webcam"

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

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
torch.set_num_threads(1)
model.optimize_for_inference()
torch.set_num_threads(4)
tracker = SORTTracker()
labler = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT)
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

        frame = cv2.resize(frame, (640, 480))
        detections = model.predict(frame, threshold=SCORE_THR)
        detections = tracker.update(detections)
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        # Tracking
        id_labels = []
        for i in range(len(detections.xyxy)):
            tid = None
            if getattr(detections, "tracker_id", None) is not None:
                try:
                    tval = detections.tracker_id[i]
                    # some libs use -1 for "untracked"
                    if tval is not None and int(tval) != -1:
                        tid = int(tval)
                except Exception:
                    pass

            if tid is not None:
                id_labels.append(f"ID {tid}")
            else:
                cid = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                id_labels.append(f"{COCO_CLASSES[cid]} {conf:.2f}")

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=id_labels)

        for i, class_id in enumerate(detections.class_id):
            if COCO_CLASSES[int(class_id)] in ["bottle", "cup"]:
                xyxy = detections.xyxy[i]
                center, found = find_bottom_ellipse(frame, xyxy)
                if not found or center is None:
                    continue

                # is this detection tracked?
                tracked = False
                if getattr(detections, "tracker_id", None) is not None:
                    try:
                        tval = detections.tracker_id[i]
                        tracked = (tval is not None) and (int(tval) != -1)
                    except Exception:
                        tracked = False

                color = (0, 255, 0) if tracked else (0, 0, 255)
                cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 5, color, -1)
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

