import gc
import os
import sys
import time

import cv2
import requests
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

# --- Settings ---
MODEL_ID = "PekingU/rtdetr_v2_r18vd"
SCORE_THR = 0.5
CAM_INDEX = 0
WINDOW_NAME = "RT-DETRv2 Webcam"
DELAY = 0.2
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# --- Prevent thread deadlocks on CPU ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

print("Launching...")

# --- Load image and model ---
image = Image.open(requests.get(url, stream=True).raw)
image_processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID)

model.eval()  # disable dropout etc.
device = torch.device("cpu")
model.to(device)

# --- Warmup inference ---
print("Running warmup inference...")
inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([(image.height, image.width)]),
    threshold=SCORE_THR,
)

# --- Display results ---
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        if score < SCORE_THR:
            continue
        label = model.config.id2label[label_id.item()]
        box = [round(i, 2) for i in box.tolist()]
        print(f"{label}: {score:.2f} {box}")

print("✅ Warmup complete")

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
print("Opened Webcam")

# Optional: set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_detections(frame, det, id2label):
    h, w = frame.shape[:2]
    for score, label_id, box in zip(det["scores"], det["labels"], det["boxes"]):
        if score < SCORE_THR:
            continue
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        x_min = max(0, min(x_min, w - 1))
        x_max = max(0, min(x_max, w - 1))
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h - 1))

        cls_name = id2label[int(label_id)]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f"{cls_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x_min, y_min - th - baseline), (x_min + tw, y_min), (0, 255, 0), -1)
        cv2.putText(frame, text, (x_min, y_min - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# --- Main loop ---
try:
    prev_t = time.time()
    print("Starting inference loop. Press 'q' or ESC to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Failed to grab frame.")
            break

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Preprocess and move to device
        inputs = image_processor(images=pil_img, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Postprocess
        h, w = frame_bgr.shape[:2]
        target_sizes = torch.tensor([(h, w)], device=device)
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=SCORE_THR
        )

        # Draw detections
        draw_detections(frame_bgr, results[0], model.config.id2label)

        # FPS overlay
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q to quit
            break

        time.sleep(DELAY)

finally:
    print("Ending")
    cap.release()
    cv2.destroyAllWindows()
    del model, image_processor, inputs, outputs, results
    gc.collect()
    sys.exit(0)
