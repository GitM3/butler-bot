import time
import time as tm

import cv2
import requests
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

# --- Settings ---
MODEL_ID = "PekingU/rtdetr_v2_r18vd"
SCORE_THR = 0.5
CAM_INDEX = 0             
WINDOW_NAME = "RT-DETRv2 webcam"

print("Launching")
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID)

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
     outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

for result in results:
     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
         score, label = score.item(), label_id.item()
         box = [round(i, 2) for i in box.tolist()]
         print(f"{model.config.id2label[label]}: {score:.2f} {box}")

print("✅ Warmup complete")

# --- Open webcam ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
print("Opened Webcam")

# Optionally set a resolution (comment out if you prefer default)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_detections(frame, det, id2label):
    h, w = frame.shape[:2]
    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    for score, label_id, box in zip(scores, labels, boxes):
        if score < SCORE_THR:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # clamp to frame bounds
        x_min = max(0, min(x_min, w - 1))
        x_max = max(0, min(x_max, w - 1))
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h - 1))

        cls_name = id2label[int(label_id)]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f"{cls_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x_min, y_min - th - baseline), (x_min + tw, y_min), (0, 255, 0), -1)
        cv2.putText(frame, text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

try:
    prev_t = time.time()
    print("Starting")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Failed to grab frame.")
            break

        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Preprocess
        inputs = image_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            print("Running model")
            outputs = model(**inputs)
        #
        # # Post-process to original size
        h, w = frame_bgr.shape[:2]
        target_sizes = torch.tensor([(h, w)])
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=SCORE_THR)

        # Draw
        draw_detections(frame_bgr, results[0], model.config.id2label)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q to quit
            break
        tm.sleep(1)
finally:
    print("Ending")
    cap.release()
    cv2.destroyAllWindows()
