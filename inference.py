import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
import os

class_mapping     = {"apple": 0, "banana": 1, "orange": 2}
inv_class_mapping = {v: k for k, v in class_mapping.items()}

# ── Load the best trained weights ───────────────────────
WEIGHTS = "runs/detect/train6/weights/best.pt"

def detect_fruits(image_path, weights=WEIGHTS, conf_threshold=0.25):
    if not os.path.exists(weights):
        print(f"Weights not found at {weights}. Train the model first.")
        return
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    model   = YOLO(weights)
    results = model(image_path, conf=conf_threshold)
    image   = cv2.imread(image_path)

    for result in results:
        for detection in result.boxes:
            x_min, y_min, x_max, y_max = [int(v) for v in detection.xyxy[0]]
            confidence = float(detection.conf[0])
            class_id   = int(detection.cls[0])
            class_name = inv_class_mapping.get(class_id, "unknown")
            label      = f"{class_name}: {confidence:.2f}"

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            cv2.rectangle(image, (x_min, y_min - th - 10), (x_min + tw, y_min), (0, 255, 0), -1)
            cv2.putText(image, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Detection: {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.show()

# ── Entry point ─────────────────────────────────────────
if __name__ == "__main__":
    # Pass image path as argument, or hardcode one for quick testing
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test_data/mixed_22.jpg"   # ← change to any test image

    detect_fruits(img_path)