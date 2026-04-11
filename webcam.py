import cv2
from ultralytics import YOLO

# ── Load trained model ───────────────────────────────────
WEIGHTS = "runs/detect/train6/weights/best.pt"

class_mapping     = {"apple": 0, "banana": 1, "orange": 2}
inv_class_mapping = {v: k for k, v in class_mapping.items()}

# Class colors (BGR format)
COLORS = {
    "apple"  : (0,   255, 0),    # green
    "banana" : (0,   255, 255),  # yellow
    "orange" : (0,   165, 255),  # orange
}

def run_webcam(camera_index=1):
    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_index}")
        print("Try changing camera_index to 0 or 2")
        return

    print(f"Camera opened at index {camera_index}")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Run detection
        results = model(frame, conf=0.25, verbose=False)

        # Draw detections
        for result in results:
            for detection in result.boxes:
                x_min, y_min, x_max, y_max = [int(v) for v in detection.xyxy[0]]
                confidence = float(detection.conf[0])
                class_id   = int(detection.cls[0])
                class_name = inv_class_mapping.get(class_id, "unknown")
                color      = COLORS.get(class_name, (255, 255, 255))
                label      = f"{class_name}: {confidence:.2f}"

                # 🔥 ADDED: Center coordinates
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2

                # 🔥 ADDED: Print coordinates
                print(f"{class_name} at ({cx}, {cy}) confidence={confidence:.2f}")

                # Bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # 🔥 ADDED: Draw center point
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                cv2.rectangle(frame,
                              (x_min, y_min - th - 10),
                              (x_min + tw, y_min),
                              color, -1)

                # Label text
                cv2.putText(frame, label,
                            (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Show FPS on screen
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Fruit Detection - External Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_webcam(camera_index=1)