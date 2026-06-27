from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Hazards to detect
hazards = ["car", "truck", "bus", "motorcycle", "dog", "cow"]

# Create output folder
os.makedirs("output", exist_ok=True)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            name = model.names[cls]

            if name in hazards:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0), 2)

                cv2.putText(frame,
                            f"ALERT: {name.upper()} DETECTED",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),3)

                # Save screenshot
                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                filepath = os.path.join("output", filename)
                cv2.imwrite(filepath, frame)

                # Save log
                with open("detections.txt","a") as f:
                    f.write(f"{datetime.now()} - {name}\n")

    cv2.imshow("AI Road Hazard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cap = cv2.VideoCapture(0)
