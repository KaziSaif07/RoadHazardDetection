import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="AI Road Hazard Detection", layout="wide")

st.title("🚦 AI Road Hazard Detection")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Hazards to detect
hazards = ["car", "truck", "bus", "motorcycle", "dog", "cow"]

# Output folder
os.makedirs("output", exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(frame)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            name = model.names[cls]

            if name in hazards:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(frame,
                            name,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

                cv2.putText(frame,
                            f"ALERT: {name.upper()} DETECTED",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                filepath = os.path.join("output", filename)
                cv2.imwrite(filepath, frame)

                with open("detections.txt", "a") as f:
                    f.write(f"{datetime.now()} - {name}\n")

    st.image(frame, channels="BGR", caption="Detection Result")
