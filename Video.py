from ultralytics import YOLO
import cv2
import numpy as np

model_path = r"C:\Projects\ObjectDetection\Custom_trained\trained_model\train_results\weights\best.pt"
model = YOLO(model_path)

video_path = r"C:\Projects\ObjectDetection\Custom_trained\Video\vid3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

np.random.seed(42)
class_colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, (640, 640))

    results = model.predict(source=img_resized, conf=0.5, iou=0.4, save=False, show=False)

    for result in results:
        boxes = result.boxes  

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0]  
            cls = int(box.cls[0])

            class_label = model.names[cls]
            color = [int(c) for c in class_colors[cls]]

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)

            label = f"{class_label}: {conf:.2f}"
            cv2.putText(
                img_resized,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    cv2.imshow("YOLOv8 Video Detection", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
