from ultralytics import YOLO
import cv2
import numpy as np

model_path = r"C:\Projects\ObjectDetection\Custom_trained\trained_model\train_results\weights\best.pt"
model = YOLO(model_path)

img_path = r"C:\Projects\ObjectDetection\Custom_trained\Images\image2.webp"
img = cv2.imread(img_path)

img_resized = cv2.resize(img, (640, 640))

results = model.predict(source=img_resized, conf=0.5, iou=0.4, save=False, show=False)

np.random.seed(42)
class_colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

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

img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

cv2.imshow("YOLOv8 Detection", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
