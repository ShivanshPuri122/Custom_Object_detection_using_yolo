from ultralytics import YOLO

def main():
    trained_model_path = r"C:\Projects\ObjectDetection\Custom_trained\trained_model\train_results\weights\best.pt"
    data_yaml_path = r"C:\Projects\ObjectDetection\Custom_trained\Dataset\data.yaml"

    model = YOLO(trained_model_path)
    results = model.val(data=data_yaml_path)

    print("Validation Results:")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")

if __name__ == "__main__":
    main()
