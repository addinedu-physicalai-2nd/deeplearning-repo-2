from ultralytics import YOLO

def load_obstacle_model():
    model = YOLO("models/obstacle.pt")
    model.to("cuda")
    return model

def run_obstacle(model, image):
    result = model.predict(image)
    return post_process_obstacle(result)


#작성 필요
def post_process_detections(result):
    grouped = {"people": [], "car": [], "trafficCone": []}
    counters = {"people": 0, "car": 0, "trafficCone": 0}

    for box in result[0].boxes:
        class_name = result[0].names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        grouped[class_name].append({
            "sequence_id": counters[class_name],
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
        })
        counters[class_name] += 1

    return grouped