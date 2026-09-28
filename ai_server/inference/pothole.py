from ultralytics import YOLO

def load_pothole_model():
    model = YOLO("models/pothole.pt")
    model.to("cuda")
    return model

def run_pothole(model, image):
    result = model.predict(image)
    return post_process_pothole(result)


#작성 필요
def post_process_pothole(result):
    detections = []
    for i, box in enumerate(result[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "sequence_id": i,
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
        })
    return {"pothole": detections}