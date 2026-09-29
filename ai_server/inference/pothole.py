from ultralytics import YOLO

def load_pothole_model():
    model = YOLO("models/pothole.pt")
    model.to("cuda")
    return model

def run_pothole(model, image):
    result = model.predict(image)
    return post_process_pothole(result,image)


#작성 필요
from risk_level_module import risk_level

def post_process_pothole(result, image):
    potholes = [
        (*box.xyxy[0].tolist(), float(box.conf[0]))
        for box in result[0].boxes
    ]
    levels = risk_level(image, potholes)

    detections = []
    for i, (box, level) in enumerate(zip(result[0].boxes, levels)):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "sequence_id": i,
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            "level": level
        })
    return {"pothole": detections}