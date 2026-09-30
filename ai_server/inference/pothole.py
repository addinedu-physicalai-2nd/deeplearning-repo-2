from pathlib import Path
from ultralytics import YOLO
from inference.model_files import ensure_models
from risk_level_module import risk_level


def load_pothole_model():
    ckpt_path = ensure_models() / "pothole.pt"
    model = YOLO(str(ckpt_path))
    model.to("cuda")
    return model

def post_process_pothole(result, image):
    boxes = [b for b in result[0].boxes if b.id is not None]
    potholes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in boxes]
    areas = risk_level(image, potholes)

    detections = []
    for box, area in zip(boxes, areas):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "sequence_id": int(box.id[0]),
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            "area": area
        })

    return {"pothole": detections}