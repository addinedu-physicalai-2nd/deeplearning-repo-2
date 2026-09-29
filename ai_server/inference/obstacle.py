from ultralytics import YOLO
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def load_obstacle_model():
    drive_dir = Path((MODELS_DIR / "model_drive_link.txt").read_text().strip())
    ckpt_path = drive_dir / "obstacle.pt"
    model = YOLO(str(ckpt_path))
    model.to("cuda")
    return model

def run_obstacle(model, image):
    result = model.predict(image)
    return post_process_obstacle(result)


CLASS_NAME_MAP = {
    "toy_person": "people",
    "toy_car": "car",
    "cone":"trafficCone"
}

def post_process_obstacle(result):
    grouped = {"people": [], "car": [], "trafficCone": []}
    counters = {"people": 0, "car": 0, "trafficCone": 0}

    for box in result[0].boxes:
        raw_name = result[0].names[int(box.cls[0])]
        class_name = CLASS_NAME_MAP.get(raw_name)

        if class_name is None:
            continue  # 매핑 안 된 클래스는 일단 스킵

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        grouped[class_name].append({
            "sequence_id": counters[class_name],
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
        })
        counters[class_name] += 1

    return grouped

#model=load_obstacle_model()
#print(model.names)