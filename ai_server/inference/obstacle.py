from ultralytics import YOLO
from inference.model_files import ensure_models
import threading

def load_obstacle_model():
    ckpt_path = ensure_models() / "obstacle.pt"
    model = YOLO(str(ckpt_path))
    model.to("cuda")
    return model

_lock = threading.Lock()

CLASS_NAME_MAP = {
    "toy_person": "people",
    "toy_car": "car",
    "cone": "trafficCone"
}


def run_obstacle(model, image):
    with _lock:
        result = model.track(image, persist=True, tracker="bytetrack.yaml", verbose=False)
    return post_process_obstacle(result)

def post_process_obstacle(result):
    grouped = {"people": [], "car": [], "trafficCone": []}

    for box in result[0].boxes:
        if box.id is None:
            continue  # 추적 ID가 아직 확정 안 된 박스는 스킵

        raw_name = result[0].names[int(box.cls[0])]
        class_name = CLASS_NAME_MAP.get(raw_name)

        if class_name is None:
            continue  # 매핑 안 된 클래스는 스킵

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        grouped[class_name].append({
            "sequence_id": int(box.id[0]),
            "confidence": float(box.conf[0]),
            "b_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
        })

    return grouped