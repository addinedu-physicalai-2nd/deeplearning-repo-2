from ultralytics import YOLO

from inference.model_files import ensure_models
from risk_level_module import risk_level


def load_pothole_model():
    ckpt_path = ensure_models() / "pothole.pt"
    model = YOLO(str(ckpt_path))
    model.to("cuda")
    return model


def run_pothole(model, image):
    result = model.track(
        image,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    return post_process_pothole(result, image)


def post_process_pothole(result, image):
    boxes = [box for box in result[0].boxes if box.id is not None]
    potholes = [
        (*box.xyxy[0].tolist(), float(box.conf[0]))
        for box in boxes
    ]
    levels = risk_level(image, potholes)

    detections = []
    for box, level in zip(boxes, levels):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "sequence_id": int(box.id[0]),
                "confidence": float(box.conf[0]),
                "b_box": {
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                },
                "level": level,
            }
        )
    return {"pothole": detections}
