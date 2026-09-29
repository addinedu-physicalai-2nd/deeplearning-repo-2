from pathlib import Path
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def _resolve_ckpt_path(txt_name):
    drive_dir = Path((MODELS_DIR / txt_name).read_text().strip())
    candidates = list(drive_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"{drive_dir} 안에 pth 파일이 없음")
    return candidates[0]

def load_lane_model():
    m = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
    ckpt_path = _resolve_ckpt_path("model_drive_link.txt")   # 실제 txt 파일명으로 바꿔줘
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()

def run_lane(model, image):
    mask, conf_map = _lane_mask(model, image)
    return post_process_lane(mask, conf_map)

def _lane_mask(model, frame):
    h, w = frame.shape[:2]
    x = cv2.resize(frame, (512, 512))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf_map = probs.max(1)[0][0].cpu().numpy()
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask, conf_map

def _polygons(mask, conf_map, cls):
    binary = ((mask == cls).astype(np.uint8) * 255)
    cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in sorted(cs, key=cv2.contourArea, reverse=True)[:8] if cv2.contourArea(c) >= 5]

    region = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(region, kept, -1, 255, -1)
    confidence = float(conf_map[region > 0].mean()) if region.any() else 0.0

    polys = []
    for c in kept:
        p = c.reshape(-1, 2)
        if len(p) > 80:
            p = p[np.linspace(0, len(p) - 1, 80).astype(int)]
        if len(p) >= 3:
            polys.append(p.astype(int).tolist())
    return polys, confidence

def _dashed_lines(mask, conf_map):
    b = ((mask == 3).astype(np.uint8) * 255)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(b, 8)
    pts, keep_ids = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 5:
            continue
        pts.append((centers[i][0], centers[i][1]))
        keep_ids.append(i)

    region = np.isin(labels, keep_ids)
    confidence = float(conf_map[region].mean()) if region.any() else 0.0

    if len(pts) < 2:
        return [], confidence
    pts.sort(key=lambda p: p[1], reverse=True)
    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    deg = min(2, len(pts) - 1)
    coef = np.polyfit(ys, xs, deg)
    y = np.arange(int(min(ys)), int(max(ys)) + 1, 3)
    x = np.polyval(coef, y)
    h, w = mask.shape
    valid = (x >= 0) & (x < w)
    p = np.column_stack((x[valid], y[valid])).astype(int)
    if len(p) > 80:
        p = p[np.linspace(0, len(p) - 1, 80).astype(int)]
    connected = [p.tolist()] if len(p) >= 2 else []
    return connected, confidence

def _lane_center(mask, connected):
    h, w = mask.shape
    b = ((mask > 0).astype(np.uint8) * 255)
    for p in connected:
        cv2.polylines(b, [np.array(p, np.int32)], False, 255, 4)
    for y in np.linspace(int(h * .90), int(h * .55), 15).astype(int):
        band = b[max(0, y - 4):min(h, y + 5)] > 0
        xs = np.where(np.any(band, axis=0))[0]
        if len(xs) < 2:
            continue
        groups = np.split(xs, np.where(np.diff(xs) > 8)[0] + 1)
        centers = [int(g.mean()) for g in groups if len(g)]
        if len(centers) < 2:
            continue
        pairs = [(centers[i], centers[i + 1]) for i in range(len(centers) - 1)
                 if centers[i + 1] - centers[i] > w * .07]
        if not pairs:
            continue
        cx = w / 2
        pair = min(pairs, key=lambda p: abs((p[0] + p[1]) / 2 - cx))
        left, right = pair
        return {"x": int((left+right)/2), "y": int(y), "left_x": int(left), "right_x": int(right)}
    return None

def post_process_lane(mask, conf_map):
    poly_w, conf_w = _polygons(mask, conf_map, 1)
    poly_y, conf_y = _polygons(mask, conf_map, 2)
    connected, conf_p = _dashed_lines(mask, conf_map)
    center = _lane_center(mask, connected)

    return {
        "lane": [
            {"class_id": 1, "confidence": conf_w, "polygons": poly_w},
            {"class_id": 2, "confidence": conf_y, "polygons": poly_y},
            {"class_id": 3, "confidence": conf_p, "polygons": connected},
        ],
        "lane_center": [center] if center else []
    }