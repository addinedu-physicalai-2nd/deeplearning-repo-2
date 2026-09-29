from pathlib import Path
import gdown

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LINK_FILE = MODELS_DIR / "model_drive_link.txt"
REQUIRED = ["obstacle.pt", "pothole.pt"]   # lane은 *.pth로 따로 확인


def _all_present():
    has_required = all((MODELS_DIR / name).exists() for name in REQUIRED)
    has_lane = any(MODELS_DIR.glob("*.pth"))
    return has_required and has_lane


def ensure_models():
    """models/ 안에 가중치가 없으면 Drive 폴더에서 받아온다."""
    if _all_present():
        return MODELS_DIR
    url = LINK_FILE.read_text().strip()
    gdown.download_folder(url=url, output=str(MODELS_DIR), quiet=False)
    if not _all_present():
        raise FileNotFoundError(f"Drive에서 받은 뒤에도 {MODELS_DIR}에 모델 파일이 부족합니다")
    return MODELS_DIR
