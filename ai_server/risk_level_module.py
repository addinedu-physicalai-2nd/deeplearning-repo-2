"""검출된 포트홀 박스를 조감도 면적으로 변환하는 모듈."""
"""
pothole 모델을 돌리는 python 코드에 
from risk_level_module import risk_level
을 선언해 주고
pothole 검출 되면

risk_level 함수를 선언

risk_level은 (image, potholes) 두 매게변수를 받는다.
image는 이미지 데이터
potholes 예시
 potholes = [
        (316, 265, 362, 282, 0.958), 첫번째 포트홀 좌표&confidence
        (168, 284, 209, 301, 0.680), 두번째 포트홀 좌표&confidence
        (302, 230, 340, 237, 0.390), 세번째 포트홀 좌표&confidence
    ]

이렇게 하면 potholes에 들어있는 순서대로 면적을 list에 담아서 반환한다.
"""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = PROJECT_ROOT / "perspective_calibration.npz"


def _load_calibration() -> tuple[np.ndarray, tuple[int, int]]:
    """risk_level.py가 생성한 조감도 보정값을 불러온다."""
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"보정 파일이 없습니다: {CALIBRATION_PATH}\n"
            "먼저 python3 src/risk_level.py를 실행하세요."
        )

    with np.load(CALIBRATION_PATH) as calibration:
        homography = calibration["homography"].astype(np.float64)
        source_size_array = calibration["source_size"].astype(int)

    source_size = (int(source_size_array[0]), int(source_size_array[1]))
    return homography, source_size


def _transform_for_image(
    homography: np.ndarray,
    source_size: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """현재 이미지 좌표를 보정 이미지 좌표에 맞춘 변환 행렬을 반환한다."""
    source_width, source_height = source_size
    image_width, image_height = image_size

    resize_to_source = np.array(
        [
            [source_width / image_width, 0, 0],
            [0, source_height / image_height, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return homography @ resize_to_source


def risk_level(
    image: np.ndarray,
    potholes: list[tuple[int, int, int, int, float]],
) -> list[int]:
    """검출 박스를 조감도에 투영하고 면적 목록을 제곱 픽셀로 반환한다.

    potholes의 각 항목은 (x1, y1, x2, y2, confidence) 튜플이어야 한다.
    """
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("image는 OpenCV 이미지(np.ndarray)여야 합니다.")
    if not isinstance(potholes, list):
        raise ValueError("potholes는 포트홀 튜플을 담은 list여야 합니다.")

    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image의 크기가 올바르지 않습니다.")

    homography, source_size = _load_calibration()
    transform = _transform_for_image(
        homography,
        source_size,
        (image_width, image_height),
    )

    areas = []
    for pothole in potholes:
        if not isinstance(pothole, tuple) or len(pothole) != 5:
            raise ValueError(
                "pothole은 (x1, y1, x2, y2, confidence) 튜플이어야 합니다: "
                f"{pothole}"
            )

        try:
            x1, y1, x2, y2 = (float(value) for value in pothole[:4])
            float(pothole[4])
        except (TypeError, ValueError) as error:
            raise ValueError(f"포트홀 좌표 또는 confidence가 잘못되었습니다: {pothole}") from error

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"바운딩 박스 좌표가 올바르지 않습니다: {pothole}")

        box_points = np.array(
            [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]],
            dtype=np.float32,
        )
        transformed_box = cv2.perspectiveTransform(box_points, transform)[0]
        area = round(abs(cv2.contourArea(transformed_box.astype(np.float32))))
        areas.append(area)

    level=''
    if area<1500:
        level='low'
    elif 1500<=area<=2000:
        level='middle'
    elif area>2000:
        level='high'

    return level