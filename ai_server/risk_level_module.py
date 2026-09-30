"""포트홀 박스를 조감도 좌표로 투영해 위험도를 계산한다.

기본 보정은 robot_view 좌표를 over_view 좌표로 변환한다. 기존 코드에서
사용하던 ``risk_level(image, potholes)`` 함수를 공개 진입점으로 유지한다.
조감도 면적은 low/middle/high 판정에만 내부적으로 사용한다.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = (
    MODULE_DIR / "perspective_calibration_robot_to_over.npz"
)

# race_test.webm의 유효 조감도 면적 분포를 로그 공간 3-군집으로 분석한 경계.
# 재현 코드와 결과: src/pothole_risk_threshold_analysis/
DEFAULT_MIDDLE_THRESHOLD_PX2 = 551
DEFAULT_HIGH_THRESHOLD_PX2 = 901


@dataclass(frozen=True)
class PotholeMeasurement:
    """한 포트홀의 조감도 측정 결과."""

    area_px2: int
    level: str
    polygon: tuple[tuple[float, float], ...]


def _validate_homography(homography: np.ndarray) -> np.ndarray:
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError("homography는 3x3 행렬이어야 합니다.")
    if not np.isfinite(homography).all():
        raise ValueError("homography에 유효하지 않은 값이 있습니다.")
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("homography가 역변환 불가능한 행렬입니다.")
    return homography


def _read_size(calibration, key: str) -> tuple[int, int]:
    values = np.asarray(calibration[key], dtype=int).reshape(-1)
    if values.size != 2 or np.any(values <= 0):
        raise ValueError(f"{key}는 양수인 (width, height)여야 합니다.")
    return int(values[0]), int(values[1])


def _load_calibration_file(
    calibration_path: Path,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"보정 파일이 없습니다: {calibration_path}\n"
            "먼저 ai_server/calibrate_perspective.py를 실행하세요."
        )

    with np.load(calibration_path, allow_pickle=False) as calibration:
        required = {"homography", "source_size"}
        missing = required.difference(calibration.files)
        if missing:
            raise ValueError(
                "보정 파일에 필수 값이 없습니다: "
                + ", ".join(sorted(missing))
            )

        homography = _validate_homography(calibration["homography"])
        source_size = _read_size(calibration, "source_size")

        destination_key = (
            "destination_size"
            if "destination_size" in calibration.files
            else "output_size"
        )
        destination_size = _read_size(calibration, destination_key)

    return homography, source_size, destination_size


def _load_calibration() -> tuple[np.ndarray, tuple[int, int]]:
    """기존 호출부와 호환되는 보정값 로더."""
    homography, source_size, _ = _load_calibration_file(CALIBRATION_PATH)
    return homography, source_size


def _transform_for_image(
    homography: np.ndarray,
    source_size: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """현재 입력 해상도를 보정 이미지 좌표계에 맞춘다."""
    source_width, source_height = source_size
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image의 크기가 올바르지 않습니다.")

    resize_to_source = np.array(
        [
            [source_width / image_width, 0, 0],
            [0, source_height / image_height, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return homography @ resize_to_source


class PotholeRiskEstimator:
    """보정값을 한 번 로드하고 반복적으로 포트홀을 측정하는 모듈."""

    def __init__(
        self,
        calibration_path: str | Path = CALIBRATION_PATH,
        middle_threshold_px2: int = DEFAULT_MIDDLE_THRESHOLD_PX2,
        high_threshold_px2: int = DEFAULT_HIGH_THRESHOLD_PX2,
    ):
        middle_threshold_px2 = int(middle_threshold_px2)
        high_threshold_px2 = int(high_threshold_px2)
        if middle_threshold_px2 <= 0:
            raise ValueError("middle_threshold_px2는 0보다 커야 합니다.")
        if high_threshold_px2 <= middle_threshold_px2:
            raise ValueError(
                "high_threshold_px2는 middle_threshold_px2보다 커야 합니다."
            )

        self.calibration_path = Path(calibration_path).resolve()
        (
            self.homography,
            self.source_size,
            self.destination_size,
        ) = _load_calibration_file(self.calibration_path)
        self.middle_threshold_px2 = middle_threshold_px2
        self.high_threshold_px2 = high_threshold_px2

    def area_to_level(self, area_px2: int) -> str:
        """조감도 픽셀 면적을 low/middle/high로 분류한다."""
        if area_px2 < self.middle_threshold_px2:
            return "low"
        if area_px2 <= self.high_threshold_px2:
            return "middle"
        return "high"

    def transform_for_image(self, image: np.ndarray) -> np.ndarray:
        """현재 이미지 해상도에 사용할 변환 행렬을 반환한다."""
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise ValueError("image는 OpenCV 이미지(np.ndarray)여야 합니다.")
        image_height, image_width = image.shape[:2]
        return _transform_for_image(
            self.homography,
            self.source_size,
            (image_width, image_height),
        )

    @staticmethod
    def _parse_pothole(
        pothole,
    ) -> tuple[float, float, float, float, float]:
        if not isinstance(pothole, (tuple, list)) or len(pothole) != 5:
            raise ValueError(
                "pothole은 (x1, y1, x2, y2, confidence) 형식이어야 합니다: "
                f"{pothole}"
            )
        try:
            x1, y1, x2, y2, confidence = (
                float(value) for value in pothole
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"포트홀 좌표 또는 confidence가 잘못되었습니다: {pothole}"
            ) from error

        if not np.isfinite((x1, y1, x2, y2, confidence)).all():
            raise ValueError(f"포트홀 값은 유한한 숫자여야 합니다: {pothole}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"바운딩 박스 좌표가 올바르지 않습니다: {pothole}")
        return x1, y1, x2, y2, confidence

    def measure(
        self,
        image: np.ndarray,
        potholes: list[tuple[float, float, float, float, float]],
    ) -> list[PotholeMeasurement]:
        """포트홀별 조감도 다각형, 픽셀 면적과 위험도를 반환한다."""
        if not isinstance(potholes, list):
            raise ValueError("potholes는 포트홀 목록이어야 합니다.")
        if not potholes:
            return []

        transform = self.transform_for_image(image)
        measurements = []
        for pothole in potholes:
            x1, y1, x2, y2, _ = self._parse_pothole(pothole)
            box_points = np.array(
                [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]],
                dtype=np.float32,
            )
            transformed = cv2.perspectiveTransform(box_points, transform)[0]
            if not np.isfinite(transformed).all():
                raise ValueError(
                    f"조감도 변환 결과가 유효하지 않습니다: {pothole}"
                )

            area_px2 = round(
                abs(cv2.contourArea(transformed.astype(np.float32)))
            )
            polygon = tuple(
                (float(point[0]), float(point[1]))
                for point in transformed
            )
            measurements.append(
                PotholeMeasurement(
                    area_px2=area_px2,
                    level=self.area_to_level(area_px2),
                    polygon=polygon,
                )
            )

        return measurements


_default_estimator = None
_default_estimator_mtime_ns = None
_default_estimator_lock = Lock()


def _get_default_estimator() -> PotholeRiskEstimator:
    """보정 파일이 바뀌면 자동으로 다시 로드하는 기본 인스턴스."""
    global _default_estimator, _default_estimator_mtime_ns

    try:
        mtime_ns = CALIBRATION_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = None

    with _default_estimator_lock:
        if (
            _default_estimator is None
            or _default_estimator_mtime_ns != mtime_ns
        ):
            _default_estimator = PotholeRiskEstimator(CALIBRATION_PATH)
            _default_estimator_mtime_ns = mtime_ns
        return _default_estimator


def _measure_potholes(
    image: np.ndarray,
    potholes: list[tuple[float, float, float, float, float]],
) -> list[PotholeMeasurement]:
    """기본 보정값으로 위험도 판정에 필요한 내부 측정값을 계산한다."""
    if not potholes:
        return []
    return _get_default_estimator().measure(image, potholes)


def risk_level(
    image: np.ndarray,
    potholes: list[tuple[float, float, float, float, float]],
) -> list[str]:
    """기존 API: 입력 순서대로 low/middle/high 목록을 반환한다."""
    return [item.level for item in _measure_potholes(image, potholes)]
