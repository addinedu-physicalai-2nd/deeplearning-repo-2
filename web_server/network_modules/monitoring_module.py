"""Monitoring HTTP 통신 모듈."""

import json
import queue

import requests


def forward_frames_to_monitoring(
    monitoring_buffer,
    state,
    logger,
    stop_event,
    monitoring_url,
    timeout=2.0,
    session_factory=requests.Session,
):
    """원본 JPEG와 같은 프레임의 AI JSON을 Monitoring으로 전송한다."""
    with session_factory() as session:
        while not stop_event.is_set():
            try:
                frame_id, image_data, ai_result = monitoring_buffer.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue

            state.update(qt_buffer_size=monitoring_buffer.qsize())
            try:
                result_data = json.dumps(
                    ai_result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                response = session.post(
                    monitoring_url,
                    files={
                        "frame": ("frame.jpg", image_data, "image/jpeg"),
                        "result": (
                            "result.json",
                            result_data,
                            "application/json",
                        ),
                    },
                    timeout=timeout,
                )
                response.raise_for_status()

                state.update(qt_frame_id=frame_id, qt_error=None)
                logger.info("Frame %s forwarded to Monitoring", frame_id)
            except (requests.RequestException, TypeError, ValueError) as error:
                state.update(qt_error=f"frame {frame_id}: {error}")
                logger.warning(
                    "Failed to forward frame %s to Monitoring: %s",
                    frame_id,
                    error,
                )
            finally:
                monitoring_buffer.task_done()
                state.update(qt_buffer_size=monitoring_buffer.qsize())
