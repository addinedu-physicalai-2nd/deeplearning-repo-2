"""AI Server HTTP 통신 모듈."""

import queue

import requests

from web_server.network_modules.connection_events import (
    ConnectionEventLogger,
)


def _put_latest(target_buffer, item):
    dropped = False
    if target_buffer.full():
        try:
            target_buffer.get_nowait()
            target_buffer.task_done()
            dropped = True
        except queue.Empty:
            pass
    target_buffer.put_nowait(item)
    return dropped


def _validate_ai_result(frame_id, ai_result):
    if not isinstance(ai_result, dict):
        raise ValueError("AI response must be a JSON object")
    if ai_result.get("frame_id") != frame_id:
        raise ValueError(
            f"frame_id mismatch: request={frame_id}, "
            f"response={ai_result.get('frame_id')}"
        )


def _normalize_ai_result(ai_result):
    """AI의 pothole.level을 내부 규격인 risk_level로 변환한다."""
    detections = ai_result.get("detections")
    if not isinstance(detections, dict):
        return ai_result

    potholes = detections.get("pothole")
    if not isinstance(potholes, list):
        return ai_result

    for pothole in potholes:
        if not isinstance(pothole, dict) or "level" not in pothole:
            continue
        if pothole.get("risk_level") is None:
            pothole["risk_level"] = pothole["level"]
        del pothole["level"]

    return ai_result


def forward_frames_to_ai(
    frame_buffer,
    monitoring_buffer,
    db_buffer,
    state,
    logger,
    stop_event,
    ai_url,
    timeout=3.0,
    session_factory=requests.Session,
):
    """AI 결과를 Monitoring에는 JPEG와, DB에는 JSON만 전달한다."""
    connection = ConnectionEventLogger(
        "AI",
        state,
        "ai_connected",
        logger,
    )
    with session_factory() as session:
        while not stop_event.is_set():
            try:
                frame_id, image_data = frame_buffer.get(timeout=0.5)
            except queue.Empty:
                continue

            state.update(buffer_size=frame_buffer.qsize())
            try:
                response = session.post(
                    ai_url,
                    params={"frame_id": frame_id},
                    files={
                        "file": ("frame.jpg", image_data, "image/jpeg")
                    },
                    timeout=timeout,
                )
                response.raise_for_status()

                ai_result = response.json()
                _validate_ai_result(frame_id, ai_result)
                _normalize_ai_result(ai_result)
                dropped = _put_latest(
                    monitoring_buffer,
                    (frame_id, image_data, ai_result),
                )
                db_dropped = _put_latest(
                    db_buffer,
                    (frame_id, ai_result),
                )

                state.update(
                    ai_frame_id=frame_id,
                    ai_result=ai_result,
                    error=None,
                    qt_buffer_size=monitoring_buffer.qsize(),
                    db_buffer_size=db_buffer.qsize(),
                )
                if dropped:
                    state.increment("qt_dropped_frames")
                if db_dropped:
                    state.increment("db_dropped_frames")

                connection.connection_succeeded(ai_url)
            except (requests.ConnectionError, requests.Timeout) as error:
                state.update(
                    ai_result=None,
                    error=f"frame {frame_id}: {error}",
                )
                if connection.connected:
                    connection.connection_lost(error)
                else:
                    connection.send_failed(error)
            except requests.RequestException as error:
                state.update(
                    ai_result=None,
                    error=f"frame {frame_id}: {error}",
                )
                connection.receive_failed(error)
            except (TypeError, ValueError) as error:
                state.update(
                    ai_result=None,
                    error=f"frame {frame_id}: {error}",
                )
                connection.receive_failed(error)
            finally:
                frame_buffer.task_done()
                state.update(buffer_size=frame_buffer.qsize())
