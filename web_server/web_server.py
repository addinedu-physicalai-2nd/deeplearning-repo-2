import json
import logging
import os
import queue
import socket
import struct
import threading
import time
from logging.handlers import QueueHandler, QueueListener

import requests
from flask import Flask, jsonify


app = Flask(__name__)

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
AI_URL = "http://192.168.0.4:8000/inference"
UDP_HEADER_FORMAT = "!HBB"
UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)
FRAME_TIMEOUT = 1.0
FRAME_BUFFER_SIZE = 10
QT_HTTP_URL = os.environ.get("QT_HTTP_URL", "http://192.168.0.4:7000/frame")
QT_HTTP_TIMEOUT = 2.0
QT_BUFFER_SIZE = 2

frame_buffer = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
qt_buffer = queue.Queue(maxsize=QT_BUFFER_SIZE)
udp_traffic_logged = threading.Event()

latest_status = {
    "received_frame_id": None,
    "ai_frame_id": None,
    "robot_address": None,
    "buffer_size": 0,
    "ai_result": None,
    "error": None,
    "qt_frame_id": None,
    "qt_buffer_size": 0,
    "qt_dropped_frames": 0,
    "qt_error": None,
}


def configure_logging():
    """로그 출력은 전용 스레드가 처리하도록 큐 기반 로깅을 시작한다."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.terminator = "\n\n"

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)

    for logger in (app.logger, logging.getLogger("werkzeug")):
        logger.handlers.clear()
        logger.addHandler(queue_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    listener = QueueListener(
        log_queue,
        console_handler,
        respect_handler_level=True,
    )
    listener.start()
    return listener


def receive_udp_frame(sock, pending_frames):
    """분할된 UDP 패킷을 frame_id별로 모아 JPEG 한 장으로 반환한다."""
    while True:
        packet, robot_address = sock.recvfrom(65535)
        now = time.monotonic()

        for key, frame in list(pending_frames.items()):
            if now - frame["updated_at"] > FRAME_TIMEOUT:
                del pending_frames[key]

        if len(packet) <= UDP_HEADER_SIZE:
            continue

        frame_id, chunk_idx, total_chunks = struct.unpack(
            UDP_HEADER_FORMAT, packet[:UDP_HEADER_SIZE]
        )
        if total_chunks == 0 or chunk_idx >= total_chunks:
            continue

        if not udp_traffic_logged.is_set():
            udp_traffic_logged.set()
            app.logger.info(
                "UDP communication active: first packet received from %s:%s",
                robot_address[0],
                robot_address[1],
            )

        key = (robot_address, frame_id)
        frame = pending_frames.get(key)
        if frame is None or frame["total_chunks"] != total_chunks:
            frame = {
                "total_chunks": total_chunks,
                "chunks": {},
                "updated_at": now,
            }
            pending_frames[key] = frame

        frame["chunks"][chunk_idx] = packet[UDP_HEADER_SIZE:]
        frame["updated_at"] = now

        if len(frame["chunks"]) != total_chunks:
            continue

        image_data = b"".join(frame["chunks"][idx] for idx in range(total_chunks))
        del pending_frames[key]

        if not image_data.startswith(b"\xff\xd8"):
            app.logger.warning("Ignored invalid JPEG frame from %s", robot_address)
            continue

        return image_data, robot_address, frame_id


def receive_frames():
    """UDP 프레임을 계속 수신하여 프레임 버퍼에 저장한다."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    pending_frames = {}

    while True:
        image_data, robot_address, frame_id = receive_udp_frame(sock, pending_frames)

        latest_status["received_frame_id"] = frame_id
        latest_status["robot_address"] = robot_address[0]

        if frame_buffer.full():
            try:
                frame_buffer.get_nowait()
                frame_buffer.task_done()
            except queue.Empty:
                pass

        frame_buffer.put((frame_id, image_data))
        latest_status["buffer_size"] = frame_buffer.qsize()


def enqueue_frame_for_qt(frame_id, image_data, ai_result):
    """Qt 전송 대기열에 최신 결과를 넣되 AI 처리 스레드를 기다리게 하지 않는다."""
    if qt_buffer.full():
        try:
            qt_buffer.get_nowait()
            qt_buffer.task_done()
            latest_status["qt_dropped_frames"] += 1
        except queue.Empty:
            pass

    try:
        qt_buffer.put_nowait((frame_id, image_data, ai_result))
    except queue.Full:
        # 소비자 스레드와의 경합으로 큐가 다시 찬 경우 현재 프레임을 버린다.
        latest_status["qt_dropped_frames"] += 1

    latest_status["qt_buffer_size"] = qt_buffer.qsize()


def forward_frames_to_qt():
    """AI 처리가 끝난 프레임과 JSON을 독립적으로 Qt HTTP 서버에 전달한다."""
    while True:
        frame_id, image_data, ai_result = qt_buffer.get()
        latest_status["qt_buffer_size"] = qt_buffer.qsize()

        try:
            result_data = json.dumps(
                ai_result,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            response = requests.post(
                QT_HTTP_URL,
                files={
                    "frame": ("frame.jpg", image_data, "image/jpeg"),
                    "result": ("result.json", result_data, "application/json"),
                },
                timeout=QT_HTTP_TIMEOUT,
            )
            response.raise_for_status()

            latest_status["qt_frame_id"] = frame_id
            latest_status["qt_error"] = None
            app.logger.info("Frame %s forwarded to HTTP receiver", frame_id)
        except (requests.RequestException, TypeError, ValueError) as error:
            # Qt 장애는 기록만 하고 UDP 수신 및 AI 통신에는 전파하지 않는다.
            latest_status["qt_error"] = f"frame {frame_id}: {error}"
            app.logger.warning(
                "Failed to forward frame %s to HTTP receiver: %s",
                frame_id,
                error,
            )
        finally:
            qt_buffer.task_done()
            latest_status["qt_buffer_size"] = qt_buffer.qsize()


def forward_frames_to_ai():
    """버퍼의 프레임을 하나씩 꺼내 AI 서버에 전달한다."""
    while True:
        frame_id, image_data = frame_buffer.get()
        latest_status["buffer_size"] = frame_buffer.qsize()

        try:
            response = requests.post(
                AI_URL,
                params={"frame_id": frame_id},
                files={
                    "file": ("frame.jpg", image_data, "image/jpeg"),
                },
                timeout=3,
            )
            response.raise_for_status()

            ai_result = response.json()
            if ai_result.get("frame_id") != frame_id:
                raise ValueError(
                    f"frame_id 불일치: 요청={frame_id}, 응답={ai_result.get('frame_id')}"
                )

            latest_status["ai_frame_id"] = frame_id
            latest_status["ai_result"] = ai_result
            latest_status["error"] = None
            enqueue_frame_for_qt(frame_id, image_data, ai_result)
            app.logger.info("Frame %s processed by AI server", frame_id)

        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            latest_status["ai_result"] = None
            latest_status["error"] = f"frame {frame_id}: {error}"
            app.logger.error("Failed to forward frame %s: %s", frame_id, error)
        finally:
            frame_buffer.task_done()
            latest_status["buffer_size"] = frame_buffer.qsize()


@app.get("/status")
def status():
    """마지막 프레임의 AI 처리 상태를 반환한다."""
    return jsonify(latest_status)


if __name__ == "__main__":
    log_listener = configure_logging()
    threading.Thread(target=receive_frames, daemon=True).start()
    threading.Thread(target=forward_frames_to_ai, daemon=True).start()
    threading.Thread(target=forward_frames_to_qt, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        log_listener.stop()
