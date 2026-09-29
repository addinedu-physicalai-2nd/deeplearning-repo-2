import queue
import socket
import struct
import threading
import time

import requests
from flask import Flask, jsonify


app = Flask(__name__)

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
AI_URL = "http://192.168.0.131:8000/inference"
UDP_HEADER_FORMAT = "!HBB"
UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)
FRAME_TIMEOUT = 1.0
FRAME_BUFFER_SIZE = 10

frame_buffer = queue.Queue(maxsize=FRAME_BUFFER_SIZE)

latest_status = {
    "received_frame_id": None,
    "ai_frame_id": None,
    "robot_address": None,
    "buffer_size": 0,
    "ai_result": None,
    "error": None,
}


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
    app.logger.info("UDP image receiver listening on %s:%s", UDP_HOST, UDP_PORT)

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


def forward_frames_to_ai():
    """버퍼의 프레임을 하나씩 꺼내 AI 서버에 전달한다."""
    while True:
        frame_id, image_data = frame_buffer.get()
        latest_status["buffer_size"] = frame_buffer.qsize()

        try:
            response = requests.post(
                AI_URL,
                data=image_data,
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Frame-ID": str(frame_id),
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
    threading.Thread(target=receive_frames, daemon=True).start()
    threading.Thread(target=forward_frames_to_ai, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
