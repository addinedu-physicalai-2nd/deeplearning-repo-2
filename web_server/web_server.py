"""기능별 모듈을 사용하는 리팩토링 Web Server."""

import os
import queue
import threading

from flask import Flask, jsonify, render_template, request

from web_server.network_modules.ai_module import forward_frames_to_ai
from web_server.network_modules.db_module import (
    delete_all_detection_data,
    fetch_dashboard_summary,
    fetch_table_page,
    store_ai_results,
)
from web_server.network_modules.logging_module import configure_logging
from web_server.network_modules.monitoring_module import (
    forward_frames_to_monitoring,
)
from web_server.network_modules.udp_module import receive_frames


UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
UDP_HEADER_FORMAT = "!HBB"
FRAME_TIMEOUT = 1.0
ROBOT_DISCONNECT_TIMEOUT = 3.0
FRAME_BUFFER_SIZE = 10

AI_URL = os.environ.get(
    "AI_URL",
    "http://192.168.0.4:8000/inference",
)
AI_TIMEOUT = 3.0

MONITORING_URL = os.environ.get(
    "MONITORING_URL",
    os.environ.get("QT_HTTP_URL", "http://192.168.0.4:5006/frame"),
)
MONITORING_TIMEOUT = 2.0
MONITORING_BUFFER_SIZE = 2

DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "1234"
DB_NAME = "road_gaurd"
DB_BUFFER_SIZE = 100
DB_RETRY_INTERVAL = 5.0
DB_RESET_CONFIRMATION = "DELETE_ALL_DATA"
DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
}

STATUS_HOST = "0.0.0.0"
STATUS_PORT = 5000


class RuntimeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {
            "received_frame_id": None,
            "robot_connected": False,
            "robot_error": None,
            "ai_frame_id": None,
            "ai_connected": False,
            "robot_address": None,
            "buffer_size": 0,
            "ai_result": None,
            "error": None,
            "qt_frame_id": None,
            "qt_connected": False,
            "qt_buffer_size": 0,
            "qt_dropped_frames": 0,
            "qt_error": None,
            "db_connected": False,
            "db_frame_id": None,
            "db_buffer_size": 0,
            "db_dropped_frames": 0,
            "db_error": None,
        }

    def update(self, **values):
        with self._lock:
            self._values.update(values)

    def increment(self, key, amount=1):
        with self._lock:
            self._values[key] += amount

    def snapshot(self):
        with self._lock:
            return dict(self._values)


app = Flask(__name__)
state = RuntimeState()
frame_buffer = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
monitoring_buffer = queue.Queue(maxsize=MONITORING_BUFFER_SIZE)
db_buffer = queue.Queue(maxsize=DB_BUFFER_SIZE)
db_operation_lock = threading.Lock()


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/status")
def status():
    return jsonify(state.snapshot())


@app.get("/api/dashboard/summary")
def dashboard_summary():
    try:
        return jsonify(fetch_dashboard_summary(DB_CONFIG))
    except Exception as error:
        app.logger.error("Failed to read dashboard summary: %s", error)
        return jsonify({"error": "DB 요약 데이터를 조회하지 못했습니다."}), 503


def discard_queued_db_results():
    """초기화 이전에 대기 중이던 DB 저장 항목을 제거한다."""
    discarded = 0
    while True:
        try:
            db_buffer.get_nowait()
        except queue.Empty:
            return discarded
        db_buffer.task_done()
        discarded += 1


@app.post("/api/database/reset")
def reset_database():
    payload = request.get_json(silent=True) or {}
    if payload.get("confirmation") != DB_RESET_CONFIRMATION:
        return jsonify({"error": "DB 초기화 확인값이 올바르지 않습니다."}), 400

    try:
        with db_operation_lock:
            result = delete_all_detection_data(DB_CONFIG)
            discarded = discard_queued_db_results()
    except Exception as error:
        app.logger.error("Failed to reset database: %s", error)
        state.update(db_error=str(error))
        return jsonify({"error": "DB 데이터를 초기화하지 못했습니다."}), 503

    state.update(
        db_connected=True,
        db_frame_id=None,
        db_buffer_size=db_buffer.qsize(),
        db_error=None,
    )
    app.logger.warning(
        "All dashboard data deleted: %s rows, %s queued results discarded",
        result["deleted_rows"],
        discarded,
    )
    return jsonify(
        {
            "status": "deleted",
            **result,
            "discarded_queued_results": discarded,
        }
    )


@app.get("/api/tables/<table_name>")
def table_data(table_name):
    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=20, type=int)
    frame_value = request.args.get("frame", default="").strip()

    if page is None or page_size is None:
        return jsonify({"error": "잘못된 페이지 값입니다."}), 400
    try:
        frame_id = int(frame_value) if frame_value else None
    except ValueError:
        return jsonify({"error": "잘못된 프레임 번호입니다."}), 400
    if frame_id is not None and frame_id < 0:
        return jsonify({"error": "프레임 번호는 0 이상이어야 합니다."}), 400

    try:
        return jsonify(
            fetch_table_page(
                DB_CONFIG,
                table_name,
                page=page,
                page_size=page_size,
                frame_id=frame_id,
            )
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.error("Failed to read table %s: %s", table_name, error)
        return jsonify({"error": "DB 테이블을 조회하지 못했습니다."}), 503


def start_worker(name, target, *args):
    thread = threading.Thread(
        target=target,
        args=args,
        name=name,
        daemon=True,
    )
    thread.start()
    return thread


def main():
    logger, log_listener = configure_logging(app)
    stop_event = threading.Event()

    workers = [
        start_worker(
            "udp-receiver",
            receive_frames,
            frame_buffer,
            state,
            logger,
            stop_event,
            UDP_HOST,
            UDP_PORT,
            UDP_HEADER_FORMAT,
            FRAME_TIMEOUT,
            ROBOT_DISCONNECT_TIMEOUT,
        ),
        start_worker(
            "ai-client",
            forward_frames_to_ai,
            frame_buffer,
            monitoring_buffer,
            db_buffer,
            state,
            logger,
            stop_event,
            AI_URL,
            AI_TIMEOUT,
        ),
        start_worker(
            "monitoring-client",
            forward_frames_to_monitoring,
            monitoring_buffer,
            state,
            logger,
            stop_event,
            MONITORING_URL,
            MONITORING_TIMEOUT,
        ),
        start_worker(
            "db-writer",
            store_ai_results,
            db_buffer,
            state,
            logger,
            stop_event,
            DB_CONFIG,
            DB_RETRY_INTERVAL,
            db_operation_lock,
        ),
    ]

    try:
        app.run(
            host=STATUS_HOST,
            port=STATUS_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=1.5)
        log_listener.stop()


if __name__ == "__main__":
    main()
