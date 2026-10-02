import sys

import json

import time

import threading

from pathlib import Path

from email.parser import BytesParser

from email.policy import default

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from urllib.parse import urlsplit

import cv2

import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets, uic

HOST = "0.0.0.0"

PORT = 5006

MAX_BODY_SIZE = 25 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent

UI_PATH = BASE_DIR / "ui.ui"

RECORDINGS_DIR = BASE_DIR / "recordings"

HTTP_SERVER = None

BRIDGE = None

class FrameBridge(QtCore.QObject):

    frame_received = QtCore.pyqtSignal(bytes, object)

def process_received_data(frame_id, image_data, ai_result):

    if BRIDGE:

        BRIDGE.frame_received.emit(image_data, ai_result)

def parse_multipart(content_type, body):

    message = BytesParser(policy=default).parsebytes(

        b"Content-Type: "

        + content_type.encode("latin-1")

        + b"\r\nMIME-Version: 1.0\r\n\r\n"

        + body

    )

    if not message.is_multipart():

        raise ValueError()

    fields = {}

    for part in message.iter_parts():

        name = part.get_param("name", header="content-disposition")

        if name:

            fields[name] = part.get_payload(decode=True)

    return fields

class FrameRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):

        pass

    def send_json(self, status_code, data):

        response = json.dumps(

            data,

            ensure_ascii=False

        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(

            "Content-Type",

            "application/json; charset=utf-8"

        )

        self.send_header(

            "Content-Length",

            str(len(response))

        )

        self.end_headers()

        self.wfile.write(response)

    def do_POST(self):

        if urlsplit(self.path).path != "/frame":

            self.send_json(404, {"error": "not found"})

            return

        content_type = self.headers.get("Content-Type", "")

        if not content_type.startswith("multipart/form-data"):

            self.send_json(

                400,

                {"error": "multipart/form-data required"}

            )

            return

        try:

            content_length = int(

                self.headers["Content-Length"]

            )

            if (

                content_length <= 0

                or content_length > MAX_BODY_SIZE

            ):

                raise ValueError()

            body = self.rfile.read(content_length)

            fields = parse_multipart(

                content_type,

                body

            )

            image_data = fields["frame"]

            ai_result = json.loads(

                fields["result"].decode("utf-8")

            )

            frame_id = ai_result["frame_id"]

        except Exception:

            self.send_json(

                400,

                {"error": "invalid frame or result"}

            )

            return

        process_received_data(

            frame_id,

            image_data,

            ai_result

        )

        self.send_json(

            200,

            {

                "status": "received",

                "frame_id": frame_id

            }

        )

def run_http_server():

    global HTTP_SERVER

    HTTP_SERVER = ThreadingHTTPServer(

        (HOST, PORT),

        FrameRequestHandler

    )

    print(

        f"HTTP receiver listening on {HOST}:{PORT}",

        flush=True

    )

    HTTP_SERVER.serve_forever()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, bridge):

        super().__init__()

        uic.loadUi(str(UI_PATH), self)

        self.bridge = bridge

        self.bridge.frame_received.connect(

            self.receive_frame

        )

        self.latest_frame = None

        self.latest_result = None

        self.frame_count = 0

        self.fps = 0.0

        self.fps_start = time.time()

        self.video_writer = None

        self.recording_path = None

        self.laneButton.setChecked(False)

        self.obstacleButton.setChecked(False)

        self.hazardButton.setChecked(False)

        self.dashcamButton.setChecked(False)

        self.laneButton.toggled.connect(

            lambda _: self.render_frame(False)

        )

        self.obstacleButton.toggled.connect(

            lambda _: self.render_frame(False)

        )

        self.hazardButton.toggled.connect(

            lambda _: self.render_frame(False)

        )

        self.dashcamButton.toggled.connect(

            self.toggle_dashcam

        )

        self.statusLabel.setText(

            "Waiting for web server..."

        )

        self.fpsValueLabel.setText("0.0")

    def receive_frame(self, image_data, ai_result):

        data = np.frombuffer(

            image_data,

            dtype=np.uint8

        )

        frame = cv2.imdecode(

            data,

            cv2.IMREAD_COLOR

        )

        if frame is None:

            return

        self.latest_frame = frame

        self.latest_result = ai_result

        self.update_fps()

        self.render_frame(True)

        self.update_status()

    def update_fps(self):

        self.frame_count += 1

        elapsed = time.time() - self.fps_start

        if elapsed >= 1.0:

            self.fps = self.frame_count / elapsed

            self.frame_count = 0

            self.fps_start = time.time()

            self.fpsValueLabel.setText(

                f"{self.fps:.1f}"

            )

    def draw_lane(self, frame, detections):

        lanes = detections.get(

            "lane",

            []

        )

        overlay = frame.copy()

        colors = {

            1: (255, 0, 0),

            2: (0, 0, 255),

            3: (80, 255, 80)

        }

        for lane in lanes:

            class_id = lane.get(

                "class_id"

            )

            color = colors.get(

                class_id,

                (255, 255, 255)

            )

            polygons = lane.get(

                "polygons",

                []

            )

            for polygon in polygons:

                pts = np.array(

                    polygon,

                    dtype=np.int32

                )

                if len(pts) < 2:

                    continue

                if class_id in (1, 2):

                    if len(pts) >= 3:

                        cv2.fillPoly(

                            overlay,

                            [pts],

                            color

                        )

                elif class_id == 3:

                    cv2.polylines(

                        overlay,

                        [pts],

                        False,

                        color,

                        4,

                        cv2.LINE_AA

                    )

        frame = cv2.addWeighted(

            overlay,

            0.40,

            frame,

            0.60,

            0

        )

        centers = detections.get(

            "lane_center",

            []

        )

        if centers:

            center = centers[0]

            if center:

                x = int(center["x"])

                y = int(center["y"])

                left_x = int(center["left_x"])

                right_x = int(center["right_x"])

                cv2.line(

                    frame,

                    (left_x, y),

                    (right_x, y),

                    (255, 255, 255),

                    1

                )

                cv2.circle(

                    frame,

                    (left_x, y),

                    4,

                    (255, 255, 255),

                    -1

                )

                cv2.circle(

                    frame,

                    (right_x, y),

                    4,

                    (255, 255, 255),

                    -1

                )

                cv2.circle(

                    frame,

                    (x, y),

                    9,

                    (255, 255, 255),

                    2

                )

                cv2.circle(

                    frame,

                    (x, y),

                    5,

                    (0, 0, 255),

                    -1

                )

        return frame

    def draw_box(

        self,

        frame,

        item,

        label,

        color

    ):

        box = item.get(

            "b_box",

            {}

        )

        try:

            x1 = int(box["x_min"])

            y1 = int(box["y_min"])

            x2 = int(box["x_max"])

            y2 = int(box["y_max"])

        except (KeyError, TypeError, ValueError):

            return

        confidence = float(

            item.get("confidence", 0)

        )

        text = (

            f"{label} "

            f"{confidence:.2f}"

        )

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            2

        )

        cv2.putText(

            frame,

            text,

            (x1, max(15, y1 - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.4,

            color,

            1,

            cv2.LINE_AA

        )

    def draw_obstacles(

        self,

        frame,

        detections

    ):

        classes = [

            ("people", "people"),

            ("car", "car"),

            ("trafficCone", "trafficCone")

        ]

        for key, label in classes:

            for item in detections.get(

                key,

                []

            ):

                self.draw_box(

                    frame,

                    item,

                    label,

                    (0, 255, 255)

                )

        return frame

    def draw_potholes(

        self,

        frame,

        detections

    ):

        for item in detections.get(

            "pothole",

            []

        ):

            box = item.get(

                "b_box",

                {}

            )

            try:

                x1 = int(box["x_min"])

                y1 = int(box["y_min"])

                x2 = int(box["x_max"])

                y2 = int(box["y_max"])

            except (

                KeyError,

                TypeError,

                ValueError

            ):

                continue

            confidence = float(

                item.get(

                    "confidence",

                    0

                )

            )

            level = item.get(

                "risk_level",

                item.get("level", "")

            )

            text = (

                f"pothole {confidence:.2f}"

            )

            color = (

                255,

                0,

                255

            )

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                color,

                2

            )

            cv2.putText(

                frame,

                text,

                (x1, max(15, y1 - 5)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.4,

                color,

                1,

                cv2.LINE_AA

            )

            if level != "":

                text_width = cv2.getTextSize(

                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1

                )[0][0]

                cv2.putText(

                    frame,

                    f" level:{level}",

                    (x1 + text_width, max(15, y1 - 5)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    color,

                    2,

                    cv2.LINE_AA

                )

        return frame

    def render_frame(

        self,

        write_record=False

    ):

        if self.latest_frame is None:

            return

        frame = self.latest_frame.copy()

        result = (

            self.latest_result

            if isinstance(

                self.latest_result,

                dict

            )

            else {}

        )

        detections = result.get(

            "detections",

            {}

        )

        if self.laneButton.isChecked():

            frame = self.draw_lane(

                frame,

                detections

            )

        if self.obstacleButton.isChecked():

            frame = self.draw_obstacles(

                frame,

                detections

            )

        if self.hazardButton.isChecked():

            frame = self.draw_potholes(

                frame,

                detections

            )

        if (

            write_record

            and self.dashcamButton.isChecked()

        ):

            self.write_recording(frame)

        self.show_frame(frame)

    def show_frame(self, frame):

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )

        h, w, ch = rgb.shape

        qimage = QtGui.QImage(

            rgb.data,

            w,

            h,

            ch * w,

            QtGui.QImage.Format.Format_RGB888

        ).copy()

        pixmap = QtGui.QPixmap.fromImage(

            qimage

        )

        pixmap = pixmap.scaled(

            self.videoLabel.size(),

            QtCore.Qt.AspectRatioMode.KeepAspectRatio,

            QtCore.Qt.TransformationMode.SmoothTransformation

        )

        self.videoLabel.setPixmap(

            pixmap

        )

    def toggle_dashcam(self, checked):

        if checked:

            RECORDINGS_DIR.mkdir(

                parents=True,

                exist_ok=True

            )

            self.video_writer = None

        else:

            self.stop_recording()

        self.update_status()

    def write_recording(self, frame):

        h, w = frame.shape[:2]

        if self.video_writer is None:

            timestamp = time.strftime(

                "%Y%m%d_%H%M%S"

            )

            self.recording_path = (

                RECORDINGS_DIR

                / f"dashcam_{timestamp}.mp4"

            )

            fourcc = cv2.VideoWriter_fourcc(

                *"mp4v"

            )

            fps = (

                self.fps

                if self.fps >= 5

                else 30.0

            )

            self.video_writer = cv2.VideoWriter(

                str(self.recording_path),

                fourcc,

                fps,

                (w, h)

            )

            print(

                f"Recording started: "

                f"{self.recording_path}"

            )

        self.video_writer.write(

            frame

        )

    def stop_recording(self):

        if self.video_writer is not None:

            self.video_writer.release()

            self.video_writer = None

            print(

                f"Recording saved: "

                f"{self.recording_path}"

            )

    def update_status(self):

        if self.latest_frame is None:

            self.statusLabel.setText(

                "Waiting for web server..."

            )

            return

        active = []

        if self.laneButton.isChecked():

            active.append("LANE")

        if self.obstacleButton.isChecked():

            active.append("OBSTACLE")

        if self.hazardButton.isChecked():

            active.append("HAZARD")

        if self.dashcamButton.isChecked():

            active.append("DASHCAM")

        if active:

            self.statusLabel.setText(

                "Connected | "

                + " | ".join(active)

            )

        else:

            self.statusLabel.setText(

                "Connected"

            )

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.latest_frame is not None:

            self.render_frame(False)

    def closeEvent(self, event):

        self.stop_recording()

        global HTTP_SERVER

        if HTTP_SERVER is not None:

            HTTP_SERVER.shutdown()

        event.accept()

if __name__ == "__main__":

    app = QtWidgets.QApplication(

        sys.argv

    )

    BRIDGE = FrameBridge()

    window = MainWindow(

        BRIDGE

    )

    window.show()

    threading.Thread(

        target=run_http_server,

        daemon=True

    ).start()

    sys.exit(

        app.exec()

    )