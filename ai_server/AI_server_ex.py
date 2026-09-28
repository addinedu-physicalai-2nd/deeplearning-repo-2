import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def detect_image(image_data):
    """추후 image_data에 AI 모델을 적용하고 탐지 결과를 반환한다."""
    return {
        "pothole": [],
        "people": [],
        "car": [],
        "trafficCone": [],
        "lane": [],
    }


class AIHandler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/infer":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            frame_id = int(self.headers["X-Frame-ID"])
            content_length = int(self.headers["Content-Length"])
        except (KeyError, TypeError, ValueError):
            self.send_json({"error": "X-Frame-ID가 필요합니다."}, 400)
            return

        image_data = self.rfile.read(content_length)
        if not image_data.startswith(b"\xff\xd8"):
            self.send_json({"error": "요청 body는 JPEG 형식이어야 합니다."}, 400)
            return

        result = {
            "frame_id": frame_id,
            "detections": detect_image(image_data),
        }
        self.send_json(result)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8001), AIHandler)
    print("AI server listening on http://0.0.0.0:8001")
    server.serve_forever()
