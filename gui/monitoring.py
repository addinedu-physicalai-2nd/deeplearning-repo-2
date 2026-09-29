"""flask_test.py에서 이미지와 AI 결과 JSON을 받는 HTTP 서버."""

import json
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOST = "0.0.0.0"
PORT = 5006
MAX_BODY_SIZE = 25 * 1024 * 1024


def process_received_data(frame_id, image_data, ai_result):
    """수신 데이터를 사용하는 코드를 이 함수에 구현한다."""
    print(
        f"frame_id={frame_id}, image_bytes={len(image_data)}, "
        f"json={json.dumps(ai_result, ensure_ascii=False)}",
        flush=True,
    )


def parse_multipart(content_type, body):
    """multipart/form-data 본문을 {필드명: bytes} 형태로 반환한다."""
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("latin-1")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )

    if not message.is_multipart():
        raise ValueError("multipart/form-data 요청이 아닙니다.")

    fields = {}
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name:
            fields[field_name] = part.get_payload(decode=True)
    return fields


class FrameRequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code, data):
        response_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data)

    def do_POST(self):
        if urlsplit(self.path).path != "/frame":
            self.send_json(404, {"error": "요청 경로를 찾을 수 없습니다."})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json(400, {"error": "multipart/form-data 요청이 필요합니다."})
            return

        try:
            content_length = int(self.headers["Content-Length"])
        except (KeyError, TypeError, ValueError):
            self.send_json(411, {"error": "올바른 Content-Length가 필요합니다."})
            return

        if content_length <= 0 or content_length > MAX_BODY_SIZE:
            self.send_json(413, {"error": "요청 데이터 크기가 허용 범위를 벗어났습니다."})
            return

        try:
            fields = parse_multipart(content_type, self.rfile.read(content_length))
            image_data = fields["frame"]
            ai_result = json.loads(fields["result"].decode("utf-8"))
            frame_id = ai_result["frame_id"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "올바른 frame과 result가 필요합니다."})
            return

        process_received_data(frame_id, image_data, ai_result)
        self.send_json(200, {"status": "received", "frame_id": frame_id})


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), FrameRequestHandler)
    print(f"HTTP receiver listening on {HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
