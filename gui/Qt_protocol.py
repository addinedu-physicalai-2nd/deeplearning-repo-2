"""flask_test.py에서 이미지와 AI 결과 JSON을 받는 HTTP 서버."""

import json

from flask import Flask, jsonify, request


app = Flask(__name__)


def process_received_data(frame_id, image_data, ai_result):
    """수신 데이터를 사용하는 코드를 이 함수에 구현한다."""
    print(
        f"frame_id={frame_id}, image_bytes={len(image_data)}, "
        f"json={json.dumps(ai_result, ensure_ascii=False)}",
        flush=True,
    )


@app.post("/frame")
def receive_frame():
    frame_file = request.files.get("frame")
    result_file = request.files.get("result")

    if frame_file is None or result_file is None:
        return jsonify(error="frame과 result가 필요합니다."), 400

    image_data = frame_file.read()

    try:
        ai_result = json.load(result_file)
        frame_id = ai_result["frame_id"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return jsonify(error="올바른 frame_id를 포함한 JSON이 필요합니다."), 400

    process_received_data(frame_id, image_data, ai_result)
    return jsonify(status="received", frame_id=frame_id), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=False)
