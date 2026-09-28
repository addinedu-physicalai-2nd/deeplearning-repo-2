from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from inference.lane import load_lane_model, run_lane
from inference.obstacle import load_obstacle_model, run_obstacle
from inference.pothole import load_pothole_model, run_pothole

executor = ThreadPoolExecutor(max_workers=3)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["lane"] = load_lane_model()
    models["obstacle"] = load_obstacle_model()
    models["pothole"] = load_pothole_model()

    yield

    models.clear()

app = FastAPI(lifespan=lifespan)


@app.post("/inference")
def inference(file: UploadFile, frame_id: int):
    raw = file.file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    fut_lane = executor.submit(run_lane, models["lane"], image)
    fut_obstacle = executor.submit(run_obstacle, models["obstacle"], image)
    fut_pothole = executor.submit(run_pothole, models["pothole"], image)

    detections = {}
    detections.update(fut_lane.result())
    detections.update(fut_obstacle.result())
    detections.update(fut_pothole.result())

    return {
        "frame_id": frame_id,
        "detections": detections
    }