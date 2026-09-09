import base64
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.api.schemas import InferenceResponse, TrackedObject
from src.engine.detector import YOLODetector
from src.tracking.tracker import ObjectTracker

pipeline: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    pipeline["detector"] = YOLODetector()
    pipeline["tracker"] = ObjectTracker()
    yield
    pipeline.clear()


app = FastAPI(title="Real-time Vision Inference & Tracking API", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "provider": pipeline["detector"].active_provider,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_counter = 0

    detector: YOLODetector = pipeline["detector"]
    tracker: ObjectTracker = pipeline["tracker"]

    try:
        while True:
            payload = await websocket.receive_text()
            start_time = time.perf_counter()

            img_bytes = base64.b64decode(payload)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            detections = detector.infer(frame)
            tracked_results = tracker.update(detections)

            latency_ms = (time.perf_counter() - start_time) * 1000
            frame_counter += 1

            response = InferenceResponse(
                frame_id=frame_counter,
                latency_ms=round(latency_ms, 2),
                tracked_objects=[TrackedObject(**obj) for obj in tracked_results],
            )
            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        print("[INFO] Client disconnected from /ws/stream")
