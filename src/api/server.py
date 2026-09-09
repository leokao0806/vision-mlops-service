import base64
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision MLOps Live Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        .container { position: relative; width: 640px; height: 480px; margin: 20px auto; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        video, canvas { position: absolute; top: 0; left: 0; width: 640px; height: 480px; }
        #metrics { font-size: 1.1rem; margin-top: 10px; color: #38bdf8; font-weight: bold; }
        button { background: #2563eb; color: white; border: none; padding: 10px 20px; font-size: 1rem; border-radius: 6px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
    </style>
</head>
<body>
    <h1>Real-time Vision Inference & Tracking</h1>
    <button id="startBtn" onclick="startStream()">Start Webcam Stream</button>
    <div class="container">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay" width="640" height="480"></canvas>
    </div>
    <div id="metrics">Status: Ready</div>

    <script>
        let ws;
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const metrics = document.getElementById('metrics');
        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 640;
        sendCanvas.height = 480;
        const sendCtx = sendCanvas.getContext('2d');

        async function startStream() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                document.getElementById('startBtn').style.display = 'none';

                ws = new WebSocket(`ws://${location.host}/ws/stream`);

                ws.onopen = () => {
                    metrics.innerText = "Status: Connected | Processing...";
                    sendFrame();
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    metrics.innerText = `Latency: ${data.latency_ms} ms | FPS: ${(1000/data.latency_ms).toFixed(1)} | Tracked Objects: ${data.tracked_objects.length}`;
                    drawDetections(data.tracked_objects);
                    requestAnimationFrame(sendFrame);
                };

                ws.onclose = () => { metrics.innerText = "Status: Disconnected"; };
            } catch (err) {
                alert("Camera access denied or unavailable: " + err.message);
            }
        }

        function sendFrame() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                sendCtx.drawImage(video, 0, 0, 640, 480);
                const base64 = sendCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                ws.send(base64);
            }
        }

        function drawDetections(objects) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            objects.forEach(obj => {
                const [x1, y1, x2, y2] = obj.bbox;
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.fillStyle = '#22c55e';
                ctx.font = '16px monospace';
                ctx.fillText(`ID: ${obj.track_id} (${(obj.score * 100).toFixed(0)}%)`, x1 + 4, y1 > 20 ? y1 - 6 : y1 + 18);
            });
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_CONTENT


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
