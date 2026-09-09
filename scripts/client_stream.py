import asyncio
import base64
import json

import cv2
import numpy as np
import websockets


async def stream_client() -> None:
    uri = "ws://127.0.0.1:8000/ws/stream"
    print(f"[INFO] Connecting to {uri}...")

    # 嘗試開啟攝影機，若無攝影機則使用合成畫面
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    if use_webcam:
        print("[INFO] Opened webcam.")
    else:
        print("[INFO] No webcam found, using synthetic moving frames.")

    try:
        async with websockets.connect(uri) as websocket:
            for frame_idx in range(100):
                if use_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    # 生成一個動態移動方塊的假畫面
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    x = int((frame_idx * 5) % 500)
                    cv2.rectangle(frame, (x, 100), (x + 80, 200), (0, 255, 0), -1)

                # 編碼為 JPEG 再轉為 Base64
                _, buffer = cv2.imencode(".jpg", frame)
                payload = base64.b64encode(buffer).decode("utf-8")

                await websocket.send(payload)
                response_raw = await websocket.recv()
                data = json.loads(response_raw)

                print(
                    f"Frame {data['frame_id']:03d} | "
                    f"Server Latency: {data['latency_ms']:5.2f} ms | "
                    f"Objects Tracked: {len(data['tracked_objects'])}"
                )
                await asyncio.sleep(0.01)

    finally:
        if use_webcam:
            cap.release()
        print("[INFO] Stream finished.")


if __name__ == "__main__":
    asyncio.run(stream_client())
