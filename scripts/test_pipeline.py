import time

import numpy as np

from src.engine.detector import YOLODetector
from src.tracking.tracker import ObjectTracker


def test_pipeline() -> None:
    print("[INFO] Initializing Detector and Tracker...")
    detector = YOLODetector()
    tracker = ObjectTracker()

    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("[INFO] Warming up...")
    for _ in range(5):
        dets = detector.infer(dummy_frame)
        _ = tracker.update(dets)

    frames_count = 60
    start_time = time.perf_counter()

    for _ in range(frames_count):
        detections = detector.infer(dummy_frame)
        _ = tracker.update(detections)

    total_time = time.perf_counter() - start_time
    avg_latency = (total_time / frames_count) * 1000
    fps = frames_count / total_time

    print("=" * 40)
    print("End-to-End Pipeline (Detect + Track):")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("=" * 40)


if __name__ == "__main__":
    test_pipeline()
