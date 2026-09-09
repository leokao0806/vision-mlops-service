import time

import numpy as np

from src.engine.detector import YOLODetector


def run_benchmark() -> None:
    detector = YOLODetector()
    print(f"[INFO] Initialized with Provider: {detector.active_provider}")

    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("[INFO] Warming up...")
    for _ in range(5):
        _ = detector.infer(dummy_frame)

    iterations = 50
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = detector.infer(dummy_frame)
    total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / iterations) * 1000
    fps = iterations / total_time

    print("=" * 40)
    print(f"Provider: {detector.active_provider}")
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("=" * 40)


if __name__ == "__main__":
    run_benchmark()
