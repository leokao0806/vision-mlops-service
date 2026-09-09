import numpy as np

from src.engine.detector import YOLODetector


def test_detector_inference_shape() -> None:
    detector = YOLODetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.infer(dummy_frame)
    assert isinstance(results, list)
