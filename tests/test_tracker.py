from src.tracking.tracker import ObjectTracker


def test_tracker_empty_input() -> None:
    tracker = ObjectTracker()
    results = tracker.update([])
    assert results == []


def test_tracker_tracks_assignment() -> None:
    tracker = ObjectTracker()
    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "score": 0.9,
            "class_id": 0,
        }
    ]
    # 更新第一幀建立 track
    tracked = tracker.update(detections)
    assert isinstance(tracked, list)
