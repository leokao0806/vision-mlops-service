from typing import Any

import numpy as np
import supervision as sv


class ObjectTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        tracker_cls = getattr(sv, "ByteTrackTracker", sv.ByteTrack)
        self.tracker = tracker_cls(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections_list:
            empty_detections = sv.Detections.empty()
            _ = self.tracker.update_with_detections(empty_detections)
            return []

        xyxy = np.array([d["bbox"] for d in detections_list], dtype=np.float32)
        confidence = np.array([d["score"] for d in detections_list], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in detections_list], dtype=np.int32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        tracked_results: list[dict[str, Any]] = []
        if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i].astype(int).tolist()
                tracked_results.append(
                    {
                        "track_id": int(tracked_detections.tracker_id[i]),
                        "bbox": box,
                        "score": float(
                            tracked_detections.confidence[i]
                            if tracked_detections.confidence is not None
                            else 0.0
                        ),
                        "class_id": int(
                            tracked_detections.class_id[i]
                            if tracked_detections.class_id is not None
                            else -1
                        ),
                    }
                )

        return tracked_results
