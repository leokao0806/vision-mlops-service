from pydantic import BaseModel, Field


class TrackedObject(BaseModel):
    track_id: int = Field(..., description="Unique tracking identifier")
    bbox: list[int] = Field(..., description="Bounding box in [x1, y1, x2, y2] format")
    score: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="COCO class identifier")


class InferenceResponse(BaseModel):
    frame_id: int = Field(..., description="Frame sequence counter")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    tracked_objects: list[TrackedObject] = Field(
        default_factory=list, description="List of tracked entities"
    )
