from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    def __init__(
        self,
        model_path: str | Path = "models/yolov8n.onnx",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        providers = ort.get_available_providers()
        active_providers = (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(str(model_path), providers=active_providers)
        self.active_provider = self.session.get_providers()[0]

        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [1, 3, 640, 640]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        img_h, img_w = image.shape[:2]
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # BCHW
        return blob, scale, 0.0

    def postprocess(self, output: np.ndarray, scale: float) -> list[dict[str, Any]]:
        # output shape: [1, 84, 8400] -> transpose to [8400, 84]
        predictions = np.squeeze(output[0]).T

        boxes = []
        scores = []
        class_ids = []

        for row in predictions:
            classes_scores = row[4:]
            class_id = int(np.argmax(classes_scores))
            score = float(classes_scores[class_id])

            if score >= self.conf_threshold:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                x1 = int((xc - w / 2) / scale)
                y1 = int((yc - h / 2) / scale)
                box_w = int(w / scale)
                box_h = int(h / scale)

                boxes.append([x1, y1, box_w, box_h])
                scores.append(score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )

        results: list[dict[str, Any]] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                results.append(
                    {
                        "bbox": [x, y, x + w, y + h],
                        "score": float(scores[idx]),
                        "class_id": int(class_ids[idx]),
                    }
                )
        return results

    def infer(self, image: np.ndarray) -> list[dict[str, Any]]:
        blob, scale, _ = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs[0], scale)
