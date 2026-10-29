"""YOLO adapter — implements DetectorPort using ultralytics."""

import cv2
import numpy as np
from ultralytics import YOLO

from domain.entities import EMPTY_DETECTION, Detection
from domain.geometry import extract_quad_from_mask


class YoloDetector:
    def __init__(
        self,
        model_path,
        confidence,
        inference_size,
        max_detections,
        device,
        min_mask_area,
    ):
        self.confidence = confidence
        self.inference_size = inference_size
        self.max_detections = max_detections
        self.device = device
        self.min_mask_area = min_mask_area

        self.model = YOLO(model_path)
        self._warmup()

    def _warmup(self):
        try:
            self.model.predict(
                np.zeros((480, 640, 3), dtype=np.uint8),
                imgsz=self.inference_size,
                device=self.device,
                half=False,
                conf=self.confidence,
                max_det=self.max_detections,
                verbose=False,
            )
            print(
                f"[Detector] Warm-up OK "
                f"(imgsz={self.inference_size}, device={self.device})"
            )
        except Exception as e:
            if self.device == "cuda":
                print(f"[Detector] CUDA warm-up FAILED: {e}")
                raise
            print(f"[Detector] Warm-up skipped: {e}")

    def detect(self, frame) -> Detection:
        results = self.model.predict(
            source=frame,
            imgsz=self.inference_size,
            device=self.device,
            half=(self.device == "cuda"),
            conf=self.confidence,
            max_det=self.max_detections,
            verbose=False,
        )

        best_box = None
        best_area = 0
        best_conf = 0.0
        best_mask_polygon = None

        for result in results:
            if result.boxes is None:
                continue
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = int((x2 - x1) * (y2 - y1))
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
                    best_conf = conf
                    best_mask_polygon = None
                    if (
                        result.masks is not None
                        and result.masks.xy is not None
                        and i < len(result.masks.xy)
                    ):
                        best_mask_polygon = result.masks.xy[i]

        if best_box is None:
            return EMPTY_DETECTION

        x1, y1, x2, y2 = best_box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        corners = extract_quad_from_mask(best_mask_polygon, self.min_mask_area)
        return Detection(
            best_box, best_area, best_conf, center, corners, best_mask_polygon
        )

    @staticmethod
    def annotate(frame, detection: Detection):
        if detection.box is None:
            return
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            frame,
            f"conf={detection.confidence:.2f} area={detection.area}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
        )
        if detection.corners is not None:
            cv2.polylines(
                frame,
                [np.asarray(detection.corners, dtype=np.int32)],
                True,
                (255, 0, 255),
                2,
            )