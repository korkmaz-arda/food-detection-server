import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import io
from PIL import Image

from utils.detect import (
    detect_trays, detect_food,
    track_lucas_kanade
)
from utils.bbox import bbox2poly

FOOD_DECAY = 1
TRAY_DECAY = 1

class DetectionManager:
    def __init__(
        self,
        tray_model_path: str = "models/tray_detector.pt",
        food_model_path: str = "models/yolo11n.pt",
        device: str = None,
        skip_frames: int = 8,
        tray_conf: float = 0.50,
        food_conf: float = 0.30,
        intersection_thresh: float = 0.5,
        food_classes: List[str] = None,
        tray_shrink_factor: float = 0.7
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing DetectionManager with device: {self.device}")

        self.tray_model = YOLO(tray_model_path).to(self.device)
        self.food_model = YOLO(food_model_path).to(self.device)

        self.skip_frames = skip_frames
        self.tray_conf = tray_conf
        self.food_conf = food_conf
        self.intersection_thresh = intersection_thresh
        self.tray_shrink_factor = tray_shrink_factor

        if food_classes:
            self.food_classes = food_classes
        else:
            self.food_classes = [
                "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake"
            ]

        print(f"Using food classes: {self.food_classes}")

        self.frame_count = 0
        self.prev_gray = None
        self.prev_trays = []
        self.prev_foods = []
        self.last_detection_time = 0
        self.detection_fps = 0

    def process_frame(self, frame_data: bytes) -> Dict:
        start_time = time.time()

        image = Image.open(io.BytesIO(frame_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        should_detect = (self.frame_count % self.skip_frames == 0) or (self.prev_gray is None)

        if should_detect:
            trays = detect_trays(frame, self.tray_model, self.tray_conf, self.device)
            foods = detect_food(
                frame, self.food_model, self.food_conf,
                self.food_classes, self.device
            )
            self.prev_trays = trays
            self.prev_foods = foods
            detection_method = "yolo"

        else:
            trays, foods = self._track_objects(frame_gray)
            detection_method = "optical_flow"

        self.prev_gray = frame_gray.copy()
        self.frame_count += 1

        process_time = time.time() - start_time
        if self.last_detection_time > 0:
            self.detection_fps = 1.0 / (time.time() - self.last_detection_time)
        self.last_detection_time = time.time()

        result = {
            "frame_number": self.frame_count,
            "method": detection_method,
            "trays": self._format_trays(trays),
            "foods": self._format_foods(foods, trays),
            "metrics": {
                "process_time_ms": process_time * 1000,
                "detection_fps": self.detection_fps,
                "device": self.device
            }
        }

        return result

    def _track_objects(self, frame_gray: np.ndarray) -> Tuple[List, List]:
        tray_boxes = [xyxy for _, xyxy, _ in self.prev_trays]
        tracked_tray_boxes = track_lucas_kanade(self.prev_gray, frame_gray, tray_boxes)
        tray_matches = self._match_boxes(tray_boxes, tracked_tray_boxes)

        new_trays = []
        for prev_idx, tracked_idx in tray_matches:
            _, _, conf = self.prev_trays[prev_idx]
            tracked_box = tracked_tray_boxes[tracked_idx]
            poly = bbox2poly(tracked_box)
            new_trays.append((poly, tracked_box, conf))

        matched_prev = [p for p, _ in tray_matches]
        for i in range(len(self.prev_trays)):
            if i not in matched_prev:
                poly, box, conf = self.prev_trays[i]
                new_conf = max(self.tray_conf, conf * TRAY_DECAY)
                new_trays.append((poly, box, new_conf))

        food_boxes = [xyxy for xyxy, _, _ in self.prev_foods]
        tracked_food_boxes = track_lucas_kanade(self.prev_gray, frame_gray, food_boxes)
        food_matches = self._match_boxes(food_boxes, tracked_food_boxes, threshold=0.2)

        new_foods = []
        for prev_idx, tracked_idx in food_matches:
            tracked_box = tracked_food_boxes[tracked_idx]
            _, conf, cls = self.prev_foods[prev_idx]
            new_foods.append((tracked_box, conf, cls))

        matched_prev_food = [p for p, _ in food_matches]
        for i in range(len(self.prev_foods)):
            if i not in matched_prev_food:
                box, conf, cls = self.prev_foods[i]
                new_conf = max(0.5, conf * FOOD_DECAY)
                new_foods.append((box, new_conf, cls))

        self.prev_trays = new_trays
        self.prev_foods = new_foods

        return new_trays, new_foods

    def _match_boxes(self, prev_boxes: List, tracked_boxes: List, threshold: float = 0.3) -> List[Tuple[int, int]]:
        if not prev_boxes or not tracked_boxes:
            return []

        matches = []
        used_tracked = set()

        for i, prev_box in enumerate(prev_boxes):
            best_match = -1
            best_iou = threshold

            for j, tracked_box in enumerate(tracked_boxes):
                if j in used_tracked:
                    continue

                iou = self._calculate_iou(prev_box, tracked_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = j

            if best_match != -1:
                matches.append((i, best_match))
                used_tracked.add(best_match)

        return matches

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def _format_trays(self, trays: List) -> List[Dict]:
        formatted = []
        for i, (poly, xyxy, conf) in enumerate(trays):
            shrunken_box = self._shrink_box(xyxy)
            x1, y1, x2, y2 = [float(v) for v in shrunken_box]
            formatted.append({
                "id": f"tray_{i}",
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class": "tray"
            })
        return formatted

    def _shrink_box(self, box: List[float]) -> List[float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        new_w = w * self.tray_shrink_factor
        new_h = h * self.tray_shrink_factor

        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2

        return [new_x1, new_y1, new_x2, new_y2]

    def _format_foods(self, foods: List, trays: List) -> List[Dict]:
        formatted = []
        for i, (xyxy, conf, class_name) in enumerate(foods):
            food_poly = bbox2poly(xyxy)
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            in_tray = False
            tray_id = None

            for j, (tray_poly, tray_xyxy, _) in enumerate(trays):
                shrunken_box = self._shrink_box(tray_xyxy)
                shrunken_poly = bbox2poly(shrunken_box)

                if shrunken_poly.is_valid and food_poly.is_valid:
                    intersection_area = shrunken_poly.intersection(food_poly).area
                    food_area = food_poly.area
                    if food_area > 0 and (intersection_area / food_area) >= self.intersection_thresh:
                        in_tray = True
                        tray_id = f"tray_{j}"
                        break

            formatted.append({
                "id": f"food_{i}",
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class": class_name,
                "in_tray": in_tray,
                "tray_id": tray_id
            })
        return formatted

    def reset(self):
        self.frame_count = 0
        self.prev_gray = None
        self.prev_trays = []
        self.prev_foods = []
        self.last_detection_time = 0
        self.detection_fps = 0

