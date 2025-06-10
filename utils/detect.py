import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

from utils.bbox import bbox2poly


def detect_trays(frame, tray_model, tray_min_conf, device):
    tray_results = tray_model.predict(frame, device=device, verbose=False)
    # tray_results = tray_model.predict(frame, device='cpu', verbose=False)
    tray_polygons = []
    for result in tray_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < tray_min_conf:
                continue
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            tray_polygon = bbox2poly(xyxy)
            tray_polygons.append((tray_polygon, xyxy, conf))
    return tray_polygons


def draw_trays(frame, tray_polygons):
    for tray_polygon, xyxy, conf in tray_polygons:
        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        label = f"Tray {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def detect_food(frame, food_model, food_min_conf, class_filter, device):
    food_results = food_model.predict(frame, device=device, verbose=False)
    # food_results = food_model.predict(frame, device='cpu', verbose=False)
    food_detections = []
    for result in food_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            conf = float(box.conf[0].cpu().numpy())
            if conf < food_min_conf:
                continue
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = result.names.get(cls_id, "unknown")
            if class_filter and class_name not in class_filter:
                continue
            food_detections.append((xyxy, conf, class_name))
    return food_detections


def draw_food(frame, food_detections, tray_polygons, intersection_threshold):
    for xyxy, conf, class_name in food_detections:
        food_polygon = bbox2poly(xyxy)
        for tray_polygon, _, _ in tray_polygons:
            if tray_polygon.is_valid and food_polygon.is_valid:
                intersection_area = tray_polygon.intersection(food_polygon).area
                food_area = food_polygon.area
                if food_area > 0 and (intersection_area / food_area) >= intersection_threshold:
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  color=(0, 255, 0), thickness=2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break  # annotate only once per food detection if it matches any tray.
    return frame


def track_lucas_kanade(prev_frame_gray, curr_frame_gray, prev_boxes):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    tracked_boxes = []

    for box in prev_boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        roi_prev = prev_frame_gray[y1:y2, x1:x2]
        if roi_prev.size == 0:
            continue
        p0 = cv2.goodFeaturesToTrack(roi_prev, mask=None, maxCorners=10, qualityLevel=0.3, minDistance=7)
        if p0 is None:
            continue

        # offset p0 to absolute frame coordinates
        p0 += np.array([[x1, y1]])

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, p0, None, **lk_params)
        if p1 is None or st.sum() < 3:
            continue

        movement = p1 - p0
        dx, dy = np.mean(movement, axis=0).flatten()

        # shift box
        new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        tracked_boxes.append(new_box)

    return tracked_boxes


def video_detect(
        tray_model, 
        food_model, 
        input_path, 
        output_path, 
        only_tray=False,
        class_filter=None, 
        tray_min_conf=0.72, 
        food_min_conf=0.3, 
        intersection_threshold=0.5
    ):

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # tray detection
        tray_polygons = detect_trays(frame, tray_model, tray_min_conf)
        frame = draw_trays(frame, tray_polygons)

        if not only_tray:
            # food detection
            food_detections = detect_food(frame, food_model, food_min_conf, class_filter)
            frame = draw_food(frame, food_detections, tray_polygons, intersection_threshold)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")


def video_detect_lucas_kanade(
        tray_model, 
        food_model, 
        input_path, 
        output_path, 
        only_tray=False,
        class_filter=None, 
        tray_min_conf=0.72, 
        food_min_conf=0.3, 
        intersection_threshold=0.5,
        detect_every_n_frames=5
    ):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    prev_gray = None
    prev_tray_boxes = []
    prev_food_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % detect_every_n_frames == 0:
            tray_polygons = detect_trays(frame, tray_model, tray_min_conf)
            tray_boxes = [xyxy for _, xyxy, _ in tray_polygons]
            frame = draw_trays(frame, tray_polygons)
            prev_tray_boxes = tray_boxes

            if not only_tray:
                food_detections = detect_food(frame, food_model, food_min_conf, class_filter)
                frame = draw_food(frame, food_detections, tray_polygons, intersection_threshold)
                prev_food_detections = food_detections
        else:
            if prev_gray is not None:
                tracked_tray_boxes = track_lucas_kanade(prev_gray, frame_gray, prev_tray_boxes)
                tray_polygons = [(bbox2poly(box), box, 0.75) for box in tracked_tray_boxes]
                frame = draw_trays(frame, tray_polygons)
                prev_tray_boxes = tracked_tray_boxes

                if not only_tray:
                    prev_boxes = [xyxy for xyxy, _, _ in prev_food_detections]
                    tracked_boxes = track_lucas_kanade(prev_gray, frame_gray, prev_boxes)
                    food_detections = [
                        (tracked_box, conf, class_name)
                        for tracked_box, (_, conf, class_name) in zip(tracked_boxes, prev_food_detections)
                    ]
                    frame = draw_food(frame, food_detections, tray_polygons, intersection_threshold)
                    prev_food_detections = food_detections

        prev_gray = frame_gray.copy()
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")