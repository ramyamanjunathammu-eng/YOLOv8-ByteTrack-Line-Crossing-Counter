import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

st.set_page_config(page_title="YOLOv8 Line Crossing Counter", layout="wide")
st.title("YOLOv8 + ByteTrack Line Crossing Counter")

# ============================================================
# VIDEO SOURCE - CHANGE THIS LINE TO USE YOUR OWN VIDEO
# ============================================================
# Option 1: Use webcam (default camera)
# VIDEO_SOURCE = 0

# Option 2: Video file in the same folder as this script
VIDEO_SOURCE = "ramya.mp4"

# Option 3: Full path example
# VIDEO_SOURCE = r"C:\Users\Ramya\Videos\my_video.mp4"
# ============================================================

CONF_THRESHOLD = 0.3
LINE_POSITION_RATIO = 2 / 3  # horizontal line at 2/3 of frame height

# Initialize YOLOv8 model pre-trained on COCO (detects persons, cars, bikes, buses, trucks)
model = YOLO("yolov8n.pt")  # Use yolov8n (nano) for speed, can change to yolov8s/l/x

# NOTE: Standard COCO-trained YOLOv8 has NO dedicated "auto-rickshaw" class.
# Autos are usually picked up under "motorcycle" (id 3) by this model, so we
# relabel that class as "auto/bike" for display purposes. For a true, accurate
# auto-rickshaw class you'd need a custom-trained model on auto-labeled data.
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "auto/bike",   # was "motorcycle" - relabeled since autos land here too
    5: "bus",
    7: "truck",
}

PERSON_CLASS_IDS = {0}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
AUTO_BIKE_CLASS_IDS = {3}  # separate bucket just for auto/bike counting

# Colors in BGR (since we draw before converting to RGB)
PERSON_COLOR = (0, 200, 0)     # green
VEHICLE_COLOR = (0, 0, 255)    # red
AUTO_BIKE_COLOR = (255, 140, 0)  # orange-blue, to visually separate from cars/buses
LINE_COLOR = (0, 255, 255)     # yellow

# ---------------- Sidebar: live counters ----------------
st.sidebar.header("Live Counts")
person_in_ph = st.sidebar.empty()
person_out_ph = st.sidebar.empty()
vehicle_in_ph = st.sidebar.empty()
vehicle_out_ph = st.sidebar.empty()
auto_in_ph = st.sidebar.empty()
auto_out_ph = st.sidebar.empty()

def update_sidebar_counts(p_in, p_out, v_in, v_out, a_in, a_out):
    person_in_ph.metric("Person IN", p_in)
    person_out_ph.metric("Person OUT", p_out)
    vehicle_in_ph.metric("Vehicle IN", v_in)
    vehicle_out_ph.metric("Vehicle OUT", v_out)
    auto_in_ph.metric("Auto/Bike IN", a_in)
    auto_out_ph.metric("Auto/Bike OUT", a_out)

# Counters
person_in_count = 0
person_out_count = 0
vehicle_in_count = 0
vehicle_out_count = 0
auto_in_count = 0
auto_out_count = 0

crossed_ids_in = set()
crossed_ids_out = set()
object_centers = dict()  # id -> deque of recent center-y positions (maxlen=2)

update_sidebar_counts(person_in_count, person_out_count, vehicle_in_count, vehicle_out_count,
                       auto_in_count, auto_out_count)

# ---------------- Main video area ----------------
stframe = st.empty()
status_ph = st.empty()

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    st.error(f"Could not open video source: {VIDEO_SOURCE}. Check the path/webcam index.")
    st.stop()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        status_ph.info("Video ended or no more frames.")
        break

    frame_count += 1
    height, width = frame.shape[:2]
    line_y = int(height * LINE_POSITION_RATIO)

    # Scale drawing sizes relative to frame resolution so text/boxes are visible
    # on both small webcam frames and large 1080p+ video files
    scale = max(width, height) / 1000
    box_thickness = max(2, int(2 * scale))
    font_scale = max(0.6, 0.6 * scale)
    text_thickness = max(1, int(2 * scale))

    # YOLOv8 inference + ByteTrack tracking (built into ultralytics)
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    # Draw the counting line
    cv2.line(frame, (0, line_y), (width, line_y), LINE_COLOR, box_thickness)

    current_frame_ids = set()
    detections_this_frame = 0

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)

        for bbox, class_id, conf, obj_id in zip(boxes, classes, confs, ids):
            if class_id not in COCO_CLASSES:
                continue

            detections_this_frame += 1
            x1, y1, x2, y2 = bbox
            center_y = int((y1 + y2) / 2)
            current_frame_ids.add(obj_id)

            label_name = COCO_CLASSES[class_id]
            is_person = class_id in PERSON_CLASS_IDS
            is_auto_bike = class_id in AUTO_BIKE_CLASS_IDS

            if is_person:
                color = PERSON_COLOR
            elif is_auto_bike:
                color = AUTO_BIKE_COLOR
            else:
                color = VEHICLE_COLOR

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Label with background so it's always readable
            label_text = f"{label_name} #{obj_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 10, label_y), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)

            # Track center-y history
            if obj_id not in object_centers:
                object_centers[obj_id] = deque(maxlen=2)
            object_centers[obj_id].append(center_y)

            if len(object_centers[obj_id]) == 2:
                prev_y, curr_y = object_centers[obj_id][0], object_centers[obj_id][1]

                if is_person:
                    if prev_y < line_y <= curr_y and obj_id not in crossed_ids_out:
                        person_out_count += 1
                        crossed_ids_out.add(obj_id)
                    elif prev_y >= line_y > curr_y and obj_id not in crossed_ids_in:
                        person_in_count += 1
                        crossed_ids_in.add(obj_id)
                elif is_auto_bike:
                    if prev_y < line_y <= curr_y and obj_id not in crossed_ids_out:
                        auto_out_count += 1
                        crossed_ids_out.add(obj_id)
                    elif prev_y >= line_y > curr_y and obj_id not in crossed_ids_in:
                        auto_in_count += 1
                        crossed_ids_in.add(obj_id)
                else:
                    if prev_y < line_y <= curr_y and obj_id not in crossed_ids_out:
                        vehicle_out_count += 1
                        crossed_ids_out.add(obj_id)
                    elif prev_y >= line_y > curr_y and obj_id not in crossed_ids_in:
                        vehicle_in_count += 1
                        crossed_ids_in.add(obj_id)

    # Drop IDs that disappeared from the frame
    lost_ids = set(object_centers.keys()) - current_frame_ids
    for lost_id in lost_ids:
        object_centers.pop(lost_id, None)
        crossed_ids_in.discard(lost_id)
        crossed_ids_out.discard(lost_id)

    # Overlay counts on the video itself too
    info_lines = [
        f"Person IN: {person_in_count}  OUT: {person_out_count}",
        f"Vehicle IN: {vehicle_in_count}  OUT: {vehicle_out_count}",
        f"Auto/Bike IN: {auto_in_count}  OUT: {auto_out_count}",
    ]
    for i, text in enumerate(info_lines):
        y = 30 + i * int(35 * scale)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), text_thickness + 2)  # outline
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), text_thickness)

    update_sidebar_counts(person_in_count, person_out_count, vehicle_in_count, vehicle_out_count,
                           auto_in_count, auto_out_count)
    status_ph.caption(f"Frame {frame_count} | Objects tracked this frame: {detections_this_frame}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame_rgb, channels="RGB", width='stretch')

cap.release()