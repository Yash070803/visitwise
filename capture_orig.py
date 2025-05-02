# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import requests
import time
from threading import Thread, Lock, Event

# --- Configuration ---
MODEL_PATH        = "models/model.pt"
OUTPUT_DIR        = "captured_faces"
STREAM_URL        = "http://192.168.31.116:5000/video"
MAX_DISTANCE_CM   = 150.0    # only capture if under 1 meter

# quality thresholds
AREA_THRESHOLD       = 5000     # px²
SHARPNESS_THRESHOLD  = 100.0    # Laplacian variance threshold
YAW_THRESHOLD        = 10       # degrees
PITCH_THRESHOLD      = 10       # degrees

# calibration coefficients
a = 9703.20
b = -0.4911842338691967

# head-pose model points
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-165.0,170.0, -135.0),
    (165.0, 170.0, -135.0),
    (-150.0,-150.0,-125.0),
    (150.0, -150.0,-125.0)
], dtype=np.float64)

# landmark indices
LANDMARK_IDS = {"nose_tip":1, "chin":199, "left_eye":33, "right_eye":263, "left_mouth":61, "right_mouth":291}

# load detectors
model = YOLO(MODEL_PATH)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)

# ensure output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# shared frame buffer
frame_buffer = None
buffer_lock = Lock()
stop_event = Event()

# reader thread: continuously fetch and rotate frames
def frame_reader():
    global frame_buffer
    cap = cv2.VideoCapture(STREAM_URL)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        with buffer_lock:
            frame_buffer = frame
    cap.release()

# start reader thread
t = Thread(target=frame_reader, daemon=True)
t.start()

# helper: pose estimation
def estimate_head_pose(lm, img_size):
    h, w = img_size
    pts = np.array([(lm[i][0]*w, lm[i][1]*h) for i in [LANDMARK_IDS['nose_tip'], LANDMARK_IDS['chin'], LANDMARK_IDS['left_eye'], LANDMARK_IDS['right_eye'], LANDMARK_IDS['left_mouth'], LANDMARK_IDS['right_mouth']]], dtype=np.float64)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, None
    R,_ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    return yaw, pitch

# helper: sharpness
def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# asynchronous upload helper
def async_upload(path):
    try:
        res = requests.post(
            'https://meghavi-kiosk-api.onrender.com/api/faces/upload',
            files={'image': open(path, 'rb')},
            data={'deviceId': 'DEV3617'}
        )
        if res.status_code == 201:
            print("Upload successful.")
        else:
            print(f"Upload failed with status: {res.status_code}")
    except Exception as e:
        print(f"Upload exception: {e}")

# main loop
cv2.namedWindow("Video Preview")
captured = False
print("Starting preview. Press 'q' to quit.")

while True:
    with buffer_lock:
        frame = frame_buffer.copy() if frame_buffer is not None else None
    if frame is None:
        time.sleep(0.01)
        continue

    cv2.imshow("Video Preview", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if captured:
        continue

    res = model(frame, conf=0.4, verbose=False)
    boxes = res[0].boxes
    if not boxes:
        continue

    for box in boxes:
        conf = float(box.conf[0])
        if conf < 0.4:
            print("Box detected but confidence too low.")
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        area = (x2-x1)*(y2-y1)
        if area < AREA_THRESHOLD:
            print(f"Face detected but too small: area={area}")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = face_mesh.process(rgb)
        if not mesh.multi_face_landmarks:
            print("Box detected but landmarks missing.")
            continue
        lm = [(p.x,p.y) for p in mesh.multi_face_landmarks[0].landmark]

        dist = a * (area ** b)
        if dist > MAX_DISTANCE_CM:
            print(f"Face detected but too far: {dist:.1f}cm")
            continue

        yaw,pitch = estimate_head_pose(lm, frame.shape[:2])
        if yaw is None or abs(yaw)>YAW_THRESHOLD or abs(pitch)>PITCH_THRESHOLD:
            print(f"Pose check fail: yaw={yaw}, pitch={pitch}")
            continue

        roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        var = laplacian_var(roi_gray)
        if var < SHARPNESS_THRESHOLD:
            print(f"Face detected but not sharp enough: var={var:.1f}")
            continue

        h, _ = frame.shape[:2]
        le_y = lm[LANDMARK_IDS['left_eye']][1]*h
        re_y = lm[LANDMARK_IDS['right_eye']][1]*h
        if abs(le_y - re_y) > 0.03*h:
            print("Face detected but not front-on (eye symmetry fail).")
            continue

        path = os.path.join(OUTPUT_DIR, 'face.jpg')
        cv2.imwrite(path, frame)
        print(f"Captured at {dist:.1f}cm, uploading...")
        Thread(target=async_upload, args=(path,), daemon=True).start()
        captured = True
        break

# cleanup
stop_event.set()
t.join()
cv2.destroyAllWindows()
