#--------------cv2--------------------------------------
'''# -*- coding: utf-8 -*-
import os
import cv2
import time
from threading import Thread, Lock, Event

# --- Configuration ---
OUTPUT_DIR      = "captured_faces"
CASCADE_DIR     = os.path.join(os.path.dirname(__file__), "cascades")
CAMERA_INDEX    = "http://packproof:5000/video"                # 0 for default camera
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
AREA_THRESHOLD  = 20000            # px², adjust for minimum face size
MIN_DISTANCE_CM = 150.0            # max distance to capture

a = 9703.20     # calibration coefficient (distance = a * area^b)
b = -0.4911842338691967

# Cascade search paths
SYSTEM_PATHS = [
    "/usr/share/opencv4/haarcascades",
    "/usr/share/opencv/haarcascades",
    "/usr/local/share/opencv4/haarcascades",
]

def find_cascade(name):
    # check local cascades folder
    local = os.path.join(CASCADE_DIR, name)
    if os.path.isfile(local):
        return local
    # check system folders
    for base in SYSTEM_PATHS:
        full = os.path.join(base, name)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(f"Cascade '{name}' not found in ./cascades or system paths.")

# Load cascades
face_xml = find_cascade("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(face_xml)

# Shared buffer for frames
default = None
frame_buffer = default
buffer_lock  = Lock()
stop_event   = Event()

# Capture thread using OpenCV
def frame_reader():
    global frame_buffer
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with buffer_lock:
            frame_buffer = frame.copy()
        time.sleep(0.01)
    cap.release()

Thread(target=frame_reader, daemon=True).start()
print("Camera thread started. Press Ctrl+C to quit.")

try:
    while not stop_event.is_set():
        with buffer_lock:
            frame = frame_buffer.copy() if frame_buffer is not None else None
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # tighter Haar parameters: more neighbors, larger minSize
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(150,150)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            area = w * h
            if area >= AREA_THRESHOLD:
                dist = a * (area ** b)
                if dist <= MIN_DISTANCE_CM:
                    crop = frame[y:y+h, x:x+w]
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{ts}.jpg"
                    path = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(path, crop)
                    print(f"Face captured: {filename}, distance={dist:.1f}cm")
                    break
        # show preview
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

stop_event.set()
cv2.destroyAllWindows()
print("Stopped.")
'''