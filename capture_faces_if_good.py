# -*- coding: utf-8 -*-
import os
import cv2
import time
from threading import Thread, Lock, Event
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import subprocess
import requests

# --- Configuration ---
RAW_DIR       = "faces_raw"
FINAL_DIR     = "captured_faces"
CASCADE_DIR   = os.path.join(os.path.dirname(__file__), "cascades")
RAW_PATH      = "/home/neonflake/Desktop/visitwise/video.h264"
MP4_PATH      = "/home/neonflake/Desktop/visitwise/video.mp4"
MAX_DISTANCE_CM = 150.0     # maximum capture distance in cm
AREA_THRESHOLD  = 20000     # minimum required face box area (px^2)
EYE_SYMM_THRESH = 20        # eye alignment tolerance (pixels)

# calibration coefficients for distance = a * (area ** b)
a = 9703.20
b = -0.4911842338691967

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False)

# ensure output directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MP4_PATH), exist_ok=True)

# helper: locate Haar cascade XMLs
SYSTEM_PATHS = [
    "/usr/share/opencv4/haarcascades",
    "/usr/share/opencv/haarcascades",
    "/usr/local/share/opencv4/haarcascades",
]

def find_cascade(name):
    local = os.path.join(CASCADE_DIR, name)
    if os.path.isfile(local):
        return local
    for base in SYSTEM_PATHS:
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cascade '{name}' not found.")

# load cascades
face_cascade = cv2.CascadeClassifier(find_cascade("lbpcascade_frontalface_improved.xml"))
eye_cascade  = cv2.CascadeClassifier(find_cascade("haarcascade_eye.xml"))
try:
    nose_cascade  = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_nose.xml"))
except FileNotFoundError:
    nose_cascade = None
try:
    mouth_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_mouth.xml"))
except FileNotFoundError:
    mouth_cascade = None

# async upload (if needed)
def async_upload(path):
    try:
        with open(path, 'rb') as f:
            res = requests.post(
                'https://visit-wise-api.onrender.com/api/faces/upload',
                files={'image': f},
                data={'deviceId': 'DEV3617'}
            )
        print("Upload status:", res.status_code)
    except Exception as e:
        print("Upload exception:", e)
picam2 = Picamera2()

def record_and_upload_video(raw_path, mp4_path, record_duration=5):
    global picam2

    print("Pausing preview and starting video recording...")

    try:
        picam2.stop()
        config = picam2.create_video_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)
        encoder = H264Encoder()
        picam2.start_recording(encoder, raw_path)
        time.sleep(record_duration)
        picam2.stop_recording()
        print("Recording stopped, converting to MP4…")

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", raw_path,
            "-c", "copy",
            mp4_path
        ], check=True)
        print("Conversion done:", mp4_path)

        # upload
        with open(mp4_path, 'rb') as f:
            res = requests.post(
                'https://visit-wise-api.onrender.com/api/faces/upload',
                files={'video': f},
                data={'deviceId': 'DEV3617'}
            )
        print("Upload status:", res.status_code)

    except Exception as e:
        print("Error:", e)

    # reconfigure back to preview mode
    print("Resuming detection stream...")
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()

# shared frame buffer
frame_buffer = None
buffer_lock  = Lock()
stop_event   = Event()

# camera capture thread
def frame_reader():
    global frame_buffer
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size":(1280,720)})
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({
        "ExposureTime":15000,
        "AnalogueGain":6.0,
        "AwbEnable":True,
        "AeEnable":True,
        "Brightness":0.2,
        "Contrast":1.2,
        "Sharpness":1.5,
        "FrameRate": 30
    })
    while not stop_event.is_set():
        img = picam2.capture_array("main")
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with buffer_lock:
            frame_buffer = frame
        time.sleep(0.01)
    picam2.stop()

t = Thread(target=frame_reader, daemon=True)
t.start()

try:
    while not stop_event.is_set():
        with buffer_lock:
            frame = frame_buffer.copy() if frame_buffer is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(gray)
        if cv2.countNonZero(fgmask) < 500:
            continue

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8, minSize=(150,150)
        )
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        area = w * h
        if area < AREA_THRESHOLD:
            print("threshold not met")
            continue

        aspect = w / h
        if not (0.7 < aspect < 1.3):
            print("aspect ratio not met")
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        crop = frame[y:y+h, x:x+w]
        raw_img_path = os.path.join(RAW_DIR, f"raw_{ts}.jpg")
        cv2.imwrite(raw_img_path, crop)
        dist = a * (area**b) + 50
        print(f"Face at {dist:.1f} cm saved: {raw_img_path}")
        time.sleep(0.01)
        record_and_upload_video(RAW_PATH, MP4_PATH)
        print("Done recording video.\n")

        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30)
        )
        if len(eyes) < 2:
            print("no eyes found")
            

        if nose_cascade:
            noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
            if len(noses) < 1:
                print("nose not found")

        if mouth_cascade:
            mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
            if len(mouths) < 1:
                print("mouth not clear")

        if dist > MAX_DISTANCE_CM:
            print(f"Too far ({dist:.1f} cm), skipping.")
            continue

        final_img_path = os.path.join(FINAL_DIR, f"good_{ts}.jpg")
        cv2.imwrite(final_img_path, crop)
        print(f"Good face saved: {final_img_path}")

        record_and_upload_video(RAW_PATH, MP4_PATH)
        print("Done recording video.\n")

        time.sleep(1)

except KeyboardInterrupt:
    stop_event.set()

cv2.destroyAllWindows()
print("Stopped.")