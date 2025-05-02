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
MAX_DISTANCE_CM = 150.0
AREA_THRESHOLD  = 20000
EYE_SYMM_THRESH = 20

# Calibration coefficients
a = 9703.20
b = -0.4911842338691967

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False)

# Ensure output directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Haar cascades setup
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

face_cascade = cv2.CascadeClassifier(find_cascade("lbpcascade_frontalface_improved.xml"))
eye_cascade  = cv2.CascadeClassifier(find_cascade("haarcascade_eye.xml"))
nose_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_nose.xml")) if os.path.exists(find_cascade("haarcascade_mcs_nose.xml")) else None
mouth_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_mouth.xml")) if os.path.exists(find_cascade("haarcascade_mcs_mouth.xml")) else None

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.set_controls({
    "ExposureTime": 15000,
    "AnalogueGain": 6.0,
    "AwbEnable": True,
    "AeEnable": True,
    "Brightness": 0.2,
    "Contrast": 1.2,
    "Sharpness": 1.5,
    "FrameRate": 30
})

# Shared resources
frame_buffer = None
buffer_lock  = Lock()
stop_event   = Event()
camera_lock  = Lock()  # New lock for camera access

def frame_reader():
    global frame_buffer
    picam2.start()
    while not stop_event.is_set():
        with camera_lock:
            img = picam2.capture_array("main")
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with buffer_lock:
            frame_buffer = frame
        time.sleep(0.01)
    picam2.stop()

def record_video():
    with camera_lock:  # Ensure exclusive camera access
        try:
            encoder = H264Encoder()
            picam2.start_recording(encoder, RAW_PATH)
            time.sleep(5)  # Record for 5 seconds
            picam2.stop_recording()
            
            # Convert to MP4
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", RAW_PATH,
                "-c", "copy",
                MP4_PATH
            ], check=True)
            
            # Upload logic here if needed
            
        except Exception as e:
            print("Recording error:", e)

t = Thread(target=frame_reader, daemon=True)
t.start()

try:
    while not stop_event.is_set():
        with buffer_lock:
            frame = frame_buffer.copy() if frame_buffer is not None else None
            
        if frame is None:
            time.sleep(0.05)
            continue
            
        # Image processing and face detection logic
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        fgmask = fgbg.apply(gray)
        if cv2.countNonZero(fgmask) < 500:
            continue
            
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(150,150))
        if len(faces) == 0:
            continue
            
        x, y, w, h = faces[0]
        area = w * h
        if area < AREA_THRESHOLD:
            continue
            
        aspect = w / h
        if not (0.7 < aspect < 1.3):
            continue
            
        # Start recording when valid face detected
        record_thread = Thread(target=record_video)
        record_thread.start()
        
        # Save face image
        ts = time.strftime("%Y%m%d_%H%M%S")
        crop = frame[y:y+h, x:x+w]
        final_path = os.path.join(FINAL_DIR, f"face_{ts}.jpg")
        cv2.imwrite(final_path, crop)
        
        time.sleep(1)  # Cooldown between detections

except KeyboardInterrupt:
    stop_event.set()
    picam2.stop()

print("Stopped.")