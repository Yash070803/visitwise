# -*- coding: utf-8 -*-
import os
import cv2
import time
from threading import Thread, Event
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import subprocess
import requests

# --- Configuration ---
RAW_DIR       = "faces_raw"
FINAL_DIR     = "captured_faces"
RAW_PATH      = "/home/neonflake/Desktop/visitwise/video.h264"
MP4_PATH      = "/home/neonflake/Desktop/visitwise/video.mp4"
CASCADE_DIR   = os.path.join(os.path.dirname(__file__), "cascades")
UPLOAD_URL    = "http://yourserver.com/api/faces/upload"
RECORD_TIME   = 5  # seconds

# ensure directories exist
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

face_cascade   = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml"))
eye_cascade    = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_eye.xml"))
nose_cascade   = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_mcs_nose.xml"))
mouth_cascade  = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_mcs_mouth.xml"))

stop_event = Event()

def async_upload(file_path):
    def _upload():
        try:
            with open(file_path, 'rb') as f:
                files = {'video': f}
                res = requests.post(UPLOAD_URL, files=files, timeout=10)
            if res.status_code == 200:
                print("Upload successful.")
            else:
                print(f"Upload failed with status: {res.status_code}")
        except Exception as e:
            print(f"Upload exception: {e}")
    Thread(target=_upload, daemon=True).start()

def record_and_upload_video(raw_path, mp4_path, duration=RECORD_TIME):
    picam2 = Picamera2()
    config = picam2.create_video_configuration()
    picam2.configure(config)
    encoder = H264Encoder()
    try:
        picam2.start_recording(encoder, raw_path)
        time.sleep(duration)
        picam2.stop_recording()
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", raw_path,
                "-c", "copy",
                mp4_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video conversion: {e}")
            return
        async_upload(mp4_path)
    finally:
        picam2.close()

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
        if len(eyes) < 2:
            continue
        if nose_cascade:
            noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
            if len(noses) < 1:
                continue
        if mouth_cascade:
            mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
            if len(mouths) < 1:
                continue
        ts = int(time.time())
        crop_path = os.path.join(FINAL_DIR, f"good_{ts}.jpg")
        cv2.imwrite(crop_path, frame[y:y+h, x:x+w])
        print(f"Good face found: {crop_path}")
        record_and_upload_video(RAW_PATH, MP4_PATH)
        break

def camera_loop():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    try:
        while not stop_event.is_set():
            frame = picam2.capture_array()
            process_frame(frame)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        picam2.close()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    camera_loop()


def print_hello():
    print("hello ")
