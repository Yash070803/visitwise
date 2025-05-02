# -*- coding: utf-8 -*-
import os
import cv2
import time
from threading import Thread, Lock
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import subprocess

# --- Configuration ---
RAW_PATH = "/home/neonflake/Desktop/visitwise/video.h264"
MP4_PATH = "/home/neonflake/Desktop/visitwise/video.mp4"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Initialize camera once
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)

# Face detection setup
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
camera_lock = Lock()
is_recording = False

def convert_to_mp4():
    """Convert the raw video to MP4 format"""
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", RAW_PATH,
            "-c", "copy",
            MP4_PATH
        ], check=True)
        print(f"Video saved: {MP4_PATH}")
    except Exception as e:
        print("Conversion error:", e)

def record_video():
    """Handle video recording"""
    global is_recording
    with camera_lock:
        try:
            encoder = H264Encoder()
            picam2.start_recording(encoder, RAW_PATH)
            time.sleep(5)  # Record for 5 seconds
            picam2.stop_recording()
            convert_to_mp4()
        except Exception as e:
            print("Recording error:", e)
    is_recording = False

def detection_loop():
    """Main face detection loop"""
    global is_recording
    picam2.start()
    
    try:
        while True:
            # Capture frame
            img = picam2.capture_array("main")
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )

            # Start recording if faces found and not already recording
            if len(faces) > 0 and not is_recording:
                is_recording = True
                Thread(target=record_video).start()
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        picam2.stop()
        print("Stopped.")

if __name__ == "__main__":
    detection_loop()