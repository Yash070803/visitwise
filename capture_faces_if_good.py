# -*- coding: utf-8 -*-
import os
import cv2
import time
import subprocess
from threading import Thread, Lock
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

# --- Configuration ---
FACE_CASCADE = "haarcascade_frontalface_default.xml"
VIDEO_PATH = "/home/neonflake/Desktop/visitwise/recording.mp4"
RESOLUTION = (1280, 720)
RECORD_DURATION = 5  # seconds

class FaceRecorder:
    def __init__(self):
        self.picam2 = Picamera2()
        self.camera_lock = Lock()
        self.recording = False
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        
        # Configure camera for dual use (preview + recording)
        self.config = self.picam2.create_video_configuration(
            main={"size": RESOLUTION},
            encode="main"
        )
        self.picam2.configure(self.config)
        self.picam2.start()

    def _convert_to_mp4(self, h264_path):
        """Convert raw H264 to MP4"""
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", h264_path,
                "-c", "copy",
                VIDEO_PATH
            ], check=True)
            print(f"Video saved: {VIDEO_PATH}")
        except Exception as e:
            print(f"Conversion error: {e}")

    def _record_video(self):
        """Handle video recording in background"""
        with self.camera_lock:
            self.recording = True
            try:
                encoder = H264Encoder()
                temp_path = "/tmp/temp_recording.h264"
                self.picam2.start_encoder(encoder, temp_path)
                time.sleep(RECORD_DURATION)
                self.picam2.stop_encoder()
                self._convert_to_mp4(temp_path)
                os.remove(temp_path)
            except Exception as e:
                print(f"Recording error: {e}")
            self.recording = False

    def run_detection(self):
        """Main detection loop"""
        try:
            while True:
                # Capture frame without interrupting encoder
                frame = self.picam2.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Face detection
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(100, 100)
                )

                if len(faces) > 0 and not self.recording:
                    print("Face detected - starting recording!")
                    Thread(target=self._record_video, daemon=True).start()

                time.sleep(0.1)  # Reduce CPU usage

        except KeyboardInterrupt:
            self.picam2.stop()
            print("\nCamera stopped")

if __name__ == "__main__":
    recorder = FaceRecorder()
    recorder.run_detection()