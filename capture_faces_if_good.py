# -*- coding: utf-8 -*-
import os
import cv2
import time
import subprocess
from threading import Thread, Lock
from picamera2 import Picamera2
from queue import Queue
from datetime import datetime

# Configuration
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
OUTPUT_DIR = "/home/neonflake/Desktop/visitwise/captures"
FRAME_RATE = 10
RECORD_SECONDS = 5

class ImageVideoConverter:
    def __init__(self):
        # Camera setup
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        
        # Detection setup
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.capture_queue = Queue()
        self.running = True
        self.lock = Lock()

        # Ensure output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def start_camera(self):
        self.picam2.start()
        print("Camera started in preview mode")

    def capture_frames(self):
        """Main face detection and frame capture loop"""
        try:
            while self.running:
                # Capture frame from preview
                frame = self.picam2.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(100, 100)
                )
                if len(faces) > 0:
                    print("Face detected - capturing frames...")
                    self._capture_sequence()
                
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.stop()

    def _capture_sequence(self):
        """Capture frames for specified duration"""
        start_time = time.time()
        frame_count = 0
        temp_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(temp_dir, exist_ok=True)

        while (time.time() - start_time) < RECORD_SECONDS:
            with self.lock:
                frame = self.picam2.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            filename = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            frame_count += 1
            time.sleep(1/FRAME_RATE)

        # Convert to video in background
        Thread(target=self.convert_to_video, args=(temp_dir,)).start()

    def convert_to_video(self, input_dir):
        """Convert captured frames to MP4"""
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(input_dir) + ".mp4")
        
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(FRAME_RATE),
                "-pattern_type", "glob",
                "-i", os.path.join(input_dir, "*.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_file
            ], check=True)
            
            print(f"Video created: {output_file}")
            # Cleanup temporary files
            for f in os.listdir(input_dir):
                os.remove(os.path.join(input_dir, f))
            os.rmdir(input_dir)
            
        except Exception as e:
            print(f"Conversion failed: {e}")

    def stop(self):
        self.running = False
        self.picam2.stop()
        print("Camera stopped")

if __name__ == "__main__":
    converter = ImageVideoConverter()
    converter.start_camera()
    
    try:
        # Start detection in main thread
        converter.capture_frames()
    except KeyboardInterrupt:
        converter.stop()