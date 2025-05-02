import time
import subprocess
import requests
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

RAW_PATH = "/home/neonflake/video.h264"
MP4_PATH = "/home/neonflake/video.mp4"
SERVER_URL = "http://192.168.31.49:5000/upload"

def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration()
    picam2.configure(config)
    encoder = H264Encoder()

    try:
        while True:
            # Start recording
            picam2.start_recording(encoder, RAW_PATH)
            time.sleep(5)
            picam2.stop_recording()

            # Convert to MP4
            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate", "30",
                    "-i", RAW_PATH,
                    "-c", "copy",
                    MP4_PATH
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during video conversion: {e}")
                continue

            # Upload video
            try:
                with open(MP4_PATH, 'rb') as f:
                    response = requests.post(SERVER_URL, files={'video': f})
                print(f"Upload response: {response.status_code}")
            except requests.RequestException as e:
                print(f"Error during upload: {e}")
    finally:
        picam2.close()

if __name__ == "__main__":
    main()
