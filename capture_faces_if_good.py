
#--------------Picamera2-------------------------------
# -*- coding: utf-8 -*-
import os
import cv2
import time
from threading import Thread, Lock, Event
from picamera2 import Picamera2
import requests
# --- Configuration ---
RAW_DIR       = "faces_raw"
FINAL_DIR     = "captured_faces"
CASCADE_DIR   = os.path.join(os.path.dirname(__file__), "cascades")

MAX_DISTANCE_CM = 150.0     # maximum capture distance in cm
AREA_THRESHOLD  = 20000     # minimum required face box area (px^2)
EYE_SYMM_THRESH = 20        # eye alignment tolerance (pixels)

fgbg = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False)


# calibration coefficients for distance = a * (area ** b)
a = 9703.20
b = -0.4911842338691967

# ensure output directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# helper: locate Haar cascade XMLs
SYSTEM_PATHS = [
    "/usr/share/opencv4/haarcascades",
    "/usr/share/opencv/haarcascades",
    "/usr/local/share/opencv4/haarcascades",
]
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

def find_cascade(name):
    # local cascades folder
    local = os.path.join(CASCADE_DIR, name)
    if os.path.isfile(local):
        return local
    # system haarcascades
    for base in SYSTEM_PATHS:
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cascade '{name}' not found in ./cascades or system paths.")

# load required cascades
face_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_frontalface_default.xml"))
eye_cascade  = cv2.CascadeClassifier(find_cascade("haarcascade_eye.xml"))
# optional nose and mouth cascades
try:
    nose_cascade  = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_nose.xml"))
except FileNotFoundError:
    nose_cascade = None
try:
    mouth_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_mcs_mouth.xml"))
except FileNotFoundError:
    mouth_cascade = None

# shared frame buffer
default = None
frame_buffer = default
buffer_lock  = Lock()
stop_event   = Event()

# camera capture thread using Picamera2
def frame_reader():
    global frame_buffer
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size":(1280,720)})
    picam2.configure(config)
    picam2.start()
    # tune image for low light and clarity
    picam2.set_controls({
        "ExposureTime":10000,  # 30 ms
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
        # convert RGB to BGR
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with buffer_lock:
            frame_buffer = frame
        time.sleep(0.01)
    picam2.stop()

# start reader thread
t = Thread(target=frame_reader, daemon=True)
t.start()
print("Camera thread started. Press Ctrl+C to quit.")

try:
    while not stop_event.is_set():
        # fetch latest frame
        with buffer_lock:
            frame = frame_buffer.copy() if frame_buffer is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        # rotate to correct orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fgmask = fgbg.apply(gray)
        if cv2.countNonZero(fgmask) < 500:  # Skip if no motion
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces with stricter parameters
        small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
        faces = face_cascade.detectMultiScale(gray,
            scaleFactor=1.1, minNeighbors=8, minSize=(150,150)
        )
        if len(faces) > 0:
            # process first detected face
            x,y,w,h = faces[0]
            area = w*h

            if area < AREA_THRESHOLD:
                print(f"Face area too small ({area}), skipping.")
                continue

            # Check aspect ratio (typical faces are ~square)
            aspect_ratio = w / h
            if not (0.7 < aspect_ratio < 1.3):
                print(f"Invalid aspect ratio {aspect_ratio:.2f}, skipping.")
                continue
            # Stage1: save raw crop
            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(RAW_DIR, f"raw_{ts}.jpg")
            crop = frame[y:y+h, x:x+w]
            cv2.imwrite(raw_path, crop)
            dist = a * (area**b) + 50
            print(f"Stage1 saved face.jpg ... captured at a distance of {dist:.1f}")

            # check eyes
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
            if len(eyes) < 2:
                print("missing eyes, skipping Stage2")
                # continue
            # optional nose
            if nose_cascade:
                noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
                if len(noses) < 1:
                    print("missing nose, skipping Stage2")
                    continue
            # optional mouth
            if mouth_cascade:
                mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
                if len(mouths) < 1:
                    print("missing mouth, skipping Stage2")
                    continue
            # eye symmetry check
            # eyes = sorted(eyes, key=lambda e: e[0])
            # mid1 = eyes[0][1] + eyes[0][3]//2
            # mid2 = eyes[1][1] + eyes[1][3]//2
            # if abs(mid1-mid2) > EYE_SYMM_THRESH:
            #     print(f"eyes not level, skip Stage2")
            #     continue
            # distance check
            dist = a * (area**b) + 50
            print(f"Distance: {dist:.1f} cm")
            if dist > MAX_DISTANCE_CM:
                print(f"too far: {dist:.1f}cm, skip Stage2")
                continue
            # Stage2: save good crop
            final_path = os.path.join(FINAL_DIR, f"good_{ts}.jpg")
            cv2.imwrite(final_path,crop)
            print(f"Stage2 saved good_{ts}.jpg (distance={dist:.1f}cm)")
            async_upload(final_path)
            print("capturing another image\n..\n..")

            
        # show preview
        #cv2.imshow("Preview", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

# cleanup
stop_event.set()
cv2.destroyAllWindows()
print("Stopped.")
