# -*- coding: utf-8 -*-
import os
import cv2
import time
import sqlite3
import numpy as np
from threading import Thread, Lock, Event
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
OUTPUT_DIR = "captured_faces"
CASCADE_DIR = os.path.join(os.path.dirname(__file__), "cascades")
CAMERA_INDEX = "http://packproof:5000/video"  # 0 for default camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
AREA_THRESHOLD = 20000       # Minimum face area (px²)
MIN_DISTANCE_CM = 150.0      # Max distance to capture
COOLDOWN_SECONDS = 10        # Time between captures

# Calibration coefficients
a = 9703.20
b = -0.4911842338691967

# Initialize face recognition models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=112, margin=10, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Database setup
os.makedirs('db', exist_ok=True)
conn = sqlite3.connect(os.path.join('db', 'facesVisitwise.db'))
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS facesVisitwise (
        id TEXT PRIMARY KEY,
        embedding BLOB
    )
''')
conn.commit()

def add_face(face_id: str, embedding: np.ndarray):
    emb_bytes = embedding.astype(np.float32).tobytes()
    c.execute(
        'INSERT OR REPLACE INTO facesVisitwise (id, embedding) VALUES (?, ?)',
        (face_id, emb_bytes)
    )
    conn.commit()

def find_match(embedding: np.ndarray, threshold: float = 0.6):
    rows = c.execute('SELECT id, embedding FROM facesVisitwise').fetchall()
    if not rows:
        return None
    ids, vecs = zip(*[(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows])
    dists = cosine_distances(embedding.reshape(1, -1), np.stack(vecs))[0]
    best = np.argmin(dists)
    return ids[best] if dists[best] < threshold else None

# --- Face Capture Setup ---
def find_cascade(name):
    local = os.path.join(CASCADE_DIR, name)
    if os.path.isfile(local):
        return local
    system_paths = [
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
    ]
    for base in system_paths:
        full = os.path.join(base, name)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(f"Cascade '{name}' not found.")

face_cascade = cv2.CascadeClassifier(find_cascade("haarcascade_frontalface_default.xml"))

# Shared frame buffer
frame_buffer = None
buffer_lock = Lock()
stop_event = Event()
last_capture_time = 0  # Cooldown tracker

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
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(150, 150)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            area = w * h
            if area >= AREA_THRESHOLD:
                dist = a * (area ** b) + 50
                if dist <= MIN_DISTANCE_CM:
                    current_time = time.time()
                    if current_time - last_capture_time >= COOLDOWN_SECONDS:
                        crop = frame[y:y+h, x:x+w]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(crop_rgb)
                        
                        # Extract embedding
                        face_tensor = mtcnn(img_pil)
                        if face_tensor is not None:
                            with torch.no_grad():
                                emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
                            match = find_match(emb)
                            
                            if match is None:
                                # New face detected
                                ts = time.strftime("%Y%m%d_%H%M%S")
                                filename = f"face_{ts}.jpg"
                                path = os.path.join(OUTPUT_DIR, filename)
                                cv2.imwrite(path, crop)
                                add_face(filename, emb)
                                last_capture_time = current_time
                                print(f"New face captured: {filename}, distance={dist:.1f}cm")
                            else:
                                print(f"Known face: {match}")
                        else:
                            print("No face found in crop")

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

stop_event.set()
cv2.destroyAllWindows()
conn.close()
print("Stopped.")