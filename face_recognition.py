import os
from PIL import Image
import torch
import numpy as np
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances


# 1. (Re-)initialize device & models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=112, margin=10, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()


# 2. Ensure the 'db' folder exists and connect to your local faces.db
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


# 3. Utility functions
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
    ids, vecs = zip(*[
        (r[0], np.frombuffer(r[1], dtype=np.float32))
        for r in rows
    ])
    dists = cosine_distances(embedding.reshape(1, -1), np.stack(vecs))[0]
    best = np.argmin(dists)
    return ids[best] if dists[best] < threshold else None

def process_image(image_path: str):
    """Detect, embed, check DB, and print New/Old face."""
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is None:
        print(f"No face detected in {os.path.basename(image_path)}")
        return
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy().flatten()
    match = find_match(emb)
    if match:
        print("Old face")
    else:
        face_id = os.path.basename(image_path)  # or generate your own ID
        add_face(face_id, emb)
        print("New face")
    


# 4. Example usage:
process_image('captured_faces/face.jpg')