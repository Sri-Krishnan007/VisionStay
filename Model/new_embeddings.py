import os
import cv2
import pickle
import face_recognition
from tqdm import tqdm
from ultralytics import YOLO
import dlib

# Load YOLOv8n-face model
yolo = YOLO("../yolov8n-face.pt")

# Optional: dlib's 68-point face landmark model for alignment
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

data_dir = "./newdata"
encodings = []

def align_face(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face = img[y1:y2, x1:x2]
    
    # Optional alignment using dlib
    rect = dlib.rectangle(x1, y1, x2, y2)
    landmarks = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rect)
    
    # You can implement affine alignment here if needed (not mandatory)
    # For simplicity, we’ll skip it for now and just return the cropped face
    return face

for person in tqdm(os.listdir(data_dir)):
    person_path = os.path.join(data_dir, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Detect faces using YOLO
        results = yolo(img)
        if len(results[0].boxes) == 0:
            continue
        
        # Use only the first detected face (or loop if multiple faces per image)
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        face = align_face(img, box)
        if face is None or face.size == 0:
            continue

        # Convert to RGB for face_recognition
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Get embedding using face_recognition
        enc = face_recognition.face_encodings(face_rgb)
        if not enc:
            continue
        embedding = enc[0]

        encodings.append({
            "name": person,
            "embedding": embedding,
            "img_path": img_path
        })

# Save embeddings
with open("new_embeddings.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("✅ Embeddings saved to new_embeddings.pkl")
