import os
import cv2
import pickle
import face_recognition
from ultralytics import YOLO
import numpy as np
import dlib

# Load saved embeddings
with open("./new_embeddings.pkl", "rb") as f:
    saved_encodings = pickle.load(f)

yolo = YOLO("../yolov8n-face.pt")
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

def align_face(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face = img[y1:y2, x1:x2]
    return face

def get_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(img_rgb)
    if not enc:
        return None
    return enc[0]

def predict_identity_with_verification(test_img_path, dataset_path="./newdata", verification_count=5, threshold=0.45):
    img = cv2.imread(test_img_path)
    results = yolo(img)
    if len(results[0].boxes) == 0:
        print("No face detected in test image!")
        return None
    
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    test_face = align_face(img, box)
    test_embedding = get_embedding(test_face)
    if test_embedding is None:
        print("Failed to extract embedding from test image.")
        return None

    # Step 1: Predict identity from saved embeddings
    min_dist = float("inf")
    identity = "Unknown"
    for record in saved_encodings:
        dist = np.linalg.norm(record["embedding"] - test_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = record["name"]

    if min_dist > threshold:
        print(f"❌ No confident match found. Closest match: {identity} (distance={min_dist:.4f})")
        return "Unknown"

    print(f"✅ Initial match from embedding: {identity} (distance={min_dist:.4f})")

    # Step 2: Siamese-style verification with dataset/{identity}/
    person_dir = os.path.join(dataset_path, identity)
    if not os.path.exists(person_dir):
        print("Person folder not found.")
        return "Unknown"

    verification_passed = 0
    checked = 0

    for img_name in os.listdir(person_dir):
        if checked >= verification_count:
            break
        img_path = os.path.join(person_dir, img_name)
        db_img = cv2.imread(img_path)
        if db_img is None:
            continue
        db_results = yolo(db_img)
        if len(db_results[0].boxes) == 0:
            continue
        db_box = db_results[0].boxes[0].xyxy[0].cpu().numpy()
        db_face = align_face(db_img, db_box)
        db_embedding = get_embedding(db_face)
        if db_embedding is None:
            continue
        checked += 1

        # Siamese-style comparison
        dist = np.linalg.norm(db_embedding - test_embedding)
        if dist < threshold:
            verification_passed += 1

    print(f"🔁 Verified with {checked} images. Matches found: {verification_passed}/{checked}")
    if verification_passed >= (verification_count // 2):  # majority voting
        print(f"✅ Final Match Confirmed: {identity}")
        return identity
    else:
        print("❌ Final verification failed. Possibly incorrect match.")
        return "Unknown"

# Example usage:
predict_identity_with_verification("../dataset/Jayant/j3.jpg")

