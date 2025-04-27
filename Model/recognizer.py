import torch
import os
import cv2
import pickle
import face_recognition
from ultralytics import YOLO
import numpy as np
import dlib # Although loaded, it's not used for alignment in the original code's align_face
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time # To calculate FPS
import collections # For deque (frame buffer)
from datetime import datetime # For snapshot filenames

# --- Configuration ---
# General
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_PATH = "./new_embeddings.pkl"
YOLO_MODEL_PATH = "./yolov8n-face.pt" # Adjust if your path is different
DLIB_PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat" # Adjust if needed
DATASET_PATH = "./newdata" # Path to the dataset for verification
WEBCAM_INDEX = 0 # 0 for default webcam, change if you have multiple

# Recognition & Verification
VERIFICATION_COUNT = 3 # How many images from dataset to check for verification
THRESHOLD = 0.45 # Distance threshold for matching

# Liveness Detection
LIVENESS_ENABLED = True # Set to False to disable liveness check
LIVENESS_BUFFER_SIZE = 4 # Number of frames to store for liveness check
PIXEL_DIFF_THRESHOLD = 25 # Pixel intensity difference threshold for absdiff
FRAME_DIFF_THRESHOLD = 0.01 # Percentage of pixels that must change between frames
LIVENESS_MIN_CHANGED_FRAMES = 2 # How many frames in the buffer must show change
LIVENESS_SNAPSHOT_DIR = "./liveness_snaps" # Directory to save snapshots on liveness pass

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Liveness Check Enabled: {LIVENESS_ENABLED}")

# --- Create Directories ---
os.makedirs(LIVENESS_SNAPSHOT_DIR, exist_ok=True)

# --- Load Models and Data ---
try:
    # Load saved embeddings
    with open(EMBEDDINGS_PATH, "rb") as f:
        saved_encodings = pickle.load(f)
    print(f"[INFO] Loaded {len(saved_encodings)} known embeddings.")

    # Load YOLO face detector
    # Automatically uses DEVICE specified during inference calls
    yolo = YOLO(YOLO_MODEL_PATH)
    print("[INFO] YOLO face detector loaded.")

    # Load dlib shape predictor (optional for potential future use)
    try:
        predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
        print("[INFO] Dlib shape predictor loaded.")
    except Exception as e:
        print(f"[WARNING] Could not load Dlib predictor from {DLIB_PREDICTOR_PATH}: {e}. Not used in current alignment.")
        predictor = None # Set to None if loading fails

except FileNotFoundError as e:
    print(f"[ERROR] Failed to load required file: {e}. Please check paths.")
    exit()
except Exception as e:
    print(f"[ERROR] An error occurred during initialization: {e}")
    exit()

# --- Helper Functions ---

def save_snapshot(frame, directory):
    """Saves a frame snapshot with a timestamp filename."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    try:
        cv2.imwrite(filepath, frame)
        # print(f"[DEBUG] Saved snapshot: {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to save snapshot to {filepath}: {e}")
        return None

def detect_liveness(frames):
    """
    Detects basic liveness by comparing pixel-level differences between frames.
    Returns True if motion suggests liveness, False otherwise.
    Saves snapshots if live.
    """
    # print("[DEBUG] Performing Liveness Check...")
    change_count = 0
    snapshot_paths = [] # Store paths of frames showing change

    if len(frames) < 2:
        # print("[DEBUG] Liveness check needs at least 2 frames.")
        return True # Not enough data to determine spoof, assume live for now

    for i in range(1, len(frames)):
        # Ensure frames are valid
        if frames[i-1] is None or frames[i] is None:
            print("[WARN] Invalid frame encountered during liveness check.")
            continue

        # Convert both frames to grayscale
        gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Resize to match if needed (shouldn't be necessary if buffer contains consistent frames)
        if gray1.shape != gray2.shape:
             print(f"[WARN] Frame shape mismatch in buffer: {gray1.shape} vs {gray2.shape}. Resizing.")
             try:
                 gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
             except cv2.error as e:
                 print(f"[ERROR] Failed to resize frame for liveness check: {e}")
                 continue # Skip this pair

        # Get absolute difference
        try:
            diff = cv2.absdiff(gray1, gray2)
        except cv2.error as e:
             print(f"[ERROR] Failed cv2.absdiff: {e}. Shapes: {gray1.shape}, {gray2.shape}")
             continue # Skip this pair


        # Apply threshold
        _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Calculate the percentage of changed pixels
        if thresh.size == 0: continue # Avoid division by zero
        diff_ratio = np.sum(thresh > 0) / thresh.size

        # print(f"[DEBUG] Liveness Frame {i} vs {i-1}: Change ratio = {diff_ratio:.4f}")

        if diff_ratio > FRAME_DIFF_THRESHOLD:
            change_count += 1
            # Optional: Save snapshot of the *current* frame when change is detected
            # snap_path = save_snapshot(frames[i], LIVENESS_SNAPSHOT_DIR)
            # if snap_path:
            #     snapshot_paths.append(snap_path)

    # print(f"[DEBUG] Liveness Check: {change_count} frame(s) showed significant change.")

    if change_count >= LIVENESS_MIN_CHANGED_FRAMES:
        # print("[DEBUG] Liveness confirmed by motion.")
        # Optional: Save the *last* frame as confirmation snapshot
        save_snapshot(frames[-1], LIVENESS_SNAPSHOT_DIR)
        return True # Liveness detected
    else:
        # print("[DEBUG] Insufficient motion detected. Potential Spoof.")
        return False # Not enough motion


def align_face(img, bbox):
    """
    Basic face alignment by cropping the bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Ensure coordinates are within image bounds and valid
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None # Invalid box
    face = img[y1:y2, x1:x2]
    return face

def get_embedding(face_img):
    """Extracts face embedding using face_recognition library."""
    if face_img is None or face_img.size == 0:
        # print("[DEBUG] Invalid face image passed to get_embedding.")
        return None
    try:
        # face_recognition expects RGB
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # The box is the whole image since it's pre-cropped
        face_locations = [(0, img_rgb.shape[1], img_rgb.shape[0], 0)]
        enc = face_recognition.face_encodings(img_rgb, known_face_locations=face_locations, num_jitters=1, model='small') # 'small' model is faster
        if not enc:
            # print("[DEBUG] No encoding found for face.")
            return None
        return enc[0]
    except Exception as e:
        # print(f"[ERROR] Error during embedding extraction: {e}") # Can be noisy
        return None

def process_frame_for_recognition(frame, frame_buffer, yolo_model, saved_encs, dataset_p, verify_count, thresh):
    """
    Detects faces, performs liveness check, finds closest match, and performs verification.
    Returns annotations (bboxes and names) for drawing.
    """
    annotations = []
    # Perform detection - Use device specified
    try:
        results = yolo_model(frame, device=DEVICE, verbose=False) # verbose=False reduces console spam
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e}")
        return annotations # Return empty on error

    if not results or len(results[0].boxes) == 0:
        # print("[DEBUG] No faces detected in this frame.")
        return annotations # Return empty list if no faces

    detected_boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    # --- Perform Liveness Check (if enabled and buffer is full) ---
    is_live = True # Assume live by default or if check is disabled/not ready
    if LIVENESS_ENABLED and len(frame_buffer) == LIVENESS_BUFFER_SIZE:
        is_live = detect_liveness(list(frame_buffer))
        if not is_live:
             print("[WARNING] Liveness Check Failed (Potential Spoof)!")
    elif LIVENESS_ENABLED:
        # print("[DEBUG] Liveness buffer not full yet...")
        pass # Continue processing, but without liveness result yet

    # --- Process each detected face ---
    for i, box in enumerate(detected_boxes):
        confidence = confidences[i]
        # print(f"[DEBUG] Detected face {i} with confidence: {confidence:.2f}")

        # --- Liveness Check Result ---
        if LIVENESS_ENABLED and not is_live:
            # If the frame is deemed not live, mark all detected faces as Spoof
            annotations.append({"bbox": box, "identity": "Spoof Detected", "live": False})
            continue # Skip recognition/verification for this face if frame is spoof

        # --- Proceed with Recognition if Live / Liveness Disabled ---
        test_face = align_face(frame, box)
        if test_face is None:
            # print("[DEBUG] Alignment failed or resulted in invalid box.")
            annotations.append({"bbox": box, "identity": "Align Fail", "live": is_live})
            continue

        test_embedding = get_embedding(test_face)
        if test_embedding is None:
            # print("[DEBUG] Embedding extraction failed for a detected face.")
            annotations.append({"bbox": box, "identity": "Embed Fail", "live": is_live})
            continue

        # Step 1: Predict identity from saved embeddings
        min_dist = float("inf")
        initial_identity = "Unknown"

        for record in saved_encs:
            dist = np.linalg.norm(record["embedding"] - test_embedding)
            if dist < min_dist:
                min_dist = dist
                initial_identity = record["name"]

        final_identity = "Unknown"
        display_text = f"Unknown ({min_dist:.2f})" # Default text

        # print(f"[DEBUG] Face {i}: Closest match: {initial_identity} (dist={min_dist:.4f}), Threshold: {thresh}")

        if min_dist <= thresh:
            # print(f"[DEBUG] Face {i}: Initial match found: {initial_identity}")
            # Step 2: Siamese-style verification (if initial match found)
            person_dir = os.path.join(dataset_p, initial_identity)
            verification_passed = 0
            checked = 0

            if os.path.exists(person_dir) and os.path.isdir(person_dir):
                img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                np.random.shuffle(img_files) # Shuffle to get random samples

                for img_name in img_files:
                    if checked >= verify_count:
                        break
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        db_img = cv2.imread(img_path)
                        if db_img is None: continue

                        # Find the *main* face in the db image
                        db_results = yolo_model(db_img, device=DEVICE, verbose=False) # Use GPU for verification image processing too
                        if not db_results or len(db_results[0].boxes) == 0: continue

                        db_box = db_results[0].boxes.xyxy[0].cpu().numpy()
                        db_face = align_face(db_img, db_box)
                        if db_face is None: continue

                        db_embedding = get_embedding(db_face)
                        if db_embedding is None: continue

                        checked += 1
                        # Compare db embedding with the *test* embedding
                        dist = np.linalg.norm(db_embedding - test_embedding)
                        # print(f"[DEBUG] Verifying {initial_identity} with {img_name}, dist: {dist:.4f}")
                        if dist < thresh: # Use the same threshold for consistency
                            verification_passed += 1
                    except Exception as e:
                         print(f"[WARN] Error processing verification image {img_path}: {e}")

                # print(f"[DEBUG] Verification for {initial_identity}: {verification_passed}/{checked} passed.")
                # Verification decision: More than half needed
                if checked > 0 and verification_passed >= (checked // 2) + 1:
                    final_identity = initial_identity
                    display_text = f"{final_identity} ({min_dist:.2f})"
                    # print(f"[INFO] Verification confirmed: {final_identity}")
                else:
                    # Verification failed even though initial match was close
                    final_identity = "Unknown"
                    display_text = f"VerifyFail ({initial_identity}? {min_dist:.2f})"
                    # print(f"[INFO] Verification failed for {initial_identity}")
            else:
                # Verification dataset missing for the initially matched person
                final_identity = "Unknown"
                display_text = f"NoVerifyData ({initial_identity}? {min_dist:.2f})"
                # print(f"[WARN] Verification dataset not found for {initial_identity}")
        else:
            # Initial match distance was too high
            final_identity = "Unknown"
            display_text = f"Unknown ({min_dist:.2f})"
            # print(f"[INFO] Face {i}: Initial match distance too high.")

        annotations.append({"bbox": box, "identity": display_text, "live": is_live})

    return annotations

# --- Matplotlib Setup ---
if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 7))  # Adjust figure size if needed
    fig.canvas.manager.set_window_title("Live Face Recognition with Liveness Check")
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder
    img_display = ax.imshow(cv2.cvtColor(placeholder_img, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()  # Hide axes
    plt.tight_layout()

    # --- Live Video Loop ---
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        exit()

    print("[INFO] Starting video stream. Press 'q' in the plot window or Ctrl+C in terminal to quit.")

    running = True
    prev_time = time.time()
    frame_count = 0
    fps_text = "FPS: ..."

    # Initialize frame buffer for liveness check
    frame_buffer = collections.deque(maxlen=LIVENESS_BUFFER_SIZE)

    # Function to handle window close event
    def handle_close(evt):
        global running
        print("Close button pressed. Exiting...")
        running = False

    # Connect the close event
    fig.canvas.mpl_connect('close_event', handle_close)

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Make a copy for the buffer *before* any drawing might occur
        frame_copy = frame.copy()
        frame_buffer.append(frame_copy)

        # --- Frame Processing ---
        # Process the *original* frame
        annotations = process_frame_for_recognition(
            frame, frame_buffer, yolo, saved_encodings, DATASET_PATH, VERIFICATION_COUNT, THRESHOLD
        )

        # --- Drawing Annotations ---
        # Clear previous drawings from axes
        for item in ax.patches + ax.texts:
            item.remove()

        # Update the image data (display the frame *before* annotations are drawn on it)
        img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw new annotations
        for ann in annotations:
            x1, y1, x2, y2 = map(int, ann['bbox'])
            identity = ann['identity']
            is_live = ann.get('live', True)  # Default to live if key missing

            # Determine color based on identity and liveness
            if not is_live and LIVENESS_ENABLED:
                color = 'magenta'  # Special color for spoof
                identity = "Spoof Detected"  # Override text
            elif 'Unknown' in identity or 'Fail' in identity or 'Processing' in identity:
                color = 'red'
            else:  # Known identity and passed liveness (or liveness disabled)
                color = 'lime'

            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Draw label
            ax.text(x1, y1 - 10, identity, color='white', fontsize=9, bbox=dict(facecolor=color, alpha=0.7, pad=1))

        # --- Calculate and Display FPS ---
        frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed >= 1.0:  # Update FPS every second
            fps = frame_count / elapsed
            fps_text = f"FPS: {fps:.2f}"
            prev_time = curr_time
            frame_count = 0

        # Display FPS on the plot
        ax.text(5, 25, fps_text, color='cyan', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        # --- Update Matplotlib Display ---
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # plt.pause(0.001) # Minimal pause for GUI responsiveness
        except Exception as e:
            if "FigureManager base class has been destroyed" in str(e) or not plt.fignum_exists(fig.number):
                print("Plot window closed. Exiting...")
                running = False
            else:
                print(f"Error during plot update: {e}")
                # Decide if the error is fatal
                # running = False # Exit on plot errors? Maybe too strict.
                # Let's try to continue if possible, but log the error.

    # --- Cleanup ---
    print("[INFO] Releasing resources...")
    cap.release()
    plt.ioff()  # Turn off interactive mode
    # Don't explicitly close if handle_close is working, it might cause errors.
    # plt.close(fig)
    print("[INFO] Exiting.")
