import base64
import io
import time
import re
from datetime import datetime
import collections
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import socket
import json
import os
import glob

# --- Import the recognizer module ---
try:
    import recognizer
except ImportError:
    print("\n[ERROR] Failed to import recognizer.py.")
    exit()
except Exception as e:
    print(f"\n[ERROR] Error importing recognizer.py: {e}")
    exit()

print("[INFO] Recognizer module imported successfully.")

app = Flask(__name__)

# --- Global State (for Liveness Buffer) ---
frame_buffer = collections.deque(maxlen=recognizer.LIVENESS_BUFFER_SIZE)
print(f"[INFO] Initialized global frame buffer with maxlen={recognizer.LIVENESS_BUFFER_SIZE}")

# --- Cooldown Tracker ---
recent_predictions = {}  # { name: last_prediction_timestamp }
PREDICTION_COOLDOWN_SECONDS = 120  # 2 minutes

# --- Helper: Clear Snapshots Folder ---
def clear_snapshots_folder(folder_path="./static/snaps/"):
    files = glob.glob(os.path.join(folder_path, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[ERROR] Could not delete {f}: {e}")

# --- Helper: Parse Recognizer Identity String ---
def parse_identity_string(identity_str):
    name = "Unknown"
    distance = None
    if identity_str == "Spoof Detected":
        name = "Spoof Detected"
    elif "Unknown" in identity_str or "Fail" in identity_str or "Processing" in identity_str:
        name = identity_str
        match = re.search(r'\(([\d.]+)\)', identity_str)
        if match:
            try:
                distance = float(match.group(1))
            except ValueError:
                pass
    else:
        match = re.search(r'^(.*?)\s*\(([\d.]+)\)$', identity_str)
        if match:
            name = match.group(1).strip()
            try:
                distance = float(match.group(2))
            except ValueError:
                name = identity_str
                distance = None
        else:
            name = identity_str
    return name, distance

# --- Helper: Send to Dashboard (Async Optional Later) ---
def send_to_dashboard(data):
    try:
        print(f"[TCP] Sending to dashboard: {data}")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6010))
        s.sendall((json.dumps(data) + "\n").encode('utf-8'))
        s.close()
    except Exception as e:
        print(f"[TCP ERROR] {e}")

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detectface', methods=['POST'])
def detect_face_api():
    start_req_time = time.time()
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[WARN] Could not decode image.")
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        print(f"[ERROR] Decoding error: {e}")
        return jsonify({"error": f"Decoding error: {e}"}), 400

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    frame_buffer.append(frame.copy())

    # --- Predict ---
    process_start_time = time.time()
    try:
        annotations = recognizer.process_frame_for_recognition(
            frame,
            frame_buffer,
            recognizer.yolo,
            recognizer.saved_encodings,
            recognizer.DATASET_PATH,
            recognizer.VERIFICATION_COUNT,
            recognizer.THRESHOLD
        )
    except Exception as e:
        print(f"[ERROR] Recognition error: {e}")
        return jsonify({
            "name": "Error",
            "distance": "N/A",
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "liveness": "N/A",
            "message": f"Recognition error: {e}"
        }), 500

    process_end_time = time.time()
    processing_duration = process_end_time - process_start_time
    total_duration = process_end_time - start_req_time

    result = {
        "name": "N/A",
        "distance": "N/A",
        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "liveness": "N/A",
        "message": "No face detected."
    }

    # --- Handle Predictions ---
    if annotations:
        first_ann = annotations[0]
        identity_str = first_ann['identity']
        is_live = first_ann.get('live', None)

        name, distance = parse_identity_string(identity_str)

        liveness_status = "N/A"
        if recognizer.LIVENESS_ENABLED:
            if is_live is True:
                liveness_status = "Live"
            elif is_live is False:
                liveness_status = "Failed (Spoof?)"
                if name != "Spoof Detected":
                    name = f"{name} (Liveness Failed)"
            else:
                liveness_status = "Checking..."
        else:
            liveness_status = "Disabled"

        now = time.time()

        # Auto-expire old cooldowns
        expired_names = [k for k, v in recent_predictions.items() if now - v > PREDICTION_COOLDOWN_SECONDS]
        for k in expired_names:
            del recent_predictions[k]

        # Check cooldown
        if name != "Unknown" and name != "Spoof Detected":
            if name in recent_predictions:
                elapsed = now - recent_predictions[name]
                if elapsed < PREDICTION_COOLDOWN_SECONDS:
                    print(f"[INFO] Skipping {name} (recently predicted {elapsed:.1f}s ago)")
                    return jsonify({
                        "name": name,
                        "distance": "N/A",
                        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        "liveness": "Recently Predicted",
                        "message": f"Skipping duplicate prediction for {name}. Cooldown {int(PREDICTION_COOLDOWN_SECONDS - elapsed)}s left."
                    })

            # Update prediction timestamp
            recent_predictions[name] = now

            # After successful prediction: clear buffer + snapshots
            frame_buffer.clear()
            clear_snapshots_folder()

            print(f"[INFO] Cleared frame buffer and snapshots after recognizing {name}.")

        result = {
            "name": name,
            "distance": f"{distance:.2f}" if distance is not None else "N/A",
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "liveness": liveness_status,
            "message": f"Processed '{identity_str}'. Liveness: {liveness_status}. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"
        }
        send_to_dashboard({
            "name": result["name"],
            "distance": result["distance"],
            "time": result["time"],
            "liveness": result["liveness"]
        })

    else:
        # No faces detected, but handle liveness if possible
        liveness_status = "N/A"
        background_spoof = False
        if recognizer.LIVENESS_ENABLED and len(frame_buffer) == recognizer.LIVENESS_BUFFER_SIZE:
            try:
                is_live = recognizer.detect_liveness(list(frame_buffer))
                if is_live is False:
                    liveness_status = "Failed (Spoof?)"
                    background_spoof = True
                elif is_live is True:
                    liveness_status = "Live"
            except Exception as e:
                print(f"[ERROR] Background liveness error: {e}")
                liveness_status = "Error"

        result["liveness"] = liveness_status
        if background_spoof:
            result["name"] = "Spoof Detected (Background)"
            result["message"] = f"Background spoof detected. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"
        else:
            result["message"] = f"No face detected. Liveness: {liveness_status}. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"

    return jsonify(result)

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("[APP] Face Detector Active.")
    app.run(port=5001, debug=False)
