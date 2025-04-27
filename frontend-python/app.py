from flask import Flask, render_template, request, redirect, session, jsonify
import requests
import threading
import socket
import json
import os
from datetime import datetime, timezone, timedelta
from dateutil.parser import isoparse  # pip install python-dateutil
import logging

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Setup logging
logging.basicConfig(filename="tcp_listener_debug.log", level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')


BACKEND_URL = 'http://localhost:8080'  # Replace if your Go server runs elsewhere
face_logs = []
student_cache = {}

def get_students():
    if not student_cache:
        try:
            students = requests.get(f'{BACKEND_URL}/get').json()
            for s in students:
                student_cache[s['roll_no']] = s
        except Exception as e:
            logging.error(f"Fetching student list failed: {e}")
    return list(student_cache.values())
# -------------------- Login --------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == '#Your email' and password == '#adminpass!':
            session['user'] = email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# -------------------- Dashboard --------------------

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')

    students = []
    try:
        res = requests.get(f"{BACKEND_URL}/get")
        if res.status_code == 200:
            students = res.json()
    except:
        pass

    return render_template('dashboard.html', students=students)


# -------------------- CRUD Endpoints --------------------
@app.route('/student-details')
def student_details():
    if 'user' not in session:
        return redirect('/')

    students = []
    try:
        response = requests.get(f'{BACKEND_URL}/get')
        if response.status_code == 200:
            students = response.json()
        else:
            print("❌ Backend returned non-200 status:", response.status_code)
    except Exception as e:
        print("❌ Error fetching students:", e)

    return render_template('studentdetails.html', students=students)

@app.route('/create', methods=['POST'])
def create_student():
    data = {
        "name": request.form['name'],
        "roll_no": request.form['roll_no'],
        "room_no": request.form.get('room_no', ''),
        "class": request.form['class'],
        "is_hosteller": 'is_hosteller' in request.form,
        "is_blacklist": 'is_blacklist' in request.form
    }
    try:
        requests.post(f'{BACKEND_URL}/create', json=data)
    except Exception as e:
        print("❌ Error sending create request:", e)
    return redirect('/student-details')

@app.route('/update', methods=['POST'])
def update_student():
    try:
        data = request.get_json()  # not request.form!
        print("Received data in Flask:", data)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        payload = {
            "name": data.get('name'),
            "roll_no": data.get('roll_no'),
            "room_no": data.get('room_no', ''),
            "class": data.get('class'),
            "is_hosteller": data.get('is_hosteller', False),
            "is_blacklist": data.get('is_blacklist', False)
        }

        # Send JSON payload to Go backend
        requests.post(f'{BACKEND_URL}/update', json=payload)

        return "Student updated", 200

    except Exception as e:
        print("❌ Error while updating student:", e)
        return jsonify({"error": str(e)}), 500



@app.route('/delete', methods=['POST'])
def delete_student():
    data = {
        "roll_no": request.form['roll_no']
    }
    try:
        requests.post(f'{BACKEND_URL}/delete', json=data)
    except Exception as e:
        print("❌ Error sending delete request:", e)
    return redirect('/student-details')

# -------------------- Detected Face --------------------

@app.route('/detectedfaces')
def detected_faces():
    if 'user' not in session:
        return redirect('/')
    return render_template('detectedfaces.html', faces=face_logs)


from datetime import datetime, timezone  # make sure timezone is imported
from flask_mail import Mail, Message

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '#your email'
app.config['MAIL_PASSWORD'] = '#your pass'

mail = Mail(app)

from flask_mail import Message
import os

ALERTED_ROLLS_FILE = "alerted_rolls.txt"
alerted_rolls = set()

# 🧠 Load already-alerted roll numbers into memory at startup
if os.path.exists(ALERTED_ROLLS_FILE):
    with open(ALERTED_ROLLS_FILE, "r") as f:
        alerted_rolls = set(line.strip() for line in f.readlines())

def send_alert_email(roll_no, timestamp):
    if roll_no in alerted_rolls:
        print(f"[EMAIL] Already alerted {roll_no}, skipping email.")
        return False

    recipient = f"{roll_no}@cit.edu.in"
    msg = Message(
        subject="🚨 Unauthorized Hostel Entry",
        sender=app.config['MAIL_USERNAME'],
        recipients=[recipient]
    )
    msg.body = (
        f"Dear Student,\n\n"
        f"You entered the hostel without valid authorization.\n"
        f"Time of detection: {timestamp}\n\n"
        f"If this is a mistake, please contact the warden immediately.\n\n"
        f"Regards,\nHostel Security Team"
    )

    try:
        with app.app_context():
            mail.send(msg)

        # ✅ Save this roll number to both memory and file
        alerted_rolls.add(roll_no)
        with open(ALERTED_ROLLS_FILE, "a") as f:
            f.write(f"{roll_no}\n")

        print(f"[EMAIL] Alert sent to {recipient}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send alert to {recipient}: {e}")
        return False




def tcp_listener():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 6010))
    server.listen(5)
    print("[TCP] Listening for incoming detections...")

    while True:
        client, _ = server.accept()
        with client:
            data = b""
            while True:
                part = client.recv(1024)
                if not part:
                    break
                data += part

            try:
                messages = data.decode('utf-8').strip().split('\n')
                for msg in messages:
                    detection = json.loads(msg)
                    face_logs.append(detection)

                    name = detection["name"]
                    detected_time_str = detection["time"]

                    # Combine today's date with parsed time (fixing 2-day bug)
                    now = datetime.now(timezone.utc)
                    parsed_time = datetime.strptime(detected_time_str, "%H:%M:%S.%f").time()
                    detection_time = datetime.combine(now.date(), parsed_time).replace(tzinfo=timezone.utc)

                    clean_name = name.split("(")[0].strip().lower()
                    students = get_students()
                    student = next((s for s in students if s['name'].strip().lower() == clean_name), None)

                    if not student:
                        logging.warning(f"No matching student for {name}")
                        continue

                    roll_no = student['roll_no']
                    logs = student.get('attendance', [])

                    status = "IN"
                    if logs:
                        last_entry = logs[-1]
                        last_time = isoparse(last_entry['timestamp'])
                        last_status = last_entry.get('status')

                        time_diff = (detection_time - last_time).total_seconds()

                        if last_status == "IN" and time_diff <= 120:
                            logging.info(f"[SKIPPED] Duplicate IN within 2 min ({time_diff:.1f}s)")
                            continue
                        if last_status == "OUT" and time_diff <= 120:
                            logging.info(f"[SKIPPED] Duplicate OUT within 2 min ({time_diff:.1f}s)")
                            continue

    # Toggle the status logically
                        status = "OUT" if last_status == "IN" else "IN"
                    else:
                        status = "IN"

                    iso_timestamp = datetime.now(timezone.utc).isoformat()

                    if student.get("is_blacklist") or not student.get("is_hosteller"):
                        email_status = send_alert_email(roll_no, iso_timestamp)
                        if not email_status:
                            logging.warning(f"Email alert failed for {roll_no}")

                    post_attendance(roll_no, iso_timestamp, status)
                    logging.info(f"[LOGGED] {status} for {roll_no} at {iso_timestamp}")

            except Exception as e:
                logging.exception(f"TCP message parsing failed: {e}")
                with open("tcp_fallback.log", "a") as f:
                    f.write(f"\nRaw Data: {data.decode(errors='replace')}\n")


# -------------------- Post Attendance  --------------------

from datetime import datetime, timezone

def post_attendance(roll_no, timestamp, status):
    payload = {
        "roll_no": roll_no,
        "entry": {
            "timestamp": timestamp,
            "status": status
        }
    }
    logging.debug(f"POST payload: {payload}")

    try:
        res = requests.post(f"{BACKEND_URL}/attendance", json=payload)
        logging.info(f"Attendance posted: {res.status_code}")
    except Exception as e:
        logging.error(f"Attendance post failed: {e}")
        
        
# --------------------View Attendance  --------------------        
@app.route('/attendance')
def view_attendance():
    if 'user' not in session:
        return redirect('/')

    students = []
    try:
        response = requests.get(f'{BACKEND_URL}/get')
        if response.status_code == 200:
            students = response.json()
           
    except Exception as e:
        print("❌ Error fetching attendance:", e)

    return render_template('attendance.html', students=students)


        
@app.route('/student-behaviour/<roll_no>', methods=['GET'])
def student_behaviour(roll_no):
    if 'user' not in session:
        return redirect('/')

    try:
        response = requests.get(f'{BACKEND_URL}/get')
        students = response.json()
        print("[DEBUG] Fetched students:", students[:2])  # Limit to first 2 for readability
        print("[DEBUG] Looking for roll_no:", roll_no)

        student = next((s for s in students if s.get('roll_no') == roll_no), None)

    except Exception as e:
        print(f"❌ Error fetching student: {e}")
        student = None

    return render_template('personalbehaviour.html', student=student, roll_no=roll_no)

        
        
import requests as httpreq
@app.route('/ask-groq', methods=['POST'])
def ask_groq():
    data = request.get_json()
    roll_no = data['roll_no']
    user_query = data['query']

    students = requests.get(f"{BACKEND_URL}/get").json()
    student = next((s for s in students if s.get('roll_no') == roll_no), None)


    if not student:
        return jsonify({"reply": "Student not found."})

    attendance_data = student.get("attendance", [])

    prompt = f"""
You are a strict attendance analysis bot. Only use the attendance data below to answer.

College working hours:
- Weekdays: 9:00–10:55, 11:10–12:55, 14:00–16:45
- Late night: any entry after 21:00
- Weekends: generally off

Here is the student's attendance:
{json.dumps(attendance_data, indent=2)}

Now answer this question using only that data:
{user_query}
"""

    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    groq_payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=groq_payload)
        reply = res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[Groq ERROR] {e}")
        reply = "There was an error contacting the AI."

    return jsonify({"reply": reply})

        
        

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("[MAIN] Starting TCP listener thread...")
        threading.Thread(target=tcp_listener, daemon=True).start()

    app.run(port=5000, debug=True)
