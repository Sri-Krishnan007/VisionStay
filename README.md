# 🚀 VisionStay Project

A complete face recognition–based attendance and behavior analysis system built using **Python**, **Go**, **MongoDB**, **Flask**, **HTML/CSS/JS**, and **YOLO**.

---

## 📦 Technology Stack

- **Python**: Model, Face Recognition, API Servers
- **Go**: Backend (Attendance and Student Management API)
- **MongoDB**: Database
- **HTML/CSS/JS**: Frontend (Dashboard)
- **YOLO**: Object Detection
- **face_recognition**: Facial feature embedding

---

## ⚙️ Project Structure

| Folder | Purpose |
|:---|:---|
| `model/` | Face recognition model server |
| `frontend_python/` | Frontend dashboard app server |
| `backend-go/` | Go backend connected to MongoDB |

---

## 🛠 Setup Instructions

### 1. 📥 Install Python Requirements


pip install -r requirements.txt
(Install all Python dependencies.)

## 2. 🛠 Preparing the Dataset
Create a folder called newdata/ inside the model/ directory.

Inside newdata/, create subfolders named after each person.

Each person's folder should contain at least 5 portrait images.

Example:

Copy
Edit
newdata/
├── John/
│    ├── img1.jpg
│    ├── img2.jpg
├── Alice/
│    ├── img1.jpg
│    ├── img2.jpg

## 3. 🏗 Model Embedding Creation
Run inside model/:

python new_emmbedding.py

✅ This will generate the .pkl file for face embeddings.

## 4. 🎯 Starting the Model Prediction Server
Inside model/:

python app.py
recognizer.py will load models and dataset.

Face prediction will use YOLO + face_recognition.

## 5. 🖥 Running the Go Backend
Inside backend-go/:

Configure your MongoDB URI inside main.go.

Then run:

go run main.go
✅ The backend Go server connects to MongoDB to manage students, attendance, behavior data.

## 6. 📬 Configuring Frontend Python App
Inside frontend_python/:

Update your admin email and password settings.

Then run:

python app.py

# ✅ This frontend dashboard will:

# Receive face prediction data via TCP from model

# Manage login, view attendance, add/update students

# Send alert mails when day scholars or blacklisted students enter

# Behavior analysis using Groq API (only attendance data)

## 🔥 How the System Works
model/app.py predicts faces live from camera using YOLO + face_recognition.

Prediction data is sent via TCP to frontend_python/app.py.

#  Frontend dashboard (admin login) can:

✅ Log Attendance (In/Out/Nil)

✅ Add, Update, View Student Details

✅ Auto-email alerts for blacklist violations

✅ Run Behavior Analysis (questions to Groq based on attendance only)

#  📊 Dashboard Features

Admin Login

Student Management (Add/View/Update)

Attendance Logging

Behavior Analysis (Smart Chatbot for student behavior)

Email Notifications (Blacklist/Day Scholar)

## 📈 Future Improvements
📸 Multiple camera streams

📋 Advanced attendance reports

🛡️ Security hardening for production deployment

## 🙌 Contributing
PRs are welcome. Open issues if you find bugs!

## ⚡ Quick Start Commands

# 1. Setup Python
pip install -r requirements.txt

# 2. Create 'newdata/' and place images

# 3. Generate embeddings
python model/new_emmbedding.py

# 4. Run servers
go run backend-go/main.go
python frontend_python/app.py
python model/app.py


## 🛡 License
MIT License - feel free to use, modify and share!

