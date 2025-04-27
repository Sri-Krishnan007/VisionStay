package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
	"strings"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

// =======================
// Structs
// =======================

type Student struct {
	Name        string    `json:"name"`
	RollNo      string    `json:"roll_no"`
	RoomNo      string    `json:"room_no"`
	Class       string    `json:"class"`
	IsHosteller bool      `json:"is_hosteller"`
	IsBlacklist bool      `json:"is_blacklist"`
	Attendance  []LogItem `json:"attendance"`
}

type LogItem struct {
	Timestamp time.Time `json:"timestamp"`
	Status    string    `json:"status"` // ✅ add this
}


// =======================
// Global Mongo Client
// =======================

var studentCollection *mongo.Collection

func initMongo() {
	uri := "#your uri"

	clientOpts := options.Client().ApplyURI(uri)

	client, err := mongo.Connect(clientOpts) // ✅ Only clientOpts passed
	if err != nil {
		log.Fatal("❌ Mongo connection failed: ", err)
	}

	studentCollection = client.Database("visionstay").Collection("students")
	fmt.Println("✅ MongoDB connection established")
}

// =======================
// Handlers
// =======================

func createStudent(w http.ResponseWriter, r *http.Request) {
	var student Student
	_ = json.NewDecoder(r.Body).Decode(&student)
	student.Attendance = []LogItem{}

	_, err := studentCollection.InsertOne(context.TODO(), student)
	if err != nil {
		http.Error(w, "Failed to insert student", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	fmt.Fprint(w, "Student created")
}

func getAllStudents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json") // ✅ ADD THIS LINE

	cursor, err := studentCollection.Find(context.TODO(), bson.D{})
	if err != nil {
		log.Println("❌ Failed to fetch students:", err) // Optional debug
		http.Error(w, "Failed to fetch students", http.StatusInternalServerError)
		return
	}
	defer cursor.Close(context.TODO())

	var students []Student
	if err := cursor.All(context.TODO(), &students); err != nil {
		log.Println("❌ Cursor decoding error:", err) // Optional debug
		http.Error(w, "Cursor error", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(students)
}

func updateStudent(w http.ResponseWriter, r *http.Request) {
	var student Student
	err := json.NewDecoder(r.Body).Decode(&student)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	student.RollNo = strings.TrimSpace(student.RollNo)

	filter := bson.M{"rollno": student.RollNo}

	// Build the $set fields
	setFields := bson.M{
		"name":        student.Name,
		"class":       student.Class,
		"ishosteller": student.IsHosteller, // ✅ update old field directly
		"is_blacklist": student.IsBlacklist,
	}

	// Handle room_no according to hosteller status
	if student.IsHosteller {
		setFields["roomno"] = student.RoomNo // ✅ room number as entered
	} else {
		setFields["roomno"] = "-" // ✅ set to hyphen when not hosteller
	}

	update := bson.M{
		"$set": setFields,
	}

	// Update in MongoDB
	result, err := studentCollection.UpdateOne(context.TODO(), filter, update)
	if err != nil {
		http.Error(w, "Update failed", http.StatusInternalServerError)
		return
	}

	// Debugging outputs
	fmt.Printf("MatchedCount: %d, ModifiedCount: %d\n", result.MatchedCount, result.ModifiedCount)
	if result.MatchedCount == 0 {
		fmt.Println("⚠️ No document matched for RollNo:", student.RollNo)
	}

	fmt.Fprint(w, "Student updated")
}


func deleteStudent(w http.ResponseWriter, r *http.Request) {
	var student Student
	_ = json.NewDecoder(r.Body).Decode(&student)

	_, err := studentCollection.DeleteOne(context.TODO(), bson.M{"roll_no": student.RollNo})
	if err != nil {
		http.Error(w, "Delete failed", http.StatusInternalServerError)
		return
	}

	fmt.Fprint(w, "Student deleted")
}

type AttendancePayload struct {
	RollNo string    `json:"roll_no"`
	Entry  LogItem   `json:"entry"`
}


func addAttendance(w http.ResponseWriter, r *http.Request) {
	var payload AttendancePayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid payload", http.StatusBadRequest)
		return
	}

	// Fix: Match actual MongoDB field
	filter := bson.M{"rollno": payload.RollNo}
	update := bson.M{"$push": bson.M{
		"attendance": payload.Entry,
	}}

	res, err := studentCollection.UpdateOne(context.TODO(), filter, update)
	if err != nil {
		log.Println("❌ Failed to update attendance:", err)
		http.Error(w, "Failed to update attendance", http.StatusInternalServerError)
		return
	}

	if res.MatchedCount == 0 {
		log.Println("⚠️ No student matched for roll number:", payload.RollNo)
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, "Attendance updated")
}


// =======================
// Main Entry
// =======================

func main() {
	initMongo()

	http.HandleFunc("/create", createStudent)
	http.HandleFunc("/get", getAllStudents)
	http.HandleFunc("/update", updateStudent)
	http.HandleFunc("/delete", deleteStudent)
	http.HandleFunc("/attendance", addAttendance)

	

	fmt.Println("🚀 Server started on port 8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
