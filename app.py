from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import base64

app = Flask(__name__)

# ==========================================
# 🌩️ MONGODB CLOUD CONNECTION
# ==========================================
MONGO_URI = "mongodb+srv://robokalamshivasri_db_user:uVjzpyjP2a9fklww@faceattendance.b7km6n1.mongodb.net/?retryWrites=true&w=majority&appName=FaceAttendance"
client = MongoClient(MONGO_URI)
db = client['face_attendance_db']
users_col = db['users'] 
logs_col = db['logs']   

# 1. Load Faces from DB
def load_known_faces():
    known_encodings = []
    known_names = []
    for user in users_col.find():
        known_encodings.append(np.array(user['encoding']))
        known_names.append(user['roll'])
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()

# 2. STRICT 2-PUNCH LOGIC (IN & OUT ONLY)
def get_attendance_status(roll):
    today = datetime.now().strftime('%d-%m-%Y')
    record = logs_col.find_one({"roll": roll, "date": today})
    
    # 1st Scan: Record lekapothe IN Time padali
    if not record: 
        return "new"
    
    # 2nd Scan: Record undi, kani OUT time inka padaledu ("-")
    if record.get("out_time") == "-": 
        return "update_out"
    
    # 3rd Scan: Record undi, OUT time kuda padipoindi (Locked)
    return "completed"

# 3. Mark IN or OUT Attendance
def mark_attendance(roll):
    today = datetime.now().strftime('%d-%m-%Y')
    now_time = datetime.now().strftime("%H:%M:%S")
    status = get_attendance_status(roll)
    
    if status == "new":
        logs_col.insert_one({"roll": roll, "in_time": now_time, "out_time": "-", "date": today})
        print(f"✅ IN TIME Logged: {roll}")
    elif status == "update_out":
        logs_col.update_one({"roll": roll, "date": today}, {"$set": {"out_time": now_time}})
        print(f"✅ OUT TIME Updated: {roll}")

# Helper: Convert Base64 from Browser to OpenCV Image
def decode_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# --- HTML ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', page='home')

@app.route('/register')
def register():
    return render_template('index.html', page='register')

@app.route('/register_cam_page', methods=['POST'])
def register_cam_page():
    roll = request.form['roll'].upper()
    return render_template('index.html', page='register_cam', roll=roll)

@app.route('/mark_attendance_page')
def mark_attendance_page():
    return render_template('index.html', page='enter_id')

@app.route('/mark_attendance_cam_page', methods=['POST'])
def mark_attendance_cam_page():
    sid = request.form['student_id'].upper()
    status = get_attendance_status(sid)
    
    # Oka vela already 2 times scan ayipothe, direct ga block chesesthundi
    if status == "completed":
        return render_template('index.html', page='already_marked', roll=sid)
        
    return render_template('index.html', page='attendance_cam', sid=sid)

@app.route('/attendance_success')
def attendance_success():
    return render_template('index.html', page='success')

@app.route('/view_attendance')
def view_attendance():
    records = list(logs_col.find({}, {"_id": 0})) 
    data = [[r['roll'], r.get('in_time', '-'), r.get('out_time', '-'), r['date']] for r in records]
    return render_template('index.html', page='logs', attendance_data=data)

@app.route('/delete_attendance/<roll>')
def delete_attendance(roll):
    today = datetime.now().strftime('%d-%m-%Y')
    logs_col.delete_one({"roll": roll.upper(), "date": today})
    return redirect(url_for('view_attendance'))

# --- API ROUTES (FOR BROWSER JS) ---

@app.route('/api_register', methods=['POST'])
def api_register():
    data = request.json
    roll = data['roll'].upper()
    images = data['images']
    
    encodings = []
    for img_b64 in images:
        img = decode_base64_image(img_b64)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame)
        if face_locs:
            encode = face_recognition.face_encodings(rgb_frame, face_locs)[0]
            encodings.append(encode)

    if encodings:
        avg_encoding = np.mean(encodings, axis=0).tolist()
        users_col.update_one({"roll": roll}, {"$set": {"encoding": avg_encoding}}, upsert=True)
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()
        return jsonify({"status": "success"})
    
    return jsonify({"status": "failed"})

@app.route('/api_recognize', methods=['POST'])
def api_recognize():
    data = request.json
    sid = data['student_id'].upper()
    img = decode_base64_image(data['image'])
    
    rgb_small = cv2.cvtColor(cv2.resize(img, (0,0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
    face_encodes = face_recognition.face_encodings(rgb_small)
    
    if not face_encodes:
        return jsonify({"status": "no_face"})

    for encodeFace in face_encodes:
        if len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, encodeFace, tolerance=0.5)
            faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)
            if len(faceDis) > 0:
                matchIdx = np.argmin(faceDis)
                if matches[matchIdx]:
                    recognized_name = known_face_names[matchIdx].upper()
                    if recognized_name == sid:
                        status = get_attendance_status(sid)
                        if status != "completed":
                            mark_attendance(recognized_name)
                            return jsonify({"status": "success"})
                    else:
                        return jsonify({"status": "mismatch"})
                        
    return jsonify({"status": "unrecognized"})

if __name__ == "__main__":
    app.run(debug=True)