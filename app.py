from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
import cv2
import numpy as np
import os


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("HAAR EMPTY:", face_cascade.empty())
# -------------------- APP SETUP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_key'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

camera = None
camera_active = False

# -------------------- LOGIN MANAGER --------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -------------------- MONGODB --------------------
client = MongoClient(
    "mongodb+srv://rahulhattinagre:Rahul123@cluster0.2bp4aoh.mongodb.net/",
    serverSelectionTimeoutMS=5000
)
db = client["age_gender_detector"]
users = db["users"]

# -------------------- USER CLASS --------------------
class User(UserMixin):
    def __init__(self, user):
        self.id = str(user["_id"])
        self.username = user["username"]
        self.email = user["email"]

# -------------------- LOAD USER --------------------
@login_manager.user_loader
def load_user(user_id):
    user = users.find_one({'_id': ObjectId(user_id)})
    return User(user) if user else None

# -------------------- LOAD MODELS --------------------
face_net = cv2.dnn.readNet(
    os.path.join(BASE_DIR, "models/face_detector/deploy.prototxt"),
    os.path.join(BASE_DIR, "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
)

age_net = cv2.dnn.readNet(
    os.path.join(BASE_DIR, "models/age/age_deploy.prototxt"),
    os.path.join(BASE_DIR, "models/age/age_net.caffemodel")
)

gender_net = cv2.dnn.readNet(
    os.path.join(BASE_DIR, "models/gender/gender_deploy.prototxt"),
    os.path.join(BASE_DIR, "models/gender/gender_net.caffemodel")
)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# -------------------- FRAME GENERATOR --------------------
def gen_frames():
    global camera, camera_active

    while camera_active and camera:
        success, frame = camera.read()
        if not success:
            continue

        h, w = frame.shape[:2]

        # -------- FACE DETECTION (HAAR) --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, fw, fh) in faces:
            # 🔹 Crop face safely
            face = frame[y:y+fh, x:x+fw]

            if face.size == 0:
                continue

            # -------- AGE & GENDER --------
            face_blob = cv2.dnn.blobFromImage(
                face,
                1.0,
                (227, 227),
                MODEL_MEAN_VALUES,
                swapRB=False
            )

            gender_net.setInput(face_blob)
            gender = GENDER_LIST[gender_net.forward().argmax()]

            age_net.setInput(face_blob)
            age = AGE_LIST[age_net.forward().argmax()]

            label = f"{gender}, {age}"

            # -------- DRAW --------
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-30), (x+fw, y), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            break  # ✅ ONLY ONE FACE

        # -------- STREAM FRAME --------
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if users.find_one({"$or": [{"username": username}, {"email": email}]}):
            flash("User already exists")
            return redirect(url_for('login'))

        users.insert_one({
            "username": username,
            "email": email,
            "password": password
        })

        flash("Signup successful. Please login.")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = users.find_one({'email': request.form['email']})
        if user and check_password_hash(user['password'], request.form['password']):
            login_user(User(user))
            return redirect(url_for('detector'))
        flash("Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/detector')
@login_required
def detector():
    return render_template('detector.html')

# -------------------- CAMERA ROUTES --------------------
@app.route('/start_camera', methods=['POST'])
@login_required
def start_camera():
    global camera, camera_active

    if camera_active:
        return ('', 204)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Camera failed to open")
        camera = None
        return ("Camera error", 500)

    camera_active = True
    print("📷 Camera opened")
    return ('', 204)
@app.route('/video_feed')
@login_required
def video_feed():
    if not camera_active:
        return "", 204

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
    )
@app.route('/stop_camera', methods=['POST'])
@login_required
def stop_camera():
    global camera, camera_active

    camera_active = False

    if camera:
        camera.release()
        camera = None
        print("📷 Camera released")

    return ('', 204)

@app.route('/profile')
@login_required
def profile():
    return "Profile Page"

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        threaded=True
    )
