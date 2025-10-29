import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import base64
import os
from werkzeug.security import generate_password_hash, check_password_hash # Core for secure passwords

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key' # KEEP THIS!

# --- Flask-Login Setup and User Management (In-Memory Store) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple User class that inherits from UserMixin
class User(UserMixin):
    def __init__(self, user_id, username, email, password_hash):
        self.id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        
    def get_id(self):
        return str(self.id)

# Dictionary to simulate a database. We'll store User objects.
USERS = {} 
USER_ID_COUNTER = 1

# --- User Lookup Helpers ---

def get_user_by_id(user_id):
    """Finds user by Flask-Login ID (which is the User.id int)"""
    for user in USERS.values():
        if str(user.id) == user_id:
            return user
    return None

def find_user(identifier):
    """Finds user by username or email"""
    # 1. Try finding by username (stored as the key)
    if identifier in USERS:
        return USERS[identifier]
    
    # 2. Try finding by email
    for user in USERS.values():
        if user.email == identifier:
            return user
    return None

@login_manager.user_loader
def load_user(user_id):
    """Loads user object based on ID stored in session"""
    return get_user_by_id(user_id)


# --- Age and Gender Detection Core Logic (No changes needed here) ---

# Define model paths and constants
MODEL_DIR = 'models/'
FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load the networks
try:
    faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
    ageNet = cv2.dnn.readNet(AGE_PROTO, AGE_MODEL)
    genderNet = cv2.dnn.readNet(GENDER_PROTO, GENDER_MODEL)
except cv2.error as e:
    print(f"Error loading models: {e}")
    # Handle error or exit application if models aren't found

# ... (detect_age_gender function is fine) ...
def detect_age_gender(frame):
    """Performs face detection, then age and gender prediction."""
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Create a 4D blob from the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    results = []
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7: # Face detection confidence threshold
            # Get face bounding box
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            
            # Crop the face
            face = frame[max(0, y1):min(y2, frame_height - 1), max(0, x1):min(x2, frame_width - 1)]
            if face.size == 0: continue # Skip empty faces
            
            # Prepare blob for age/gender prediction (227x227)
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict Gender
            genderNet.setInput(face_blob)
            genderPreds = genderNet.forward()
            gender = GENDER_LIST[genderPreds[0].argmax()]
            
            # Predict Age
            ageNet.setInput(face_blob)
            agePreds = ageNet.forward()
            age = AGE_LIST[agePreds[0].argmax()]
            
            # Append result
            results.append({
                'box': [x1, y1, x2, y2],
                'gender': gender,
                'age': age
            })

    return results

# --- Flask Routes (URLs) ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('detector'))
        
    if request.method == 'POST':
        global USER_ID_COUNTER
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check for existing user
        if find_user(username) or find_user(email):
            flash('Username or Email already exists!', 'danger')
            return render_template('signup.html')
        
        # Hash password and create user
        hashed_password = generate_password_hash(password)
        
        new_user = User(
            user_id=USER_ID_COUNTER,
            username=username,
            email=email,
            password_hash=hashed_password
        )
        USERS[username] = new_user # Store by username as key
        USER_ID_COUNTER += 1

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('detector'))

    if request.method == 'POST':
        # Field name from the HTML form (must match the login.html I provided)
        identifier = request.form.get('username_or_email') 
        password = request.form.get('password')

        # Find user by either username or email
        user = find_user(identifier)
        
        # Check if user exists AND if password matches the hash
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('detector'))
        else:
            # Note: Changed to 'Admin ID' to match the image you liked
            flash('Invalid Admin ID or password', 'danger') 
            return render_template('login.html')

    return render_template('login.html')

# ... (rest of the routes are fine) ...

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/detector')
@login_required
def detector():
    # current_user.username is now available on the template
    return render_template('detector.html', username=current_user.username) 

@app.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    # Receive the image data (base64 encoded string from JS)
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    
    # Decode base64 to OpenCV image format
    np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform detection
    results = detect_age_gender(frame)
    
    return jsonify({'results': results})

if __name__ == '__main__':
    # Create the models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}. Please add your model files here.")
        
    app.run(debug=True)