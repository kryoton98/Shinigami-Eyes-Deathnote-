import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import random
import urllib.request
import os
import datetime
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template_string, jsonify, Response

app = Flask(__name__)

# Ensure required directories exist
os.makedirs("captures", exist_ok=True)
os.makedirs("static", exist_ok=True)

# --- FONT AUTO-DOWNLOADER ---
FONT_PATH = "custom_font.ttf"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/creepster/Creepster-Regular.ttf"

if not os.path.exists(FONT_PATH):
    print("[*] Downloading frenzy font for the first time...")
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    except Exception:
        pass

try:
    FONT_NAME = ImageFont.truetype(FONT_PATH, 45)
    FONT_LIFE = ImageFont.truetype(FONT_PATH, 35)
except IOError:
    FONT_NAME = ImageFont.load_default()
    FONT_LIFE = ImageFont.load_default()

# --- THE SHINIGAMI MEMORY BANK ---
known_encodings = []
known_profiles = [] 
global_raw_frame = None
global_filtered_frame = None 

MALE_NAMES = ["LIGHT Y.", "L. LAWLIET", "RYUK", "NATE R.", "MIHAEL K.", "TERU M.", "SOICHIRO Y.", "MATSUDA"]
FEMALE_NAMES = ["MISA A.", "REM", "KIYOMI T."]
ALL_NAMES = MALE_NAMES + FEMALE_NAMES

def get_unique_name():
    """Ensures no two people get the exact same name."""
    assigned_names = [profile["name"] for profile in known_profiles]
    available_names = [name for name in ALL_NAMES if name not in assigned_names]
    
    if available_names:
        return random.choice(available_names)
    else:
        return f"{random.choice(ALL_NAMES)} - {random.randint(10, 99)}"

def generate_shinigami_stream():
    global global_raw_frame, global_filtered_frame
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    current_face_locations = []
    current_profiles = []

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        image = cv2.flip(image, 1)
        global_raw_frame = image.copy() # Save clean frame for capture
        
        h, w, _ = image.shape
        glow_mask = np.zeros_like(image)
        
        # 1. Background Face Recognition
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % 10 == 0:
            current_face_locations = face_recognition.face_locations(rgb_small_frame)
            current_encodings = face_recognition.face_encodings(rgb_small_frame, current_face_locations)
            
            current_profiles = []
            for face_encoding in current_encodings:
                match_idx = -1
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        match_idx = np.argmin(face_distances)
                
                if match_idx != -1:
                    current_profiles.append(known_profiles[match_idx])
                else:
                    new_name = get_unique_name()
                    raw_life = str(random.randint(10000000, 99999999))
                    new_life = " ".join(raw_life) 
                    new_profile = {"name": new_name, "life": new_life}
                    
                    known_encodings.append(face_encoding)
                    known_profiles.append(new_profile)
                    current_profiles.append(new_profile)
                    
        frame_count += 1

        # 2. MediaPipe Graphics Tracking
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [362, 385, 386, 263, 374, 380]
                right_eye = [33, 159, 158, 133, 153, 145]
                left_eye_pts = np.array([[int(face_landmarks.landmark[j].x * w), int(face_landmarks.landmark[j].y * h)] for j in left_eye])
                right_eye_pts = np.array([[int(face_landmarks.landmark[j].x * w), int(face_landmarks.landmark[j].y * h)] for j in right_eye])
                cv2.fillPoly(glow_mask, [left_eye_pts], (0, 0, 255))
                cv2.fillPoly(glow_mask, [right_eye_pts], (0, 0, 255))

            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            for face_landmarks in results.multi_face_landmarks:
                head_x = int(face_landmarks.landmark[10].x * w)
                head_y = int(face_landmarks.landmark[10].y * h) - 50
                
                mp_cx = int(face_landmarks.landmark[1].x * (w / 4))
                mp_cy = int(face_landmarks.landmark[1].y * (h / 4))
                
                display_profile = {"name": "???", "life": "0 0 0 0 0 0 0 0"}
                min_dist = float('inf')
                
                for idx, (top, right, bottom, left) in enumerate(current_face_locations):
                    fr_cx, fr_cy = (left + right) // 2, (top + bottom) // 2
                    dist = (mp_cx - fr_cx)**2 + (mp_cy - fr_cy)**2
                    if dist < min_dist and dist < 1500: 
                        min_dist = dist
                        if idx < len(current_profiles):
                            display_profile = current_profiles[idx]

                # Draw neatly stacked text
                name_bbox = draw.textbbox((0, 0), display_profile["name"], font=FONT_NAME)
                life_bbox = draw.textbbox((0, 0), display_profile["life"], font=FONT_LIFE)
                
                draw.text((head_x - ((name_bbox[2] - name_bbox[0]) // 2), head_y - 80), display_profile["name"], font=FONT_NAME, fill=(255, 0, 0, 255))
                draw.text((head_x - ((life_bbox[2] - life_bbox[0]) // 2), head_y - 30), display_profile["life"], font=FONT_LIFE, fill=(220, 0, 0, 255))

            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        glow_mask = cv2.GaussianBlur(glow_mask, (45, 45), 0)
        image = cv2.addWeighted(image, 1.0, glow_mask, 1.5, 0)

        global_filtered_frame = image.copy() # Save filtered frame for capture

        ret, buffer = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- WEB UI INTERFACE ---

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Shinigami Terminal</title>
    <style>
        body { 
            margin: 0; background: #000; overflow: hidden; 
            display: flex; flex-direction: column; height: 100vh; font-family: 'Courier New', monospace; 
        }

        /* --- TOP HEADER --- */
        .header { 
            padding: 15px; background: #0a0a0a; border-bottom: 3px solid #500; text-align: center; z-index: 20; 
        }
        h1 { margin: 0; letter-spacing: 5px; font-size: 2.2em; text-transform: uppercase; color: #f00; }
        .subtitle { color: #888; font-size: 14px; margin-top: 5px; }

        /* --- MAIN SPLIT SCREEN --- */
        .content-wrapper { display: flex; flex: 1; height: calc(100vh - 85px); }

        /* --- THE LEFT SIDEBAR --- */
        .sidebar {
            width: 25%;
            background: #050505;
            border-right: 3px solid #500;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
            box-shadow: 10px 0 30px rgba(255, 0, 0, 0.15);
            z-index: 10;
        }

        .broadcast-img {
            width: 85%;
            max-width: 400px;
            display: none;
            filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.5));
            animation: flicker 4s infinite alternate;
            margin-bottom: 40px;
        }
        .broadcast-img.active { display: block; }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
            width: 80%;
        }

        button { 
            padding: 15px 10px; background: #900; color: #000; 
            border: 2px solid #f00; font-size: 16px; font-weight: bold; 
            cursor: pointer; text-transform: uppercase; transition: 0.2s; 
        }
        button:hover { background: #f00; color: #fff; box-shadow: 0 0 15px #f00; }
        
        #status { margin-top: 20px; font-size: 16px; font-weight: bold; text-align: center; height: 20px; color: #0f0; }

        .status-text {
            position: absolute; bottom: 30px; color: #f00;
            font-size: 1.2vw; letter-spacing: 4px; animation: blink 1.5s infinite;
        }

        /* --- THE RIGHT CAMERA FEED --- */
        .main-feed { width: 75%; display: flex; justify-content: center; align-items: center; background: #000; }
        .main-feed img { width: 100%; height: 100%; object-fit: cover; }

        @keyframes flicker {
            0% { opacity: 1; filter: drop-shadow(0 0 20px rgba(255,255,255,0.8)); }
            5% { opacity: 0.8; }
            10% { opacity: 1; }
            50% { opacity: 0.9; filter: drop-shadow(0 0 10px rgba(255,255,255,0.4)); }
            100% { opacity: 1; filter: drop-shadow(0 0 40px rgba(255,255,255,1)); }
        }
        @keyframes blink { 50% { opacity: 0; } }
    </style>
</head>
<body>

    <div class="header">
        <h1>SHINIGAMI SURVEILLANCE</h1>
        <div class="subtitle">LIVE SOUL TRACKING ACTIVE // MEMORY BANK ONLINE</div>
    </div>

    <div class="content-wrapper">
        <div class="sidebar">
            <img src="/static/L_logo.png" id="l-logo" class="broadcast-img active" alt="L Broadcast">
            <img src="/static/Kira_logo.png" id="kira-logo" class="broadcast-img" alt="Kira Broadcast">
            
            <div class="controls">
                <button onclick="capturePhoto('clean')">CAPTURE CLEAN</button>
                <button onclick="capturePhoto('shinigami')">CAPTURE SHINIGAMI</button>
            </div>
            <div id="status"></div>

            <div class="status-text">SIGNAL INTERCEPTED</div>
        </div>

        <div class="main-feed">
            <img src="/video_feed">
        </div>
    </div>

    <audio id="shutterSound" src="/static/shutter.mp3" preload="auto"></audio>
    <audio id="laughSound" src="/static/laugh.mp3" preload="auto"></audio>
    <audio id="chimeSound" src="/static/chime.mp3" preload="auto"></audio>

    <script>
        // Secret Switch: Press 'L' or 'K' to swap the sidebar image
        document.addEventListener('keydown', function(event) {
            const key = event.key.toLowerCase();
            if (key === 'l') {
                document.getElementById('l-logo').classList.add('active');
                document.getElementById('kira-logo').classList.remove('active');
            } else if (key === 'k') {
                document.getElementById('kira-logo').classList.add('active');
                document.getElementById('l-logo').classList.remove('active');
            }
        });

        // Photo Capture Logic
        function capturePhoto(type) {
            const status = document.getElementById('status');
            status.innerHTML = "Capturing...";
            status.style.color = "#fff";
            
            const endpoint = (type === 'clean') ? '/capture_clean' : '/capture_shinigami';
            
            fetch(endpoint, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if(data.status === 'success') {
                    document.getElementById('shutterSound').play();
                    status.innerHTML = `SAVED TO FOLDER!`;
                    status.style.color = "#0f0";
                    setTimeout(() => status.innerHTML = "", 3000);
                } else {
                    status.innerHTML = "ERROR CAPTURING.";
                    status.style.color = "#f00";
                }
            });
        }

        // Automated Soundboard Logic (Ryuk laughs & Chimes randomly every 15-20 mins)
        function scheduleRandomSound() {
            const minTime = 15 * 60 * 1000; 
            const maxTime = 20 * 60 * 1000; 
            const delay = Math.random() * (maxTime - minTime) + minTime;

            setTimeout(() => {
                const sounds = ['laughSound', 'chimeSound'];
                const soundId = sounds[Math.floor(Math.random() * sounds.length)];
                
                document.getElementById(soundId).play().catch(e => console.log("Audio play blocked by browser."));
                scheduleRandomSound();
            }, delay);
        }

        scheduleRandomSound();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_DASHBOARD)

@app.route('/video_feed')
def video_feed():
    return Response(generate_shinigami_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_clean', methods=['POST'])
def capture_clean():
    global global_raw_frame
    if global_raw_frame is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("captures", f"clean_{timestamp}.jpg")
        cv2.imwrite(filename, global_raw_frame)
        return jsonify({"status": "success", "file": filename})
    return jsonify({"status": "error"})

@app.route('/capture_shinigami', methods=['POST'])
def capture_shinigami():
    global global_filtered_frame
    if global_filtered_frame is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("captures", f"shinigami_{timestamp}.jpg")
        cv2.imwrite(filename, global_filtered_frame)
        return jsonify({"status": "success", "file": filename})
    return jsonify({"status": "error"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)