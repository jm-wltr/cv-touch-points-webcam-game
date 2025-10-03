from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import random
import numpy as np
import base64
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Enable CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize MediaPipe with lite models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50

LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

# Global game state
game_state = {
    'score': 0,
    'target_pos': [320, 240],
    'target_body_part': 'Right Hand',
    'debug_mode': False,
    'frame_width': 640,
    'frame_height': 480,
    'active': False
}

cap = None
streaming = False
stream_thread = None

def generate_target(frame_width, frame_height):
    """Generate a random target position and body part"""
    margin = 100
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return [x, y], body_part

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_fist_closed(hand_landmarks):
    """Detect if hand is making a fist"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    fingertips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    closed_count = 0
    for tip, pip in zip(fingertips, finger_pips):
        tip_point = hand_landmarks.landmark[tip]
        pip_point = hand_landmarks.landmark[pip]
        
        tip_dist = ((tip_point.x - wrist.x)**2 + (tip_point.y - wrist.y)**2)**0.5
        pip_dist = ((pip_point.x - wrist.x)**2 + (pip_point.y - wrist.y)**2)**0.5
        
        if tip_dist < pip_dist * 1.1:
            closed_count += 1
    
    return closed_count >= 3

def video_stream():
    """Stream video frames via WebSocket"""
    global cap, streaming
    
    frame_count = 0
    
    while streaming:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every frame for smooth video
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process based on game state
        if game_state['active']:
            # Game mode - process pose
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                if game_state['debug_mode']:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                
                landmark = LANDMARK_MAP[game_state['target_body_part']]
                body_part_landmark = results.pose_landmarks.landmark[landmark]
                
                body_part_x = int(body_part_landmark.x * frame.shape[1])
                body_part_y = int(body_part_landmark.y * frame.shape[0])
                
                # Draw tracked body part
                cv2.circle(frame, (body_part_x, body_part_y), 15, (255, 0, 255), -1)
                
                distance = calculate_distance((body_part_x, body_part_y), game_state['target_pos'])
                
                if distance < TOUCH_THRESHOLD:
                    game_state['score'] += 1
                    game_state['target_pos'], game_state['target_body_part'] = generate_target(
                        game_state['frame_width'], game_state['frame_height']
                    )
                    # Emit score update
                    socketio.emit('game_state', game_state)
            
            # Draw target
            cv2.circle(frame, tuple(game_state['target_pos']), 30, (0, 0, 255), 3)
            cv2.circle(frame, tuple(game_state['target_pos']), 5, (0, 0, 255), -1)
            
        else:
            # Menu mode - process hands
            hand_results = hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_x = int(wrist.x * frame.shape[1])
                    hand_y = int(wrist.y * frame.shape[0])
                    fist_closed = is_fist_closed(hand_landmarks)
                    
                    # Emit hand data
                    socketio.emit('hand_data', {
                        'x': hand_x,
                        'y': hand_y,
                        'fist_closed': fist_closed
                    })
                    break
            
            # Draw skeleton if debug mode
            if game_state['debug_mode']:
                pose_results = pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit frame via WebSocket
        socketio.emit('video_frame', {'frame': frame_base64})
        
        # Control frame rate (30 FPS)
        time.sleep(0.033)

@app.route('/')
def index():
    return jsonify({'status': 'Server is running', 'message': 'Use WebSocket for video streaming'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('init_camera')
def handle_init_camera():
    """Initialize camera"""
    global cap, streaming, stream_thread
    
    try:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                game_state['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                game_state['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera initialized: {game_state['frame_width']}x{game_state['frame_height']}")
                
                # Start video streaming thread
                if not streaming:
                    streaming = True
                    stream_thread = threading.Thread(target=video_stream)
                    stream_thread.daemon = True
                    stream_thread.start()
                
                emit('camera_ready', {
                    'success': True, 
                    'width': game_state['frame_width'], 
                    'height': game_state['frame_height']
                })
            else:
                emit('camera_ready', {'success': False, 'error': 'Cannot open camera'})
        else:
            emit('camera_ready', {
                'success': True, 
                'width': game_state['frame_width'], 
                'height': game_state['frame_height']
            })
    except Exception as e:
        print(f"Error initializing camera: {e}")
        emit('camera_ready', {'success': False, 'error': str(e)})

@socketio.on('start_game')
def handle_start_game():
    """Start a new game"""
    game_state['score'] = 0
    game_state['active'] = True
    game_state['target_pos'], game_state['target_body_part'] = generate_target(
        game_state['frame_width'], game_state['frame_height']
    )
    print(f"Game started. Target: {game_state['target_body_part']} at {game_state['target_pos']}")
    emit('game_started', game_state)

@socketio.on('stop_game')
def handle_stop_game():
    """Stop the game"""
    game_state['active'] = False
    print(f"Game stopped. Final score: {game_state['score']}")
    emit('game_stopped', {'final_score': game_state['score']})

@socketio.on('toggle_debug')
def handle_toggle_debug():
    """Toggle debug mode"""
    game_state['debug_mode'] = not game_state['debug_mode']
    print(f"Debug mode: {game_state['debug_mode']}")
    emit('debug_toggled', {'debug_mode': game_state['debug_mode']})

@socketio.on('get_game_state')
def handle_get_game_state():
    """Get current game state"""
    emit('game_state', game_state)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Flask-SocketIO server on http://localhost:5001")
    print("Make sure React frontend is running on http://localhost:3000")
    print("="*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)