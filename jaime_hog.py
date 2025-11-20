"""
Multi-Person Body Touch Game - HOG Detector Version
Based on jaime.py with HOG person detection + sequential MediaPipe

Architecture:
  1. OpenCV HOG person detector finds people (no YOLO needed!)
  2. Crop each person from frame
  3. Run MediaPipe Pose on each crop
  4. Transform coordinates back
  5. Render with Jaime's beautiful PIL UI

Performance:
  - No YOLO dependency (lighter)
  - ~15-18 FPS with 2 players (CPU)
  - HOG is less accurate than YOLO but faster to set up

Requirements:
    pip install opencv-python numpy mediapipe pillow

Author: Based on Jaime's original with HOG multi-person detection
"""

import cv2
import mediapipe as mp
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Lite for speed
)

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50
MAX_PLAYERS = 2
HOG_SCALE = 1.05  # Increase for speed, decrease for accuracy
HOG_PADDING = (16, 16)  # Padding for HOG windows

LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

# Player colors (BGR)
PLAYER_COLORS = [
    (100, 255, 100),  # Green
    (255, 100, 100),  # Blue
    (100, 100, 255),  # Red
]

# Load fonts (Jaime's style)
try:
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    FONT_LARGE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    FONT_MEDIUM = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    print("✓ Loaded system fonts")
except:
    try:
        FONT_TITLE = ImageFont.truetype("arial.ttf", 60)
        FONT_LARGE = ImageFont.truetype("arial.ttf", 40)
        FONT_MEDIUM = ImageFont.truetype("arial.ttf", 28)
        FONT_SMALL = ImageFont.truetype("arial.ttf", 20)
        print("✓ Loaded Arial fonts")
    except:
        FONT_TITLE = ImageFont.load_default()
        FONT_LARGE = ImageFont.load_default()
        FONT_MEDIUM = ImageFont.load_default()
        FONT_SMALL = ImageFont.load_default()
        print("⚠ Using default fonts")

# ============================================================================
# UI HELPERS (Jaime's beautiful rendering)
# ============================================================================

def draw_text_with_background(draw, text, position, font, text_color, bg_color, padding=10):
    """Draw text with a background box"""
    bbox = draw.textbbox(position, text, font=font)
    bg_box = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
    draw.rectangle(bg_box, fill=bg_color)
    draw.text(position, text, font=font, fill=text_color)

# ============================================================================
# PERSON DETECTION & TRACKING
# ============================================================================

def detect_people(frame):
    """Detect people using HOG descriptor"""
    # Resize for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect
    boxes, weights = hog.detectMultiScale(
        small_frame,
        winStride=(8, 8),
        padding=HOG_PADDING,
        scale=HOG_SCALE
    )

    # Scale back to original frame
    detections = []
    for (x, y, w, h), weight in zip(boxes, weights):
        # Scale coordinates back
        x, y, w, h = x*2, y*2, w*2, h*2

        # Filter by confidence
        if weight > 0.5:  # Confidence threshold
            detections.append({
                'bbox': [x, y, x+w, y+h],
                'confidence': weight
            })

    # Sort by confidence and limit
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:MAX_PLAYERS]

    return detections

def extract_padded_crop(frame, bbox, padding=0.2):
    """Extract person crop with padding"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)

    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(w, int(x2 + pad_w))
    crop_y2 = min(h, int(y2 + pad_h))

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_info = {
        'x1': crop_x1,
        'y1': crop_y1,
        'x2': crop_x2,
        'y2': crop_y2,
        'width': crop_x2 - crop_x1,
        'height': crop_y2 - crop_y1
    }

    return crop, crop_info

def process_person_crop(crop, crop_info):
    """Process crop with MediaPipe"""
    if crop.shape[0] < 50 or crop.shape[1] < 50:
        return None

    # MediaPipe processing
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_crop)

    if not results.pose_landmarks:
        return None

    # Transform landmarks to full frame coordinates
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        crop_x = landmark.x * crop_info['width']
        crop_y = landmark.y * crop_info['height']

        full_x = crop_info['x1'] + crop_x
        full_y = crop_info['y1'] + crop_y

        landmarks.append({
            'x': full_x,
            'y': full_y,
            'visibility': landmark.visibility
        })

    return {
        'landmarks': landmarks,
        'crop_info': crop_info
    }

# ============================================================================
# TRACKER
# ============================================================================

class SimpleTracker:
    """Simple IoU-based tracker"""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def compute_iou(self, box1, box2):
        """Compute IoU between boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    def update(self, detections):
        """Update tracks"""
        matched = {}
        unmatched_dets = list(range(len(detections)))

        # Match to existing tracks
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det = None

            for det_idx in unmatched_dets:
                iou = self.compute_iou(track['bbox'], detections[det_idx]['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_det = det_idx

            if best_det is not None:
                matched[track_id] = best_det
                unmatched_dets.remove(best_det)

        # Update matched tracks
        new_tracks = {}
        for track_id, det_idx in matched.items():
            new_tracks[track_id] = detections[det_idx]

        # Create new tracks
        for det_idx in unmatched_dets:
            if len(new_tracks) < MAX_PLAYERS:
                new_tracks[self.next_id] = detections[det_idx]
                self.next_id += 1

        self.tracks = new_tracks
        return list(self.tracks.keys())

    def get_track(self, track_id):
        return self.tracks.get(track_id)

# ============================================================================
# GAME LOGIC
# ============================================================================

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generate_target(frame_width, frame_height):
    margin = 80
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def play_game(cap, frame_width, frame_height):
    """Main game loop"""
    tracker = SimpleTracker()
    targets = {}
    score = 0
    registration_window = 90
    registered_players = set()
    frame_count = 0

    print("\n" + "="*70)
    print("Game starting! HOG detector will find up to 2 players...")
    print("="*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Detect people with HOG
        detections = detect_people(frame)

        # Update tracker
        player_ids = tracker.update(detections)

        # Process each player with MediaPipe
        person_results = {}
        for player_id in player_ids:
            track = tracker.get_track(player_id)
            crop, crop_info = extract_padded_crop(frame, track['bbox'])

            result = process_person_crop(crop, crop_info)
            if result:
                person_results[player_id] = result

        # Convert to PIL for beautiful rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        # Registration phase
        if frame_count <= registration_window:
            registered_players.update(player_ids)

            progress = (frame_count / registration_window) * 100
            draw_text_with_background(draw, f"Registration: {progress:.0f}%", (20, 20),
                                     FONT_LARGE, (0, 255, 255), (0, 0, 0, 180), padding=15)
            draw_text_with_background(draw, f"Players: {len(player_ids)}/{MAX_PLAYERS}", (20, 80),
                                     FONT_MEDIUM, (255, 255, 255), (0, 0, 0, 180), padding=15)
        else:
            # Initialize targets
            if frame_count == registration_window + 1:
                for player_id in registered_players:
                    pos, body_part = generate_target(frame_width, frame_height)
                    targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            # Add/remove targets
            for player_id in player_ids:
                if player_id not in targets and player_id in registered_players:
                    pos, body_part = generate_target(frame_width, frame_height)
                    targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            for player_id in list(targets.keys()):
                if player_id not in player_ids:
                    del targets[player_id]

            # Game logic
            for player_id, result in person_results.items():
                if player_id not in targets:
                    continue

                target_info = targets[player_id]
                color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                landmarks = result['landmarks']

                # Get body part landmark
                landmark_idx = list(LANDMARK_MAP.values()).index(LANDMARK_MAP[target_info['body_part']])

                if landmark_idx < len(landmarks):
                    landmark = landmarks[landmark_idx]
                    if landmark['visibility'] > 0.5:
                        body_x, body_y = int(landmark['x']), int(landmark['y'])

                        # Draw body part tracker
                        draw.ellipse([body_x - 15, body_y - 15, body_x + 15, body_y + 15],
                                   fill=color)

                        # Check collision
                        distance = calculate_distance((body_x, body_y), target_info['pos'])
                        if distance < TOUCH_THRESHOLD:
                            target_info['touched'] = True

                # Draw target
                pos = target_info['pos']
                if target_info['touched']:
                    draw.ellipse([pos[0] - 30, pos[1] - 30, pos[0] + 30, pos[1] + 30],
                               fill=color, outline=(255, 255, 255), width=3)
                else:
                    draw.ellipse([pos[0] - 30, pos[1] - 30, pos[0] + 30, pos[1] + 30],
                               outline=color, width=3)
                draw.ellipse([pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5], fill=color)

                # Label
                label = f"P{player_id}: {target_info['body_part']}"
                draw.text((pos[0] - 50, pos[1] - 50), label, font=FONT_SMALL, fill=color)

            # Check if all touched
            if targets and all(t['touched'] for t in targets.values()):
                score += 1
                for player_id in registered_players:
                    if player_id in targets:
                        pos, body_part = generate_target(frame_width, frame_height)
                        targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            # Draw HUD
            draw_text_with_background(draw, f"Team Score: {score}", (20, 20),
                                     FONT_LARGE, (0, 255, 0), (0, 0, 0, 180), padding=15)

            if targets:
                touched = sum(1 for t in targets.values() if t['touched'])
                draw_text_with_background(draw, f"Progress: {touched}/{len(targets)}", (20, 80),
                                         FONT_MEDIUM, (255, 255, 255), (0, 0, 0, 180), padding=15)

        # Convert back to BGR and draw skeletons
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        for player_id, result in person_results.items():
            color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]

            # Draw landmarks
            for landmark in result['landmarks']:
                if landmark['visibility'] > 0.5:
                    cv2.circle(frame_bgr, (int(landmark['x']), int(landmark['y'])),
                             4, color, -1)

            # Draw bounding box
            track = tracker.get_track(player_id)
            if track:
                bbox = track['bbox']
                cv2.rectangle(frame_bgr,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color, 2)
                cv2.putText(frame_bgr, f"Player {player_id}",
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show
        cv2.imshow('HOG Multi-Person Game', frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == 27:
            return 'menu'

    return 'quit'

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Multi-Person Body Touch Game - HOG Detector Version")
    print("="*70)
    print()
    print("Using: OpenCV HOG person detector + MediaPipe Pose")
    print("No YOLO needed!")
    print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {frame_width}×{frame_height}")
    print(f"Max players: {MAX_PLAYERS}")
    print(f"Expected FPS: ~15-18 FPS")
    print()

    result = play_game(cap, frame_width, frame_height)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\n" + "="*70)
    print("Thanks for playing!")
    print("="*70)

if __name__ == "__main__":
    main()
