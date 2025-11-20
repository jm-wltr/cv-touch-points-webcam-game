"""
Multi-Person Body Touch Game - MediaPipe num_poses Version
Based on jaime.py but with experimental MediaPipe multi-person support

Uses: MediaPipe Pose with num_poses parameter (if available in version 0.10.8+)
Reference: https://gist.github.com/lanzani/f85175d8fbdafcabb7d480dd1bb769d9

Note: This is experimental - num_poses parameter may not be in all MediaPipe versions.
      If it doesn't work, use jaime_hog.py instead.

Requirements:
    pip install opencv-python numpy mediapipe pillow

Author: Based on Jaime's original with multi-person attempt
"""

import cv2
import mediapipe as mp
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize MediaPipe Pose with multi-person support (experimental)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

try:
    # Try to initialize with num_poses parameter
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,  # Lite model for speed
        num_poses=2  # EXPERIMENTAL: Try to detect 2 people
    )
    print("✓ MediaPipe initialized with num_poses=2 (experimental)")
    MULTI_PERSON_MODE = True
except TypeError:
    # Fallback if num_poses not supported
    print("✗ num_poses parameter not supported in this MediaPipe version")
    print("  Falling back to single-person mode")
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    MULTI_PERSON_MODE = False

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50
MAX_PLAYERS = 2

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

# Load fonts
try:
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    FONT_LARGE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    FONT_MEDIUM = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except:
    try:
        FONT_TITLE = ImageFont.truetype("arial.ttf", 60)
        FONT_LARGE = ImageFont.truetype("arial.ttf", 40)
        FONT_MEDIUM = ImageFont.truetype("arial.ttf", 28)
        FONT_SMALL = ImageFont.truetype("arial.ttf", 20)
    except:
        FONT_TITLE = ImageFont.load_default()
        FONT_LARGE = ImageFont.load_default()
        FONT_MEDIUM = ImageFont.load_default()
        FONT_SMALL = ImageFont.load_default()

# ============================================================================
# UI HELPERS (from jaime.py)
# ============================================================================

def draw_text_with_background(draw, text, position, font, text_color, bg_color, padding=10):
    """Draw text with a background box"""
    bbox = draw.textbbox(position, text, font=font)
    bg_box = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
    draw.rectangle(bg_box, fill=bg_color)
    draw.text(position, text, font=font, fill=text_color)

def draw_button_pil(draw, text, position, size, color, hover=False, enabled=True):
    """Draw a beautiful button using PIL"""
    x, y = position
    w, h = size

    if hover and enabled:
        color = tuple(min(c + 30, 255) for c in color)

    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=color)
    border_color = (255, 255, 255) if hover else (200, 200, 200)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, outline=border_color, width=3)

    font = FONT_MEDIUM
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (w - text_w) // 2
    text_y = y + (h - text_h) // 2

    text_color = (255, 255, 255) if enabled else (150, 150, 150)
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    return (x, y, x + w, y + h)

# ============================================================================
# GAME LOGIC
# ============================================================================

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generate_target(frame_width, frame_height):
    margin = 100
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def play_game(cap, frame_width, frame_height):
    """Main game loop with multi-person support"""

    # Initialize targets
    targets = {}  # player_id -> {pos, body_part, touched}
    score = 0
    registration_window = 90
    registered_players = set()
    frame_count = 0

    print("\n" + "="*70)
    if MULTI_PERSON_MODE:
        print("Multi-person mode enabled! Up to 2 players can play.")
    else:
        print("Single-person mode (num_poses not available)")
    print("="*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = pose.process(rgb_frame)

        # Convert to PIL for nice rendering
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        # Extract multiple poses if available
        current_players = []

        if MULTI_PERSON_MODE and hasattr(results, 'poses'):
            # Multi-person: results.poses should be a list
            for idx, pose_landmarks in enumerate(results.poses):
                if idx < MAX_PLAYERS:
                    current_players.append({
                        'id': idx,
                        'landmarks': pose_landmarks
                    })
        elif results.pose_landmarks:
            # Single person fallback
            current_players.append({
                'id': 0,
                'landmarks': results.pose_landmarks
            })

        frame_count += 1

        # Registration phase
        if frame_count <= registration_window:
            for player in current_players:
                registered_players.add(player['id'])

            progress = (frame_count / registration_window) * 100
            draw_text_with_background(draw, f"Registration: {progress:.0f}%", (20, 20),
                                     FONT_LARGE, (0, 255, 255), (0, 0, 0, 180), padding=15)
            draw_text_with_background(draw, f"Players detected: {len(current_players)}", (20, 80),
                                     FONT_MEDIUM, (255, 255, 255), (0, 0, 0, 180), padding=15)
        else:
            # Initialize targets after registration
            if frame_count == registration_window + 1:
                for player_id in registered_players:
                    pos, body_part = generate_target(frame_width, frame_height)
                    targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            # Add/remove targets as needed
            for player in current_players:
                player_id = player['id']
                if player_id not in targets and player_id in registered_players:
                    pos, body_part = generate_target(frame_width, frame_height)
                    targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            # Check collisions
            for player in current_players:
                player_id = player['id']
                if player_id not in targets:
                    continue

                target_info = targets[player_id]
                color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]

                # Get body part position
                landmark = LANDMARK_MAP[target_info['body_part']]
                body_part_landmark = player['landmarks'].landmark[landmark]

                body_part_x = int(body_part_landmark.x * frame_width)
                body_part_y = int(body_part_landmark.y * frame_height)

                # Draw body part tracker
                draw.ellipse([body_part_x - 15, body_part_y - 15, body_part_x + 15, body_part_y + 15],
                           fill=color)

                # Check distance
                distance = calculate_distance((body_part_x, body_part_y), target_info['pos'])

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

                # Draw label
                label = f"P{player_id}: {target_info['body_part']}"
                draw.text((pos[0] - 50, pos[1] - 50), label, font=FONT_SMALL, fill=color)

            # Check if all touched
            if targets and all(t['touched'] for t in targets.values()):
                score += 1
                # Regenerate targets
                for player_id in registered_players:
                    pos, body_part = generate_target(frame_width, frame_height)
                    targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

            # Draw HUD
            draw_text_with_background(draw, f"Team Score: {score}", (20, 20),
                                     FONT_LARGE, (0, 255, 0), (0, 0, 0, 180), padding=15)

            if targets:
                touched = sum(1 for t in targets.values() if t['touched'])
                draw_text_with_background(draw, f"Progress: {touched}/{len(targets)}", (20, 80),
                                         FONT_MEDIUM, (255, 255, 255), (0, 0, 0, 180), padding=15)

        # Draw skeletons
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        for player in current_players:
            color = PLAYER_COLORS[player['id'] % len(PLAYER_COLORS)]
            mp_drawing.draw_landmarks(
                frame_bgr,
                player['landmarks'],
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )

        # Show
        cv2.imshow('Multi-Person Body Touch Game', frame_bgr)

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
    print("Multi-Person Body Touch Game - MediaPipe num_poses Version")
    print("="*70)
    print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {frame_width}×{frame_height}")
    print(f"Multi-person: {MULTI_PERSON_MODE}")
    print()

    result = play_game(cap, frame_width, frame_height)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\nThanks for playing!")

if __name__ == "__main__":
    main()
