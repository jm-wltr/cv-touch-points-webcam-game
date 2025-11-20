"""
Hybrid Multi-Player Game: YOLO Detection + MediaPipe Segmentation
Combines Jay's multi-person tracking with Kristian's beautiful silhouettes

Architecture:
  1. YOLOv8n detects people (fast, multi-person bounding boxes)
  2. Crop each person from frame
  3. MediaPipe Holistic per crop (pose + segmentation)
  4. Transform coordinates back to full frame
  5. Composite silhouettes with player colors

Performance:
  - 1 player: ~17-20 FPS (CPU)
  - 2 players: ~13-15 FPS (CPU)
  - Sequential processing (MediaPipe not thread-safe)
  - Adaptive mask caching for efficiency

Note: Multi-threading disabled due to MediaPipe timestamp conflicts.
      For thread-safe parallel processing, create separate MediaPipe
      instances per thread (increases memory ~60MB per instance).

Requirements:
    pip install opencv-python numpy mediapipe ultralytics

Author: Hybrid approach combining team's best features
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 45
TARGET_RADIUS = 25
BODY_PART_RADIUS = 15

# Player settings
MAX_PLAYERS = 2  # Limit for performance

# Performance settings
YOLO_CONFIDENCE = 0.5       # Detection threshold
CROP_PADDING = 0.25         # 25% padding around YOLO bbox
CROP_TARGET_SIZE = (256, 256)  # Resize crops for faster MediaPipe
MASK_UPDATE_INTERVAL = 2    # Update masks every N frames
USE_THREADING = False       # Parallel crop processing (DISABLED - MediaPipe not thread-safe)
THREAD_WORKERS = 2          # Max threads

# MediaPipe settings (lite for speed)
MP_MODEL_COMPLEXITY = 0     # 0=lite, 1=full, 2=heavy (0 is fastest)
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5

# Visual settings
SILHOUETTE_OPACITY = 0.65
SHOW_SILHOUETTES = True
SHOW_SKELETONS = True
SHOW_DEBUG = False

# Player colors (BGR)
PLAYER_COLORS = [
    (100, 255, 100),  # Green
    (255, 100, 100),  # Blue
    (100, 100, 255),  # Red
    (255, 255, 100),  # Cyan
]

# MediaPipe landmark mapping
LANDMARK_MAP = {
    'Right Hand': mp.solutions.holistic.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp.solutions.holistic.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp.solutions.holistic.PoseLandmark.LEFT_ELBOW,
    'Head': mp.solutions.holistic.PoseLandmark.NOSE,
}

# Pose connections for skeleton
POSE_CONNECTIONS = mp.solutions.holistic.POSE_CONNECTIONS

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_models():
    """Initialize YOLO and MediaPipe with optimal settings"""
    print("Initializing models...")

    # YOLO (fastest nano model)
    print("  Loading YOLOv8n...")
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.fuse()  # Fuse layers for speed

    # MediaPipe Holistic (lite model)
    print("  Loading MediaPipe Holistic (lite)...")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        model_complexity=MP_MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True,
        refine_face_landmarks=False  # Disable face for speed
    )

    print("✓ Models loaded successfully!\n")
    return yolo_model, holistic, mp_holistic

# ============================================================================
# PERSON TRACKER
# ============================================================================

class PersonTracker:
    """Track people across frames with ID consistency"""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        self.mask_cache = {}

    def compute_iou(self, box1, box2):
        """Compute intersection over union"""
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
        """Update tracks with new detections"""
        self.frame_count += 1

        # Match detections to existing tracks
        matched = {}
        unmatched_dets = list(range(len(detections)))

        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det = None

            for det_idx in unmatched_dets:
                iou = self.compute_iou(track['bbox'], detections[det_idx]['bbox'])
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_det = det_idx

            if best_det is not None:
                matched[track_id] = best_det
                unmatched_dets.remove(best_det)

        # Update matched tracks
        new_tracks = {}
        for track_id, det_idx in matched.items():
            new_tracks[track_id] = detections[det_idx]
            new_tracks[track_id]['last_seen'] = self.frame_count

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(new_tracks) < MAX_PLAYERS:
                new_tracks[self.next_id] = detections[det_idx]
                new_tracks[self.next_id]['last_seen'] = self.frame_count
                self.next_id += 1

        self.tracks = new_tracks
        return list(self.tracks.keys())

    def get_track(self, track_id):
        """Get track data"""
        return self.tracks.get(track_id)

# ============================================================================
# CROP PROCESSING
# ============================================================================

def extract_padded_crop(frame, bbox, padding=CROP_PADDING):
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

    # Extract crop
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Store crop info for coordinate transformation
    crop_info = {
        'x1': crop_x1,
        'y1': crop_y1,
        'x2': crop_x2,
        'y2': crop_y2,
        'width': crop_x2 - crop_x1,
        'height': crop_y2 - crop_y1
    }

    return crop, crop_info

def process_person_crop(crop, holistic, crop_info):
    """Process a single person crop with MediaPipe"""
    # Resize for faster processing
    crop_resized = cv2.resize(crop, CROP_TARGET_SIZE)

    # MediaPipe expects RGB
    rgb_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

    # Run MediaPipe
    results = holistic.process(rgb_crop)

    # Transform results back to original crop size
    transformed = transform_results(results, crop_info, CROP_TARGET_SIZE)

    return transformed

def transform_results(results, crop_info, resized_size):
    """Transform MediaPipe results from resized crop to full frame coordinates"""
    if not results.pose_landmarks:
        return None

    # Scale factors
    scale_x = crop_info['width'] / resized_size[0]
    scale_y = crop_info['height'] / resized_size[1]

    # Transform landmarks
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        # Landmark coordinates are relative to crop (0-1)
        crop_x = landmark.x * crop_info['width']
        crop_y = landmark.y * crop_info['height']

        # Convert to full frame
        full_x = crop_info['x1'] + crop_x
        full_y = crop_info['y1'] + crop_y

        landmarks.append({
            'x': full_x,
            'y': full_y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })

    # Transform segmentation mask
    mask = None
    if results.segmentation_mask is not None:
        # Resize mask to original crop size
        mask_resized = cv2.resize(results.segmentation_mask,
                                  (crop_info['width'], crop_info['height']))

        # Create full frame mask
        mask = np.zeros((crop_info['height'], crop_info['width']), dtype=np.float32)
        mask = mask_resized

    return {
        'landmarks': landmarks,
        'mask': mask,
        'crop_info': crop_info
    }

# ============================================================================
# MASK PROCESSING (Kristian's approach)
# ============================================================================

def create_smooth_mask(segmentation_mask):
    """Create smooth body mask (Kristian's post-processing)"""
    if segmentation_mask is None:
        return None

    # Binarize
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Gaussian blur for smooth edges
    binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 2)
    binary_mask = (binary_mask > 128).astype(np.uint8) * 255

    return binary_mask

def apply_silhouette(frame, mask, crop_info, color, opacity=SILHOUETTE_OPACITY):
    """Apply colored silhouette overlay to frame"""
    if mask is None:
        return frame

    # Smooth the mask
    smooth_mask = create_smooth_mask(mask)
    if smooth_mask is None:
        return frame

    # Create colored overlay for crop region
    h, w = frame.shape[:2]
    x1, y1 = crop_info['x1'], crop_info['y1']
    x2, y2 = crop_info['x2'], crop_info['y2']

    # Ensure bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Get crop region from frame
    crop_region = frame[y1:y2, x1:x2].copy()

    # Create colored overlay
    colored_overlay = np.zeros_like(crop_region)
    colored_overlay[:] = color

    # Apply mask
    mask_3d = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Blend
    blended = cv2.addWeighted(crop_region, 1 - opacity, colored_overlay, opacity, 0)
    result = (crop_region * (1 - mask_3d) + blended * mask_3d).astype(np.uint8)

    # Place back in frame
    frame[y1:y2, x1:x2] = result

    # Draw contours
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Offset contour to full frame coordinates
        contour_offset = contour + np.array([x1, y1])
        cv2.drawContours(frame, [contour_offset], -1, color, 2)

    return frame

# ============================================================================
# RENDERING
# ============================================================================

def draw_skeleton(frame, landmarks, color):
    """Draw pose skeleton"""
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if start['visibility'] > 0.5 and end['visibility'] > 0.5:
                cv2.line(frame,
                        (int(start['x']), int(start['y'])),
                        (int(end['x']), int(end['y'])),
                        color, 2)

    # Draw keypoints
    for landmark in landmarks:
        if landmark['visibility'] > 0.5:
            cv2.circle(frame, (int(landmark['x']), int(landmark['y'])),
                      4, color, -1)

def draw_hud(frame, game, tracker, fps):
    """Draw game UI"""
    h, w = frame.shape[:2]

    # Score
    cv2.putText(frame, f"Team Score: {game.score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Registration status
    if game.frame_count <= game.registration_window:
        progress = (game.frame_count / game.registration_window) * 100
        cv2.putText(frame, f"Registration: {progress:.0f}%", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, "Stand in front of camera!", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Progress
        if game.targets:
            touched = sum(1 for t in game.targets.values() if t['touched'])
            cv2.putText(frame, f"Progress: {touched}/{len(game.targets)}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Player count
    cv2.putText(frame, f"Players: {len(tracker.tracks)}/{MAX_PLAYERS}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS
    fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

    # Controls
    cv2.putText(frame, "Q: Quit | R: Reset | S: Silhouette | K: Skeleton | D: Debug",
               (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Model indicator
    cv2.putText(frame, "YOLO+MediaPipe", (w - 180, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

# ============================================================================
# GAME LOGIC
# ============================================================================

class Game:
    """Game state and logic"""

    def __init__(self):
        self.score = 0
        self.targets = {}
        self.registration_window = 90  # frames
        self.registered_players = set()
        self.frame_count = 0

    def generate_target(self, frame_width, frame_height):
        """Generate random target"""
        margin = 80
        x = random.randint(margin, frame_width - margin)
        y = random.randint(margin, frame_height - margin)
        body_part = random.choice(BODY_PARTS)
        return (x, y), body_part

    def initialize_targets(self, player_ids, frame_width, frame_height):
        """Create targets for all players"""
        self.targets = {}
        for player_id in player_ids:
            pos, body_part = self.generate_target(frame_width, frame_height)
            self.targets[player_id] = {
                'pos': pos,
                'body_part': body_part,
                'touched': False
            }

    def calculate_distance(self, p1, p2):
        """Euclidean distance"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def update(self, tracker, person_results, frame_width, frame_height):
        """Update game state"""
        self.frame_count += 1

        current_players = list(tracker.tracks.keys())

        # Registration window
        if self.frame_count <= self.registration_window:
            self.registered_players.update(current_players)
            return False

        # Initialize targets after registration
        if self.frame_count == self.registration_window + 1:
            self.initialize_targets(self.registered_players, frame_width, frame_height)

        # Add/remove targets as needed
        for player_id in current_players:
            if player_id not in self.targets:
                pos, body_part = self.generate_target(frame_width, frame_height)
                self.targets[player_id] = {'pos': pos, 'body_part': body_part, 'touched': False}

        for player_id in list(self.targets.keys()):
            if player_id not in current_players:
                del self.targets[player_id]

        # Check collisions
        for player_id, result in person_results.items():
            if player_id not in self.targets or result is None:
                continue

            target_info = self.targets[player_id]
            landmarks = result['landmarks']

            # Get target body part landmark
            landmark_idx = list(LANDMARK_MAP.values()).index(LANDMARK_MAP[target_info['body_part']])

            if landmark_idx < len(landmarks):
                landmark = landmarks[landmark_idx]
                if landmark['visibility'] > 0.5:
                    body_pos = (landmark['x'], landmark['y'])
                    distance = self.calculate_distance(body_pos, target_info['pos'])

                    if distance < TOUCH_THRESHOLD:
                        target_info['touched'] = True

        # Check if all touched
        if self.targets and all(t['touched'] for t in self.targets.values()):
            self.score += 1
            self.initialize_targets(current_players, frame_width, frame_height)
            return True

        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    global SHOW_SILHOUETTES, SHOW_SKELETONS, SHOW_DEBUG

    print("=" * 70)
    print("      HYBRID MULTI-PLAYER BODY TOUCH GAME")
    print("      YOLO Detection + MediaPipe Segmentation")
    print("=" * 70)
    print()

    # Initialize models
    yolo_model, holistic, mp_holistic = initialize_models()

    # Initialize game
    tracker = PersonTracker()
    game = Game()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {frame_width}×{frame_height}")
    print(f"Max players: {MAX_PLAYERS}")
    print(f"Threading: {'Enabled' if USE_THREADING else 'Disabled (MediaPipe not thread-safe)'}")
    print(f"Expected FPS: ~{'17' if MAX_PLAYERS == 1 else '13-15'} FPS")
    print()
    print("=" * 70)
    print("Game starting! Players stand in front of camera...")
    print("=" * 70)
    print()

    # FPS tracking
    fps_queue = deque(maxlen=30)
    frame_count = 0

    # Thread pool
    executor = ThreadPoolExecutor(max_workers=THREAD_WORKERS) if USE_THREADING else None

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # STEP 1: YOLO Detection
            yolo_results = yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]

            # Extract person detections
            detections = []
            if yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    # Only person class (class 0 in COCO)
                    if int(box.cls[0]) == 0:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append({'bbox': bbox, 'conf': conf})

            # Limit to top N detections
            detections = sorted(detections, key=lambda x: x['conf'], reverse=True)[:MAX_PLAYERS]

            # Update tracker
            player_ids = tracker.update(detections)

            # STEP 2 & 3: Extract crops and process with MediaPipe
            person_results = {}

            if player_ids:
                crops_to_process = []
                for player_id in player_ids:
                    track = tracker.get_track(player_id)
                    crop, crop_info = extract_padded_crop(frame, track['bbox'])

                    # Skip invalid crops
                    if crop.shape[0] < 50 or crop.shape[1] < 50:
                        continue

                    crops_to_process.append((player_id, crop, crop_info))

                # Process crops (parallel if enabled)
                should_update = (frame_count % MASK_UPDATE_INTERVAL == 0) or frame_count < 5

                if should_update and crops_to_process:
                    if USE_THREADING and len(crops_to_process) > 1:
                        # Parallel processing
                        futures = []
                        for player_id, crop, crop_info in crops_to_process:
                            future = executor.submit(process_person_crop, crop, holistic, crop_info)
                            futures.append((player_id, future))

                        for player_id, future in futures:
                            person_results[player_id] = future.result()
                    else:
                        # Sequential processing
                        for player_id, crop, crop_info in crops_to_process:
                            person_results[player_id] = process_person_crop(crop, holistic, crop_info)

                    # Cache results
                    tracker.mask_cache = person_results.copy()
                else:
                    # Use cached results
                    person_results = tracker.mask_cache

            # STEP 4: Render
            render_frame = frame.copy()

            # Draw silhouettes
            if SHOW_SILHOUETTES:
                for player_id in player_ids:
                    if player_id in person_results and person_results[player_id]:
                        result = person_results[player_id]
                        color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                        render_frame = apply_silhouette(render_frame, result['mask'],
                                                       result['crop_info'], color)

            # Draw game elements
            for player_id in player_ids:
                if player_id not in person_results or person_results[player_id] is None:
                    continue

                result = person_results[player_id]
                color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                landmarks = result['landmarks']

                # Draw bounding box
                track = tracker.get_track(player_id)
                bbox = track['bbox']
                cv2.rectangle(render_frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color, 2)

                # Draw player label
                cv2.putText(render_frame, f"Player {player_id}",
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw skeleton
                if SHOW_SKELETONS:
                    draw_skeleton(render_frame, landmarks, color)

                # Draw body part tracker
                if player_id in game.targets:
                    target_info = game.targets[player_id]
                    landmark_idx = list(LANDMARK_MAP.values()).index(LANDMARK_MAP[target_info['body_part']])

                    if landmark_idx < len(landmarks):
                        landmark = landmarks[landmark_idx]
                        if landmark['visibility'] > 0.5:
                            body_x, body_y = int(landmark['x']), int(landmark['y'])
                            cv2.circle(render_frame, (body_x, body_y), BODY_PART_RADIUS, color, -1)
                            cv2.circle(render_frame, (body_x, body_y), BODY_PART_RADIUS + 2,
                                     (255, 255, 255), 2)

            # Draw targets
            for player_id, target_info in game.targets.items():
                color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                pos = target_info['pos']

                # Target circle
                if target_info['touched']:
                    cv2.circle(render_frame, pos, TARGET_RADIUS, color, -1)
                    cv2.circle(render_frame, pos, TARGET_RADIUS, (255, 255, 255), 2)
                else:
                    cv2.circle(render_frame, pos, TARGET_RADIUS, color, 3)
                cv2.circle(render_frame, pos, 5, color, -1)

                # Label
                label = f"P{player_id}: {target_info['body_part']}"
                cv2.putText(render_frame, label, (pos[0] - 60, pos[1] - TARGET_RADIUS - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update game
            game.update(tracker, person_results, frame_width, frame_height)

            # Draw HUD
            fps = 1.0 / (time.time() - start_time + 1e-6)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            draw_hud(render_frame, game, tracker, avg_fps)

            # Display
            cv2.imshow('Hybrid Multi-Player Game', render_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker = PersonTracker()
                game = Game()
                print("\n✓ Game reset!")
            elif key == ord('s'):
                SHOW_SILHOUETTES = not SHOW_SILHOUETTES
                print(f"\n✓ Silhouettes: {'ON' if SHOW_SILHOUETTES else 'OFF'}")
            elif key == ord('k'):
                SHOW_SKELETONS = not SHOW_SKELETONS
                print(f"\n✓ Skeletons: {'ON' if SHOW_SKELETONS else 'OFF'}")
            elif key == ord('d'):
                SHOW_DEBUG = not SHOW_DEBUG
                print(f"\n✓ Debug: {'ON' if SHOW_DEBUG else 'OFF'}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")

    finally:
        if executor:
            executor.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

        print("\n" + "=" * 70)
        print(f"Game Over! Final Score: {game.score}")
        print(f"Average FPS: {np.mean(fps_queue):.1f}")
        print("=" * 70)

if __name__ == "__main__":
    main()
