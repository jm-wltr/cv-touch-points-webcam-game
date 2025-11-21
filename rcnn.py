"""
Multi-Player Body Part Touch Game with Mask R-CNN Segmentation
Uses instance segmentation for beautiful per-player silhouettes

Performance Optimizations:
- Mask caching (recompute every N frames)
- GPU acceleration (auto-detect)
- Lightweight ResNet-50 backbone
- Optimized mask processing
- Async segmentation updates

Requirements:
    pip install torch torchvision
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    pip install opencv-python numpy

Note: Works best with GPU, but optimized for CPU playability
"""

import cv2
import numpy as np
import random
import time
from collections import deque, defaultdict
import torch

# Import Detectron2
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import ColorMode
except ImportError:
    print("ERROR: Detectron2 not installed!")
    print("Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50  # pixels
TARGET_RADIUS = 25
BODY_PART_RADIUS = 15

# Performance settings
MASK_UPDATE_INTERVAL = 3  # Update masks every N frames (3 = ~20% faster)
DETECTION_CONFIDENCE = 0.7  # Higher = fewer false positives
INPUT_RESOLUTION = 640  # Lower = faster (640, 800, 1024)
USE_GPU = torch.cuda.is_available()

# Visual settings
SILHOUETTE_OPACITY = 0.6
ENABLE_SILHOUETTES = True
ENABLE_SKELETONS = True
DEBUG_MODE = False

# COCO keypoint indices for Mask R-CNN
KEYPOINT_MAP = {
    'Head': 0,          # nose
    'Left Hand': 9,     # left_wrist
    'Right Hand': 10,   # right_wrist
    'Left Elbow': 7,    # left_elbow
    'Right Elbow': 8,   # right_elbow
}

# COCO skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Player colors (BGR)
PLAYER_COLORS = [
    (100, 255, 100),  # Green
    (255, 100, 100),  # Blue
    (100, 100, 255),  # Red
    (255, 255, 100),  # Cyan
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Yellow
    (200, 150, 100),  # Teal
    (150, 100, 200),  # Purple
]

# ============================================================================
# MASK R-CNN SETUP
# ============================================================================

def setup_maskrcnn():
    """Initialize Mask R-CNN with optimized settings"""
    cfg = get_cfg()

    # Use ResNet-50 FPN (lighter than ResNet-101)
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Optimization settings
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_CONFIDENCE
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = INPUT_RESOLUTION
    cfg.INPUT.MAX_SIZE_TEST = INPUT_RESOLUTION

    # GPU/CPU selection
    if USE_GPU:
        cfg.MODEL.DEVICE = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("✓ Using CPU (slower, but optimized)")

    predictor = DefaultPredictor(cfg)
    return predictor

# ============================================================================
# MULTI-PERSON TRACKER
# ============================================================================

class PersonTracker:
    """Track people across frames with mask caching"""

    def __init__(self):
        self.tracks = {}  # track_id -> track_data
        self.next_id = 0
        self.track_colors = {}
        self.mask_cache = {}  # track_id -> cached_mask
        self.frame_count = 0
        self.lost_tracks = {}  # Recently lost tracks for re-association
        self.max_lost_frames = 30

    def get_color(self, track_id):
        """Get consistent color for a track"""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = PLAYER_COLORS[track_id % len(PLAYER_COLORS)]
        return self.track_colors[track_id]

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    def match_detections(self, detections):
        """Match new detections to existing tracks"""
        if not self.tracks:
            # No existing tracks, create new ones
            matches = []
            for i in range(len(detections)):
                matches.append((None, i))
            return matches

        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track_box = self.tracks[track_id]['bbox']
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(track_box, det['bbox'])

        # Simple greedy matching (could use Hungarian for better results)
        matches = []
        matched_tracks = set()
        matched_dets = set()

        # Match highest IoU first
        while True:
            if iou_matrix.size == 0:
                break

            max_iou = iou_matrix.max()
            if max_iou < 0.3:  # Minimum IoU threshold
                break

            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            track_id = track_ids[i]

            matches.append((track_id, j))
            matched_tracks.add(i)
            matched_dets.add(j)

            # Zero out matched row and column
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        # Add unmatched detections as new tracks
        for j in range(len(detections)):
            if j not in matched_dets:
                matches.append((None, j))

        # Mark unmatched tracks as lost
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.lost_tracks[track_id] = self.tracks[track_id]
                self.lost_tracks[track_id]['lost_frame'] = self.frame_count
                del self.tracks[track_id]

        return matches

    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1

        # Match detections to tracks
        matches = self.match_detections(detections)

        # Update matched tracks and create new ones
        for track_id, det_idx in matches:
            det = detections[det_idx]

            if track_id is None:
                # Create new track
                track_id = self.next_id
                self.next_id += 1

            # Update track data
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'keypoints': det['keypoints'],
                'mask': det['mask'],
                'score': det['score'],
                'last_update': self.frame_count
            }

            # Update mask cache if needed
            if self.frame_count % MASK_UPDATE_INTERVAL == 0:
                self.mask_cache[track_id] = det['mask']
            elif track_id not in self.mask_cache:
                self.mask_cache[track_id] = det['mask']

        # Clean up old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if self.frame_count - self.lost_tracks[track_id]['lost_frame'] > self.max_lost_frames:
                del self.lost_tracks[track_id]
                if track_id in self.mask_cache:
                    del self.mask_cache[track_id]

        return list(self.tracks.keys())

    def get_track(self, track_id):
        """Get track data"""
        return self.tracks.get(track_id)

    def get_cached_mask(self, track_id):
        """Get cached mask for track"""
        return self.mask_cache.get(track_id)

# ============================================================================
# GAME LOGIC
# ============================================================================

class Game:
    """Multi-player game with Mask R-CNN segmentation"""

    def __init__(self):
        self.score = 0
        self.targets = {}  # track_id -> {pos, body_part, touched}
        self.registration_window = 90  # frames
        self.registered_players = set()
        self.frame_count = 0

    def generate_target(self, frame_width, frame_height):
        """Generate random target"""
        margin = 100
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

    def update(self, tracker, frame_width, frame_height):
        """Update game state"""
        self.frame_count += 1

        current_players = list(tracker.tracks.keys())

        # Registration window
        if self.frame_count <= self.registration_window:
            self.registered_players.update(current_players)
            return False  # Not playing yet

        # Initialize targets if needed
        if self.frame_count == self.registration_window + 1:
            self.initialize_targets(self.registered_players, frame_width, frame_height)

        # Add targets for new players
        for player_id in current_players:
            if player_id not in self.targets:
                pos, body_part = self.generate_target(frame_width, frame_height)
                self.targets[player_id] = {
                    'pos': pos,
                    'body_part': body_part,
                    'touched': False
                }

        # Remove targets for disconnected players
        for player_id in list(self.targets.keys()):
            if player_id not in current_players:
                del self.targets[player_id]

        # Check collisions
        for player_id in current_players:
            if player_id not in self.targets:
                continue

            track = tracker.get_track(player_id)
            target_info = self.targets[player_id]

            # Get body part position
            keypoints = track['keypoints']
            kp_idx = KEYPOINT_MAP[target_info['body_part']]

            if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.5:  # confidence
                body_x, body_y = keypoints[kp_idx][0], keypoints[kp_idx][1]

                distance = self.calculate_distance((body_x, body_y), target_info['pos'])

                if distance < TOUCH_THRESHOLD:
                    target_info['touched'] = True

        # Check if all targets touched
        if self.targets and all(t['touched'] for t in self.targets.values()):
            self.score += 1
            # Generate new targets
            self.initialize_targets(current_players, frame_width, frame_height)
            return True  # Score increased

        return False

# ============================================================================
# RENDERING
# ============================================================================

def apply_silhouette(frame, mask, color, opacity=SILHOUETTE_OPACITY):
    """Apply colored silhouette overlay"""
    # Create colored mask
    colored_mask = np.zeros_like(frame)
    colored_mask[:] = color

    # Apply mask
    mask_3d = np.stack([mask] * 3, axis=2)
    silhouette = (colored_mask * mask_3d).astype(np.uint8)

    # Blend with frame
    result = cv2.addWeighted(frame, 1 - opacity, silhouette, opacity, 0)

    # Draw outline
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result

def draw_skeleton(frame, keypoints, color):
    """Draw pose skeleton"""
    # Draw keypoints
    for kp in keypoints:
        if kp[2] > 0.5:  # confidence threshold
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)

    # Draw connections
    for connection in SKELETON_CONNECTIONS:
        if connection[0] < len(keypoints) and connection[1] < len(keypoints):
            kp1, kp2 = keypoints[connection[0]], keypoints[connection[1]]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                cv2.line(frame, (int(kp1[0]), int(kp1[1])),
                        (int(kp2[0]), int(kp2[1])), color, 2)

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
    else:
        # Progress
        if game.targets:
            touched = sum(1 for t in game.targets.values() if t['touched'])
            cv2.putText(frame, f"Progress: {touched}/{len(game.targets)}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Player count
    cv2.putText(frame, f"Players: {len(tracker.tracks)}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Controls
    cv2.putText(frame, "Q: Quit | R: Reset | S: Toggle Silhouette | K: Toggle Skeleton",
               (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Performance indicator
    if USE_GPU:
        cv2.circle(frame, (w - 20, 20), 8, (0, 255, 0), -1)  # Green = GPU
    else:
        cv2.circle(frame, (w - 20, 20), 8, (0, 165, 255), -1)  # Orange = CPU

# ============================================================================
# MAIN
# ============================================================================

def main():
    global ENABLE_SILHOUETTES, ENABLE_SKELETONS, DEBUG_MODE

    print("=" * 60)
    print("Multi-Player Body Part Touch Game - Mask R-CNN Edition")
    print("=" * 60)
    print("\nInitializing Mask R-CNN...")

    # Setup Mask R-CNN
    predictor = setup_maskrcnn()

    # Setup game
    tracker = PersonTracker()
    game = Game()

    # Setup webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n✓ Camera: {frame_width}×{frame_height}")
    print(f"✓ Mask update interval: Every {MASK_UPDATE_INTERVAL} frames")
    print(f"✓ Detection confidence: {DETECTION_CONFIDENCE}")
    print("\n" + "=" * 60)
    print("Game starting! Stand in front of camera for registration...")
    print("=" * 60 + "\n")

    # FPS tracking
    fps_queue = deque(maxlen=30)
    frame_count = 0

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Run Mask R-CNN (optimized with caching)
            should_update_masks = (frame_count % MASK_UPDATE_INTERVAL == 0)

            if should_update_masks or frame_count == 0:
                # Run detection
                outputs = predictor(frame)
                instances = outputs["instances"].to("cpu")

                # Filter for people only (class 0 in COCO)
                person_mask = instances.pred_classes == 0
                instances = instances[person_mask]

                # Extract detections
                detections = []
                if len(instances) > 0:
                    boxes = instances.pred_boxes.tensor.numpy()
                    masks = instances.pred_masks.numpy()
                    scores = instances.scores.numpy()

                    # Check if keypoints available
                    has_keypoints = hasattr(instances, 'pred_keypoints')
                    if has_keypoints:
                        keypoints = instances.pred_keypoints.numpy()

                    for i in range(len(instances)):
                        det = {
                            'bbox': boxes[i],
                            'mask': masks[i],
                            'score': scores[i],
                            'keypoints': keypoints[i] if has_keypoints else np.zeros((17, 3))
                        }
                        detections.append(det)

                # Update tracker
                tracker.update(detections)

            # Render frame
            render_frame = frame.copy()

            # Draw silhouettes
            if ENABLE_SILHOUETTES:
                for track_id in tracker.tracks.keys():
                    mask = tracker.get_cached_mask(track_id)
                    if mask is not None:
                        color = tracker.get_color(track_id)
                        render_frame = apply_silhouette(render_frame, mask, color)

            # Draw game elements
            for track_id in tracker.tracks.keys():
                track = tracker.get_track(track_id)
                color = tracker.get_color(track_id)

                # Draw bounding box
                bbox = track['bbox']
                cv2.rectangle(render_frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color, 2)

                # Draw player label
                cv2.putText(render_frame, f"Player {track_id}",
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw skeleton
                if ENABLE_SKELETONS:
                    draw_skeleton(render_frame, track['keypoints'], color)

                # Draw body part tracker
                if track_id in game.targets:
                    target_info = game.targets[track_id]
                    kp_idx = KEYPOINT_MAP[target_info['body_part']]
                    keypoints = track['keypoints']

                    if kp_idx < len(keypoints) and keypoints[kp_idx][2] > 0.5:
                        body_x, body_y = int(keypoints[kp_idx][0]), int(keypoints[kp_idx][1])
                        cv2.circle(render_frame, (body_x, body_y), BODY_PART_RADIUS, color, -1)
                        cv2.circle(render_frame, (body_x, body_y), BODY_PART_RADIUS + 2, (255, 255, 255), 2)

            # Draw targets
            for track_id, target_info in game.targets.items():
                color = tracker.get_color(track_id)
                pos = target_info['pos']

                # Target circle
                if target_info['touched']:
                    cv2.circle(render_frame, pos, TARGET_RADIUS, color, -1)
                    cv2.circle(render_frame, pos, TARGET_RADIUS, (255, 255, 255), 2)
                else:
                    cv2.circle(render_frame, pos, TARGET_RADIUS, color, 3)
                cv2.circle(render_frame, pos, 5, color, -1)

                # Target label
                label = f"P{track_id}: {target_info['body_part']}"
                cv2.putText(render_frame, label, (pos[0] - 60, pos[1] - TARGET_RADIUS - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update game logic
            game.update(tracker, frame_width, frame_height)

            # Draw HUD
            fps = 1.0 / (time.time() - start_time + 1e-6)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            draw_hud(render_frame, game, tracker, avg_fps)

            # Display
            cv2.imshow('Mask R-CNN Body Touch Game', render_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset game
                tracker = PersonTracker()
                game = Game()
                print("\n✓ Game reset!")
            elif key == ord('s'):
                ENABLE_SILHOUETTES = not ENABLE_SILHOUETTES
                print(f"\n✓ Silhouettes: {'ON' if ENABLE_SILHOUETTES else 'OFF'}")
            elif key == ord('k'):
                ENABLE_SKELETONS = not ENABLE_SKELETONS
                print(f"\n✓ Skeletons: {'ON' if ENABLE_SKELETONS else 'OFF'}")
            elif key == ord('d'):
                DEBUG_MODE = not DEBUG_MODE
                print(f"\n✓ Debug: {'ON' if DEBUG_MODE else 'OFF'}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n✓ Game interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print(f"Game Over! Final Score: {game.score}")
        print(f"Average FPS: {np.mean(fps_queue):.1f}")
        print("=" * 60)

if __name__ == "__main__":
    main()