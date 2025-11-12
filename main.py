import cv2
import random
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment
import time
import os

# Initialize YOLO Pose model
model_path = os.path.join(os.path.dirname(__file__), '..', 'multi-pose', 'yolov8n-pose.pt')
if not os.path.exists(model_path):
    model_path = 'yolov8n-pose.pt'  # Fallback to current directory
model = YOLO(model_path)

class InsightFaceRecognizer:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def extract_face_features(self, frame, bbox):
        """Extract 512D face embedding from InsightFace"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 20 or (y2-y1) < 20:
                return None
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
            faces = self.app.get(face_crop)
            if len(faces) == 0:
                return None
            
            best_face = max(faces, key=lambda x: x.det_score)
            return best_face.embedding
            
        except Exception as e:
            return None
    
    def compare_features(self, feat1, feat2):
        """Compare two face embeddings using cosine similarity"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            similarity = np.dot(feat1, feat2) / (
                np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6
            )
            similarity = (similarity + 1) / 2
            return similarity
        except:
            return 0.0


class MultiPersonTracker:
    """Multi-person pose tracker with registration and re-identification"""
    def __init__(self, track_thresh=0.5, match_thresh=0.65, reid_thresh=0.55, track_buffer=90, registration_window=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.reid_thresh = reid_thresh
        self.track_buffer = track_buffer
        self.registration_window = registration_window
        
        self.tracked_tracks = {}
        self.lost_tracks = {}
        self.face_database = {}
        self.registered_ids = set()
        
        self.frame_id = 0
        self.track_id_count = 0
        
        self.face_recognizer = InsightFaceRecognizer()
        self.face_extract_interval = 5
        self.last_face_extract_frame = {}
    
    def compute_iou(self, box1, box2):
        """Compute IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    def compute_pose_similarity(self, kps1, kps2):
        """Compute pose similarity"""
        visible1 = kps1[:, 2] > 0.5
        visible2 = kps2[:, 2] > 0.5
        visible_both = visible1 & visible2
        
        if not visible_both.any():
            return 0.0
        
        diff = kps1[visible_both, :2] - kps2[visible_both, :2]
        distances = np.linalg.norm(diff, axis=1)
        
        bbox_scale = np.sqrt(
            (kps1[:, 0].max() - kps1[:, 0].min()) * 
            (kps1[:, 1].max() - kps1[:, 1].min())
        ) + 1e-6
        
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) * 2
        variances = (sigmas[visible_both] * 2) ** 2
        oks = np.exp(-distances**2 / (2 * bbox_scale**2 * variances))
        
        return oks.mean()
    
    def match_detections_to_tracks(self, track_ids, detections, threshold):
        """Match detections to tracks using Hungarian algorithm"""
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))
        
        sim_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracked_tracks.get(track_id) or self.lost_tracks.get(track_id)
            if track is None:
                continue
                
            for j, det in enumerate(detections):
                iou = self.compute_iou(track['bbox'], det['bbox'])
                pose = self.compute_pose_similarity(track['keypoints'], det['keypoints'])
                sim_matrix[i, j] = 0.5 * iou + 0.5 * pose
        
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        
        matches = []
        for i, j in zip(row_ind, col_ind):
            if sim_matrix[i, j] >= threshold:
                matches.append((track_ids[i], j))
        
        matched_track_ids = [m[0] for m in matches]
        matched_det_ids = [m[1] for m in matches]
        unmatched_tracks = [t for t in track_ids if t not in matched_track_ids]
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_det_ids]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def get_face_bbox_from_keypoints(self, keypoints, person_bbox):
        """Get face region from pose keypoints"""
        head_kps = keypoints[:5]
        visible_kps = head_kps[head_kps[:, 2] > 0.5]
        
        if len(visible_kps) >= 2:
            xs, ys = visible_kps[:, 0], visible_kps[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            
            w, h = x_max - x_min, y_max - y_min
            size = max(w, h)
            padding = int(size * 0.8)
            
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = x_max + padding
            y_max = y_max + padding
            
            return (x_min, y_min, x_max, y_max)
        else:
            x1, y1, x2, y2 = map(int, person_bbox)
            height = y2 - y1
            return (x1, y1, x2, y1 + height // 3)
    
    def get_best_face_match(self, face_features):
        """Find best matching registered person"""
        if face_features is None or len(self.face_database) == 0:
            return None, 0.0
        
        best_track_id = None
        best_similarity = 0.0
        
        try:
            for track_id in self.registered_ids:
                if track_id not in self.face_database or self.face_database[track_id] is None:
                    continue
                
                similarity = self.face_recognizer.compare_features(
                    face_features, 
                    self.face_database[track_id]
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_track_id = track_id
        except Exception as e:
            return None, 0.0
        
        return best_track_id, best_similarity
    
    def update_face_database(self, track_id, new_features):
        """Update face database with exponential moving average"""
        if new_features is None:
            return
        
        if track_id not in self.face_database or self.face_database[track_id] is None:
            self.face_database[track_id] = new_features
        else:
            self.face_database[track_id] = 0.7 * self.face_database[track_id] + 0.3 * new_features
            self.face_database[track_id] = self.face_database[track_id] / (
                np.linalg.norm(self.face_database[track_id]) + 1e-6
            )
    
    def update(self, detections, frame):
        """Update tracker"""
        self.frame_id += 1
        
        high_dets = [d for d in detections if d['score'] >= self.track_thresh]
        allow_new_registrations = self.frame_id <= self.registration_window
        
        unmatched_tracks = list(self.tracked_tracks.keys())
        
        # Match existing tracks to detections
        matches, unmatched_tracks, unmatched_dets = self.match_detections_to_tracks(
            unmatched_tracks, 
            high_dets, 
            threshold=self.match_thresh
        )
        
        # Update matched tracks
        for track_id, det_idx in matches:
            det = high_dets[det_idx]
            self.tracked_tracks[track_id].update({
                'bbox': det['bbox'],
                'keypoints': det['keypoints'],
                'score': det['score'],
                'frame_id': self.frame_id,
            })
            
            # Update face database periodically
            if track_id not in self.last_face_extract_frame or \
               (self.frame_id - self.last_face_extract_frame[track_id]) >= self.face_extract_interval:
                face_bbox = self.get_face_bbox_from_keypoints(det['keypoints'], det['bbox'])
                face_features = self.face_recognizer.extract_face_features(frame, face_bbox)
                if face_features is not None:
                    self.update_face_database(track_id, face_features)
                    self.last_face_extract_frame[track_id] = self.frame_id
        
        # Handle unmatched detections
        if unmatched_dets:
            # Extract faces for unmatched detections
            for det_idx in unmatched_dets:
                det = high_dets[det_idx]
                face_bbox = self.get_face_bbox_from_keypoints(det['keypoints'], det['bbox'])
                det['face_features'] = self.face_recognizer.extract_face_features(frame, face_bbox)
            
            # Try to match with registered people using face recognition
            reid_matches = []
            for det_idx in unmatched_dets[:]:
                det = high_dets[det_idx]
                
                if det.get('face_features') is not None:
                    best_track_id, best_sim = self.get_best_face_match(det['face_features'])
                    
                    if best_track_id is not None and best_sim >= self.reid_thresh:
                        reid_matches.append((best_track_id, det_idx, best_sim))
            
            # Apply re-identification
            for track_id, det_idx, similarity in reid_matches:
                det = high_dets[det_idx]
                
                if track_id in self.lost_tracks:
                    self.tracked_tracks[track_id] = self.lost_tracks.pop(track_id)
                else:
                    self.tracked_tracks[track_id] = {'track_id': track_id}
                
                self.tracked_tracks[track_id].update({
                    'bbox': det['bbox'],
                    'keypoints': det['keypoints'],
                    'score': det['score'],
                    'frame_id': self.frame_id,
                })
                
                self.update_face_database(track_id, det['face_features'])
                self.last_face_extract_frame[track_id] = self.frame_id
                unmatched_dets.remove(det_idx)
            
            # Register new people (only during registration window)
            if allow_new_registrations:
                for det_idx in unmatched_dets:
                    det = high_dets[det_idx]
                    new_id = self.track_id_count
                    
                    self.tracked_tracks[new_id] = {
                        'track_id': new_id,
                        'bbox': det['bbox'],
                        'keypoints': det['keypoints'],
                        'score': det['score'],
                        'frame_id': self.frame_id,
                    }
                    
                    if det.get('face_features') is not None:
                        self.update_face_database(new_id, det['face_features'])
                        self.last_face_extract_frame[new_id] = self.frame_id
                    
                    self.registered_ids.add(new_id)
                    self.track_id_count += 1
        
        # Track lifecycle management
        for track_id in unmatched_tracks:
            self.lost_tracks[track_id] = self.tracked_tracks[track_id]
            del self.tracked_tracks[track_id]
        
        for track_id in list(self.lost_tracks.keys()):
            if self.frame_id - self.lost_tracks[track_id]['frame_id'] > self.track_buffer:
                del self.lost_tracks[track_id]
        
        return list(self.tracked_tracks.values())


# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50  # pixels distance to consider "touching"
score = 0

# Track colors for consistent player identification
track_colors = {}
def get_track_color(track_id):
    if track_id not in track_colors:
        np.random.seed(track_id * 42)
        track_colors[track_id] = tuple(map(int, np.random.randint(80, 255, 3)))
    return track_colors[track_id]

# Mapping body parts to YOLO pose keypoint indices
# YOLO keypoints: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
# 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
# 9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
# 13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
LANDMARK_MAP = {
    'Right Hand': 10,  # right_wrist
    'Left Hand': 9,    # left_wrist
    'Right Elbow': 8,  # right_elbow
    'Left Elbow': 7,   # left_elbow
    'Head': 0          # nose
}

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def generate_target(frame_width, frame_height):
    """Generate a random target position and body part"""
    margin = 100  # Keep targets away from edges
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def generate_targets_for_players(frame_width, frame_height, player_ids):
    """Generate targets for all active players"""
    targets = {}
    for player_id in player_ids:
        targets[player_id] = {
            'pos': generate_target(frame_width, frame_height)[0],
            'body_part': random.choice(BODY_PARTS),
            'touched': False
        }
    return targets

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    global score
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracker
    tracker = MultiPersonTracker(
        track_thresh=0.5,
        match_thresh=0.6,
        reid_thresh=0.55,
        track_buffer=90,
        registration_window=30
    )
    
    # Track targets for each player
    player_targets = {}
    
    print("Multi-Player Body Part Touch Game Started!")
    print("Press 'q' to quit, 'r' to reset registration")
    
    fps_counter = deque(maxlen=30)
    
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Detect poses with YOLO
        results = model(frame, verbose=False)[0]
        
        detections = []
        if results.boxes is not None and results.keypoints is not None:
            for box, kps in zip(results.boxes.data, results.keypoints.data):
                x1, y1, x2, y2, conf, _ = box
                keypoints = kps.cpu().numpy()
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(conf),
                    'keypoints': keypoints
                })
        
        # Track people
        tracks = tracker.update(detections, frame)
        
        # Get list of current tracked player IDs
        current_player_ids = [track['track_id'] for track in tracks]
        
        # Initialize targets for new players or when registration finishes
        if tracker.frame_id == tracker.registration_window + 1 and len(tracker.registered_ids) > 0:
            player_targets = generate_targets_for_players(frame_width, frame_height, tracker.registered_ids)
        
        # Add targets for any new players not yet in player_targets
        for player_id in current_player_ids:
            if player_id not in player_targets:
                player_targets[player_id] = {
                    'pos': generate_target(frame_width, frame_height)[0],
                    'body_part': random.choice(BODY_PARTS),
                    'touched': False
                }
        
        # Remove targets for players no longer tracked
        for player_id in list(player_targets.keys()):
            if player_id not in current_player_ids:
                del player_targets[player_id]
        
        # Draw all tracked people with skeletons
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track['keypoints']
            
            # Get consistent color for this player
            color = get_track_color(track_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f'Player {track_id}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw skeleton
            for kp in keypoints:
                if kp[2] > 0.5:  # confidence threshold
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            
            for connection in POSE_CONNECTIONS:
                if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                    kp1, kp2 = keypoints[connection[0]], keypoints[connection[1]]
                    if kp1[2] > 0.5 and kp2[2] > 0.5:
                        cv2.line(frame, (int(kp1[0]), int(kp1[1])),
                                (int(kp2[0]), int(kp2[1])), color, 2)
        
        # Game logic for all players
        for track in tracks:
            track_id = track['track_id']
            
            # Skip if no target for this player
            if track_id not in player_targets:
                continue
            
            keypoints = track['keypoints']
            target_info = player_targets[track_id]
            color = get_track_color(track_id)
            
            # Get the current target body part position
            landmark_idx = LANDMARK_MAP[target_info['body_part']]
            body_part_kp = keypoints[landmark_idx]
            
            # Check if keypoint is visible
            if body_part_kp[2] > 0.5:  # confidence threshold
                body_part_x = int(body_part_kp[0])
                body_part_y = int(body_part_kp[1])
                
                # Draw a circle on the tracked body part
                cv2.circle(frame, (body_part_x, body_part_y), 15, color, -1)
                
                # Calculate distance to target
                distance = calculate_distance((body_part_x, body_part_y), target_info['pos'])
                
                # Check if body part touches the target
                if distance < TOUCH_THRESHOLD:
                    target_info['touched'] = True
        
        # Check if ALL players have touched their targets
        if len(player_targets) > 0 and all(target['touched'] for target in player_targets.values()):
            score += 1
            # Generate new targets for all players
            player_targets = generate_targets_for_players(frame_width, frame_height, current_player_ids)
        
        # Draw targets for each player
        for track_id, target_info in player_targets.items():
            color = get_track_color(track_id)
            target_pos = target_info['pos']
            
            # Draw target with player's color
            # Solid circle if touched, hollow if not
            if target_info['touched']:
                cv2.circle(frame, target_pos, 30, color, -1)  # Filled
                cv2.circle(frame, target_pos, 30, (255, 255, 255), 2)  # White outline
            else:
                cv2.circle(frame, target_pos, 30, color, 3)  # Hollow
            cv2.circle(frame, target_pos, 5, color, -1)
            
            # Draw body part label near target
            label = f"P{track_id}: {target_info['body_part']}"
            cv2.putText(frame, label, (target_pos[0] - 50, target_pos[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw game UI
        score_text = f"Team Score: {score}"
        cv2.putText(frame, score_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Show how many players have touched their targets
        if len(player_targets) > 0:
            touched_count = sum(1 for target in player_targets.values() if target['touched'])
            progress_text = f"Progress: {touched_count}/{len(player_targets)} targets"
            cv2.putText(frame, progress_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show registration status
        if tracker.frame_id <= tracker.registration_window:
            progress = (tracker.frame_id / tracker.registration_window) * 100
            status = f'Registration: {progress:.0f}%'
            cv2.putText(frame, status, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            status = f'Players: {len(current_player_ids)}'
            cv2.putText(frame, status, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS counter
        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter)
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame_height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Body Part Touch Game', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracker and game
            tracker = MultiPersonTracker(
                track_thresh=0.5,
                match_thresh=0.6,
                reid_thresh=0.55,
                track_buffer=90,
                registration_window=30
            )
            player_targets = {}
            score = 0
            print("Game reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nGame Over! Final Team Score: {score}")

if __name__ == "__main__":
    main()
