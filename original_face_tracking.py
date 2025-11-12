"""
Multi-person pose tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import time

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
            
            # Detect faces and extract embeddings
            faces = self.app.get(face_crop)
            
            if len(faces) == 0:
                return None
            
            # Get the face with highest detection score
            best_face = max(faces, key=lambda x: x.det_score)
            
            # Return 512D embedding
            return best_face.embedding
            
        except Exception as e:
            return None
    
    def compare_features(self, feat1, feat2):
        """Compare two face embeddings using cosine similarity"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (
                np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6
            )
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
            return similarity
        except:
            return 0.0


class RegistrationOnlyTracker:
    """ByteTrack with registration-only calibration"""
    def __init__(self, track_thresh=0.5, match_thresh=0.65, reid_thresh=0.55, track_buffer=90):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.reid_thresh = reid_thresh  # Threshold for re-identification
        self.track_buffer = track_buffer
        
        self.tracked_tracks = {}
        self.lost_tracks = {}
        self.removed_tracks = {}
        
        # Face database: ONLY contains registered people
        self.face_database = {}
        self.registered_ids = set()  # IDs registered during calibration
        
        self.frame_id = 0
        self.track_id_count = 0
        
        # Face recognizer
        self.face_recognizer = InsightFaceRecognizer()
        
        # Calibration
        self.calibration_mode = True
        self.calibration_frames = 0
        self.max_calibration_frames = 90  # 3 seconds for registration
        
        # Performance optimization
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
        """Find best matching REGISTERED person"""
        if face_features is None or len(self.face_database) == 0:
            return None, 0.0
        
        best_track_id = None
        best_similarity = 0.0
        
        try:
            # ONLY search registered IDs
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
            # Exponential moving average
            self.face_database[track_id] = 0.7 * self.face_database[track_id] + 0.3 * new_features
            # Renormalize
            self.face_database[track_id] = self.face_database[track_id] / (
                np.linalg.norm(self.face_database[track_id]) + 1e-6
            )
    
    def update(self, detections, frame):
        """Update tracker"""
        self.frame_id += 1
        
        high_dets = [d for d in detections if d['score'] >= self.track_thresh]
        
        # ==================== CALIBRATION MODE ====================
        if self.calibration_mode:
            self.calibration_frames += 1
            
            # Extract faces during calibration
            for det in high_dets:
                face_bbox = self.get_face_bbox_from_keypoints(det['keypoints'], det['bbox'])
                det['face_features'] = self.face_recognizer.extract_face_features(frame, face_bbox)
            
            # Simple tracking during calibration
            unmatched_tracks = list(self.tracked_tracks.keys())
            unmatched_dets = list(range(len(high_dets)))
            matches = []
            
            if unmatched_tracks and unmatched_dets:
                sim_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)))
                
                for i, track_id in enumerate(unmatched_tracks):
                    track = self.tracked_tracks[track_id]
                    for j, det_idx in enumerate(unmatched_dets):
                        det = high_dets[det_idx]
                        iou = self.compute_iou(track['bbox'], det['bbox'])
                        pose = self.compute_pose_similarity(track['keypoints'], det['keypoints'])
                        sim_matrix[i, j] = 0.6 * iou + 0.4 * pose
                
                row_ind, col_ind = linear_sum_assignment(-sim_matrix)
                
                for i, j in zip(row_ind, col_ind):
                    if sim_matrix[i, j] >= 0.5:
                        matches.append((unmatched_tracks[i], unmatched_dets[j]))
                
                matched_det_ids = [m[1] for m in matches]
                unmatched_dets = [d for d in unmatched_dets if d not in matched_det_ids]
            
            # Update matched tracks
            for track_id, det_idx in matches:
                det = high_dets[det_idx]
                self.tracked_tracks[track_id].update({
                    'bbox': det['bbox'],
                    'keypoints': det['keypoints'],
                    'score': det['score'],
                    'frame_id': self.frame_id,
                })
                self.update_face_database(track_id, det['face_features'])
            
            # Create new tracks during calibration (REGISTRATION)
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
                self.update_face_database(new_id, det['face_features'])
                self.registered_ids.add(new_id)
                self.track_id_count += 1
            
            # End calibration
            if self.calibration_frames >= self.max_calibration_frames:
                self.calibration_mode = False
                for tid in sorted(self.registered_ids):
                    has_face = tid in self.face_database and self.face_database[tid] is not None

            return list(self.tracked_tracks.values())
        
        # ==================== NORMAL MODE (POST-CALIBRATION) ====================
        # Only match to registered people - no new IDs created
        
        unmatched_tracks = list(self.tracked_tracks.keys())
        unmatched_dets = list(range(len(high_dets)))
        matches = []
        
        # First pass: Fast matching (IoU + Pose) for active tracks
        if unmatched_tracks and unmatched_dets:
            iou_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)))
            pose_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)))
            
            for i, track_id in enumerate(unmatched_tracks):
                track = self.tracked_tracks[track_id]
                for j, det_idx in enumerate(unmatched_dets):
                    det = high_dets[det_idx]
                    iou_matrix[i, j] = self.compute_iou(track['bbox'], det['bbox'])
                    pose_matrix[i, j] = self.compute_pose_similarity(track['keypoints'], det['keypoints'])
            
            sim_matrix = 0.5 * iou_matrix + 0.5 * pose_matrix
            
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            
            for i, j in zip(row_ind, col_ind):
                if sim_matrix[i, j] >= self.match_thresh:
                    matches.append((unmatched_tracks[i], unmatched_dets[j]))
            
            matched_track_ids = [m[0] for m in matches]
            matched_det_ids = [m[1] for m in matches]
            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_track_ids]
            unmatched_dets = [d for d in unmatched_dets if d not in matched_det_ids]
        
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
        
        # Second pass: RE-IDENTIFICATION (only for registered people)
        if unmatched_dets:
            # Extract faces for unmatched detections
            for det_idx in unmatched_dets:
                det = high_dets[det_idx]
                face_bbox = self.get_face_bbox_from_keypoints(det['keypoints'], det['bbox'])
                det['face_features'] = self.face_recognizer.extract_face_features(frame, face_bbox)
            
            # Try to match with ALL registered people (active, lost, or removed)
            all_inactive = {**self.lost_tracks, **self.removed_tracks}
            
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
                
                # Reactivate from inactive or create if completely lost
                if track_id in all_inactive:
                    self.tracked_tracks[track_id] = all_inactive[track_id]
                else:
                    # Create fresh track for registered person
                    self.tracked_tracks[track_id] = {
                        'track_id': track_id,
                    }
                
                self.tracked_tracks[track_id].update({
                    'bbox': det['bbox'],
                    'keypoints': det['keypoints'],
                    'score': det['score'],
                    'frame_id': self.frame_id,
                })
                
                if track_id in self.lost_tracks:
                    del self.lost_tracks[track_id]
                if track_id in self.removed_tracks:
                    del self.removed_tracks[track_id]
                
                self.update_face_database(track_id, det['face_features'])
                self.last_face_extract_frame[track_id] = self.frame_id
                unmatched_dets.remove(det_idx)
        
        # Move unmatched tracks to lost
        for track_id in unmatched_tracks:
            self.lost_tracks[track_id] = self.tracked_tracks[track_id]
            del self.tracked_tracks[track_id]
        
        # Move old lost tracks to removed (but keep in database)
        for track_id in list(self.lost_tracks.keys()):
            if self.frame_id - self.lost_tracks[track_id]['frame_id'] > self.track_buffer:
                self.removed_tracks[track_id] = self.lost_tracks[track_id]
                del self.lost_tracks[track_id]
        
        return list(self.tracked_tracks.values())


# Initialize
model = YOLO('yolov8n-pose.pt')
tracker = RegistrationOnlyTracker(
    track_thresh=0.5,
    match_thresh=0.6,
    reid_thresh=0.55,  # Threshold for re-identification
    track_buffer=90
)

cap = cv2.VideoCapture(0)

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

track_colors = {}
def get_track_color(track_id):
    if track_id not in track_colors:
        np.random.seed(track_id * 42)
        track_colors[track_id] = tuple(map(int, np.random.randint(80, 255, 3)))
    return track_colors[track_id]

fps_counter = deque(maxlen=30)

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect poses
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
    
    # Track
    tracks = tracker.update(detections, frame)
    
    # Draw
    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        keypoints = track['keypoints']
        color = get_track_color(track_id)
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f'Person {track_id}'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Skeleton
        for kp in keypoints:
            if kp[2] > 0.5:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
        
        for connection in POSE_CONNECTIONS:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                kp1, kp2 = keypoints[connection[0]], keypoints[connection[1]]
                if kp1[2] > 0.5 and kp2[2] > 0.5:
                    cv2.line(frame, (int(kp1[0]), int(kp1[1])),
                            (int(kp2[0]), int(kp2[1])), color, 2)
    
    # Status
    if tracker.calibration_mode:
        progress = (tracker.calibration_frames / tracker.max_calibration_frames) * 100
        status = f'REGISTERING... {progress:.0f}%'
        cv2.putText(frame, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        bar_w = 400
        cv2.rectangle(frame, (10, 60), (10 + bar_w, 80), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 60), (10 + int(bar_w * progress / 100), 80), (0, 255, 0), -1)
        
        # Show registered count
        reg_text = f'Registered: {len(tracker.registered_ids)} people'
        cv2.putText(frame, reg_text, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        status = f'Tracking: {len(tracks)}/{len(tracker.registered_ids)} registered'
        cv2.putText(frame, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # FPS
    fps = 1.0 / (time.time() - start_time)
    fps_counter.append(fps)
    avg_fps = np.mean(fps_counter)
    cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Registration-Only Tracking', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and not tracker.calibration_mode:
        # Reset and re-register
        tracker = RegistrationOnlyTracker(
            track_thresh=0.5,
            match_thresh=0.6,
            reid_thresh=0.55,
            track_buffer=90
        )

cap.release()
cv2.destroyAllWindows()