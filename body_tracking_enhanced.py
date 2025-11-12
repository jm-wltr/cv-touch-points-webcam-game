import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from scipy import ndimage
from scipy.interpolate import interp1d
import threading
import queue

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# Enhanced tracking models
pose_tracker = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use most accurate model
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)

segmentation_model = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1  # Use more accurate model
)

# Game Configuration
class GameConfig:
    # Display settings
    WINDOW_NAME = "Body Tracking 2D - Enhanced"
    TARGET_FPS = 30
    
    # Game mechanics
    BODY_PARTS = [
        'Right Hand', 'Left Hand', 
        'Right Elbow', 'Left Elbow',
        'Head', 'Right Shoulder', 'Left Shoulder',
        'Right Knee', 'Left Knee'
    ]
    
    TOUCH_THRESHOLD = 45
    TARGET_RADIUS = 30
    TRACKER_RADIUS = 20
    
    # Visual settings
    BODY_COLOR = (100, 255, 100)  # Light green
    BODY_OUTLINE_COLOR = (0, 150, 0)  # Dark green
    TARGET_COLOR = (0, 100, 255)  # Red-orange
    TRACKER_COLOR = (255, 100, 255)  # Pink
    
    # Smoothing parameters
    KALMAN_PROCESS_NOISE = 0.03
    KALMAN_MEASUREMENT_NOISE = 0.1
    CONTOUR_SMOOTH_FACTOR = 5
    POSITION_BUFFER_SIZE = 7

# Landmark mapping
POSE_LANDMARKS = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE,
    'Right Shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'Left Shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'Right Knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'Left Knee': mp_pose.PoseLandmark.LEFT_KNEE
}

class KalmanFilter2D:
    """2D Kalman filter for smooth position tracking"""
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.dt = 1.0  # Time step
        
        # State transition matrix
        self.F = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # Initial covariance
        self.P = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def update(self, measurement):
        """Update filter with new measurement"""
        if not self.initialized:
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self.initialized = True
            return measurement
        
        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return (float(self.x[0]), float(self.x[1]))

class BodyTracker:
    """Enhanced body tracking with multiple smoothing techniques"""
    def __init__(self):
        self.kalman_filters = {part: KalmanFilter2D() for part in GameConfig.BODY_PARTS}
        self.position_buffers = {part: deque(maxlen=GameConfig.POSITION_BUFFER_SIZE) 
                                 for part in GameConfig.BODY_PARTS}
        self.last_valid_positions = {}
        self.confidence_scores = {part: 0.0 for part in GameConfig.BODY_PARTS}
        
    def update_position(self, body_part, position, confidence=1.0):
        """Update tracked position with smoothing"""
        if position is None:
            return self.last_valid_positions.get(body_part)
        
        # Apply Kalman filter
        filtered_pos = self.kalman_filters[body_part].update(position)
        
        # Add to buffer for averaging
        self.position_buffers[body_part].append(filtered_pos)
        
        # Calculate weighted average
        if len(self.position_buffers[body_part]) > 0:
            weights = np.linspace(0.5, 1.0, len(self.position_buffers[body_part]))
            weights = weights / weights.sum()
            
            positions = list(self.position_buffers[body_part])
            avg_x = sum(p[0] * w for p, w in zip(positions, weights))
            avg_y = sum(p[1] * w for p, w in zip(positions, weights))
            
            smoothed_pos = (int(avg_x), int(avg_y))
            self.last_valid_positions[body_part] = smoothed_pos
            self.confidence_scores[body_part] = confidence
            
            return smoothed_pos
        
        return None
    
    def get_confidence(self, body_part):
        """Get tracking confidence for a body part"""
        return self.confidence_scores.get(body_part, 0.0)

class ContourProcessor:
    """Process and smooth body contours"""
    
    @staticmethod
    def smooth_contour(contour, factor=5):
        """Smooth contour points using spline interpolation"""
        if len(contour) < 4:
            return contour
        
        # Extract x and y coordinates
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        
        # Create parameter t
        t = np.linspace(0, 1, len(x))
        
        # Interpolate with more points
        t_smooth = np.linspace(0, 1, len(x) * factor)
        
        try:
            # Use cubic spline interpolation
            fx = interp1d(t, x, kind='cubic', assume_sorted=True)
            fy = interp1d(t, y, kind='cubic', assume_sorted=True)
            
            x_smooth = fx(t_smooth)
            y_smooth = fy(t_smooth)
            
            # Reshape back to contour format
            smooth_contour = np.array([[[int(x), int(y)]] 
                                      for x, y in zip(x_smooth, y_smooth)], dtype=np.int32)
            
            # Simplify to reduce points while maintaining smoothness
            epsilon = 2.0
            smooth_contour = cv2.approxPolyDP(smooth_contour, epsilon, True)
            
            return smooth_contour
        except:
            return contour
    
    @staticmethod
    def create_body_mask(segmentation_mask, frame_shape):
        """Create high-quality body mask from segmentation"""
        if segmentation_mask is None:
            return None
        
        # Convert to binary mask with threshold
        binary_mask = (segmentation_mask > 0.6).astype(np.uint8) * 255
        
        # Apply bilateral filter for edge-preserving smoothing
        binary_mask = cv2.bilateralFilter(binary_mask, 9, 75, 75)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Final threshold
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        
        return binary_mask
    
    @staticmethod
    def get_smooth_body_contour(mask):
        """Extract and smooth the main body contour"""
        if mask is None:
            return None
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (main body)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Smooth the contour
        smooth_contour = ContourProcessor.smooth_contour(main_contour, factor=3)
        
        return smooth_contour

class BodyRenderer:
    """Render body visualization with effects"""
    
    @staticmethod
    def apply_body_effect(frame, mask, contour, color=GameConfig.BODY_COLOR, opacity=0.7):
        """Apply enhanced body visualization"""
        if mask is None:
            return frame
        
        result = frame.copy()
        
        # Create gradient fill for body
        overlay = np.zeros_like(frame)
        
        # Fill body with gradient effect
        if contour is not None:
            # Create gradient mask
            h, w = mask.shape
            gradient = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Base color
            gradient[:] = color
            
            # Add subtle gradient
            for i in range(h):
                factor = 0.8 + 0.2 * (i / h)
                gradient[i] = tuple(int(c * factor) for c in color)
            
            # Apply mask
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            overlay = (gradient * mask_3d).astype(np.uint8)
        
        # Blend with original
        cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0, result)
        
        # Draw smooth outline
        if contour is not None:
            # Draw with anti-aliasing
            cv2.drawContours(result, [contour], -1, GameConfig.BODY_OUTLINE_COLOR, 3, cv2.LINE_AA)
            
            # Add glow effect
            glow = np.zeros_like(frame)
            cv2.drawContours(glow, [contour], -1, GameConfig.BODY_OUTLINE_COLOR, 8, cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (15, 15), 5)
            cv2.addWeighted(result, 1, glow, 0.3, 0, result)
        
        return result
    
    @staticmethod
    def draw_tracker(frame, position, radius=GameConfig.TRACKER_RADIUS, color=GameConfig.TRACKER_COLOR):
        """Draw body part tracker with effects"""
        if position is None:
            return
        
        x, y = position
        
        # Draw outer glow
        glow_radius = radius + 10
        for i in range(3):
            alpha = 0.1 * (3 - i)
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), glow_radius - i*3, color, -1)
            cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0, frame)
        
        # Draw main circle
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Inner highlight
        highlight_offset = radius // 3
        cv2.circle(frame, (x - highlight_offset, y - highlight_offset), 
                  radius // 4, (255, 255, 255), -1)
    
    @staticmethod
    def draw_target(frame, position, radius=GameConfig.TARGET_RADIUS, pulse_factor=1.0):
        """Draw animated target"""
        if position is None:
            return
        
        x, y = position
        animated_radius = int(radius * pulse_factor)
        
        # Draw target rings
        for i in range(3):
            ring_radius = animated_radius - i * 8
            if ring_radius > 0:
                thickness = 3 - i
                cv2.circle(frame, (x, y), ring_radius, GameConfig.TARGET_COLOR, thickness, cv2.LINE_AA)
        
        # Center dot
        cv2.circle(frame, (x, y), 5, GameConfig.TARGET_COLOR, -1)

class Game:
    """Main game class"""
    def __init__(self):
        self.body_tracker = BodyTracker()
        self.score = 0
        self.high_score = 0
        self.combo_multiplier = 1.0
        self.last_touch_time = 0
        self.target_position = None
        self.target_body_part = None
        self.show_silhouette = True
        self.debug_mode = False
        
    def generate_target(self, width, height):
        """Generate new target"""
        margin = 100
        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)
        body_part = random.choice(GameConfig.BODY_PARTS)
        return (x, y), body_part
    
    def check_collision(self, tracker_pos, target_pos):
        """Check if tracker touches target"""
        if tracker_pos is None or target_pos is None:
            return False
        
        distance = np.sqrt((tracker_pos[0] - target_pos[0])**2 + 
                          (tracker_pos[1] - target_pos[1])**2)
        return distance < GameConfig.TOUCH_THRESHOLD
    
    def update_score(self, points):
        """Update score with combo system"""
        current_time = time.time()
        
        # Update combo
        if current_time - self.last_touch_time < 2.0:
            self.combo_multiplier = min(self.combo_multiplier + 0.5, 5.0)
        else:
            self.combo_multiplier = 1.0
        
        # Calculate points
        final_points = int(points * self.combo_multiplier)
        self.score += final_points
        self.high_score = max(self.high_score, self.score)
        self.last_touch_time = current_time
        
        return final_points
    
    def draw_hud(self, frame, width, height):
        """Draw game HUD"""
        # Score
        cv2.putText(frame, f"Score: {self.score}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # High score
        cv2.putText(frame, f"Best: {self.high_score}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Target instruction
        if self.target_body_part:
            instruction = f"Touch with: {self.target_body_part}"
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            
            # Background
            cv2.rectangle(frame, (text_x - 10, 10), 
                         (text_x + text_size[0] + 10, 60), 
                         (0, 0, 0), -1)
            cv2.putText(frame, instruction, (text_x, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Combo indicator
        if self.combo_multiplier > 1:
            combo_text = f"COMBO x{self.combo_multiplier:.1f}"
            cv2.putText(frame, combo_text, (width - 250, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
        
        # Controls
        controls = "ESC: Menu | S: Silhouette | D: Debug | Q: Quit"
        cv2.putText(frame, controls, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run the game"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow(GameConfig.WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        # Initialize game
        self.target_position, self.target_body_part = self.generate_target(width, height)
        target_spawn_time = time.time()
        
        print(f"\n{GameConfig.WINDOW_NAME}")
        print(f"Resolution: {width}x{height}")
        print("\nControls:")
        print("  S - Toggle silhouette")
        print("  D - Toggle debug mode")
        print("  Q - Quit game\n")
        
        frame_count = 0
        fps_timer = time.time()
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            pose_results = pose_tracker.process(rgb_frame)
            
            # Process segmentation
            seg_results = segmentation_model.process(rgb_frame)
            
            # Apply body effect if enabled
            if self.show_silhouette and seg_results.segmentation_mask is not None:
                mask = ContourProcessor.create_body_mask(
                    seg_results.segmentation_mask, frame.shape[:2])
                contour = ContourProcessor.get_smooth_body_contour(mask)
                frame = BodyRenderer.apply_body_effect(frame, mask, contour)
            
            # Track target body part
            tracker_position = None
            if pose_results.pose_landmarks and self.target_body_part in POSE_LANDMARKS:
                landmark = POSE_LANDMARKS[self.target_body_part]
                pose_landmark = pose_results.pose_landmarks.landmark[landmark]
                
                # Get position with confidence
                raw_x = int(pose_landmark.x * width)
                raw_y = int(pose_landmark.y * height)
                confidence = pose_landmark.visibility
                
                # Update tracker with smoothing
                tracker_position = self.body_tracker.update_position(
                    self.target_body_part, (raw_x, raw_y), confidence)
                
                # Check collision
                if self.check_collision(tracker_position, self.target_position):
                    points = self.update_score(10)
                    
                    # Visual feedback
                    cv2.putText(frame, f"+{points}!", self.target_position,
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Generate new target
                    self.target_position, self.target_body_part = self.generate_target(width, height)
                    target_spawn_time = time.time()
            
            # Draw target with animation
            pulse = abs(np.sin((time.time() - target_spawn_time) * 4)) * 0.3 + 0.8
            BodyRenderer.draw_target(frame, self.target_position, pulse_factor=pulse)
            
            # Draw tracker
            if tracker_position:
                BodyRenderer.draw_tracker(frame, tracker_position)
                
                # Draw connection line in debug mode
                if self.debug_mode and self.target_position:
                    cv2.line(frame, tracker_position, self.target_position,
                            (100, 100, 100), 1, cv2.LINE_AA)
            
            # Draw HUD
            self.draw_hud(frame, width, height)
            
            # Debug information
            if self.debug_mode:
                debug_y = 150
                debug_info = [
                    f"FPS: {fps:.1f}",
                    f"Target: {self.target_body_part}",
                    f"Position: {tracker_position}",
                    f"Confidence: {self.body_tracker.get_confidence(self.target_body_part):.2f}"
                ]
                for info in debug_info:
                    cv2.putText(frame, info, (20, debug_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    debug_y += 30
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
            
            cv2.imshow(GameConfig.WINDOW_NAME, frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_silhouette = not self.show_silhouette
                print(f"Silhouette: {'ON' if self.show_silhouette else 'OFF'}")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug: {'ON' if self.debug_mode else 'OFF'}")
            elif key == 27:  # ESC
                # Could implement menu here
                pass
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose_tracker.close()
        segmentation_model.close()
        
        print(f"\nGame Over!")
        print(f"Final Score: {self.score}")
        print(f"High Score: {self.high_score}")

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
