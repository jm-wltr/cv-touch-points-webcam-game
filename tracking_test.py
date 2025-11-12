#!/usr/bin/env python3
"""
Body Tracking Consistency Test
Compare tracking quality between different smoothing techniques
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

class TrackingTest:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            smooth_landmarks=True
        )
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Tracking data storage
        self.raw_positions = deque(maxlen=300)
        self.smoothed_positions = deque(maxlen=300)
        self.kalman_positions = deque(maxlen=300)
        
        # Simple Kalman filter state
        self.kalman_x = None
        self.kalman_y = None
        self.kalman_vx = 0
        self.kalman_vy = 0
        
        # Smoothing buffer
        self.position_buffer = deque(maxlen=5)
        
    def simple_kalman_update(self, x, y, dt=1.0/30):
        """Simple 2D Kalman filter"""
        if self.kalman_x is None:
            self.kalman_x = x
            self.kalman_y = y
            return x, y
        
        # Predict
        predicted_x = self.kalman_x + self.kalman_vx * dt
        predicted_y = self.kalman_y + self.kalman_vy * dt
        
        # Update
        alpha = 0.2  # Smoothing factor
        self.kalman_x = predicted_x * (1 - alpha) + x * alpha
        self.kalman_y = predicted_y * (1 - alpha) + y * alpha
        
        # Update velocity
        self.kalman_vx = (self.kalman_x - predicted_x) / dt * 0.1
        self.kalman_vy = (self.kalman_y - predicted_y) / dt * 0.1
        
        return self.kalman_x, self.kalman_y
    
    def moving_average_smooth(self, x, y):
        """Moving average smoothing"""
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) > 0:
            avg_x = sum(p[0] for p in self.position_buffer) / len(self.position_buffer)
            avg_y = sum(p[1] for p in self.position_buffer) / len(self.position_buffer)
            return avg_x, avg_y
        
        return x, y
    
    def process_frame(self, frame):
        """Process single frame and extract tracking data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        seg_results = self.segmentation.process(rgb_frame)
        
        if results.pose_landmarks:
            # Track right hand
            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Raw position
            raw_x = right_hand.x * frame.shape[1]
            raw_y = right_hand.y * frame.shape[0]
            self.raw_positions.append((raw_x, raw_y))
            
            # Moving average smoothed
            smooth_x, smooth_y = self.moving_average_smooth(raw_x, raw_y)
            self.smoothed_positions.append((smooth_x, smooth_y))
            
            # Kalman filtered
            kalman_x, kalman_y = self.simple_kalman_update(raw_x, raw_y)
            self.kalman_positions.append((kalman_x, kalman_y))
            
            return True, (raw_x, raw_y), (smooth_x, smooth_y), (kalman_x, kalman_y)
        
        return False, None, None, None
    
    def calculate_jitter(self, positions):
        """Calculate jitter metric (sum of position changes)"""
        if len(positions) < 2:
            return 0
        
        jitter = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            jitter += np.sqrt(dx*dx + dy*dy)
        
        return jitter / len(positions)
    
    def run_test(self, duration=10):
        """Run tracking test for specified duration"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow('Tracking Test', cv2.WINDOW_NORMAL)
        
        print(f"Starting {duration} second tracking test...")
        print("Move your right hand slowly in a circle or figure-8 pattern")
        print("Press 'Q' to quit early\n")
        
        start_time = time.time()
        frame_count = 0
        
        while cap.isOpened() and (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            success, raw_pos, smooth_pos, kalman_pos = self.process_frame(frame)
            
            # Visualize different tracking methods
            if success:
                # Draw raw position (red)
                cv2.circle(frame, (int(raw_pos[0]), int(raw_pos[1])), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Raw", (int(raw_pos[0])+15, int(raw_pos[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw smoothed position (green)
                cv2.circle(frame, (int(smooth_pos[0]), int(smooth_pos[1])), 10, (0, 255, 0), -1)
                cv2.putText(frame, "Smoothed", (int(smooth_pos[0])+15, int(smooth_pos[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw Kalman position (blue)
                cv2.circle(frame, (int(kalman_pos[0]), int(kalman_pos[1])), 10, (255, 0, 0), -1)
                cv2.putText(frame, "Kalman", (int(kalman_pos[0])+15, int(kalman_pos[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw UI
            remaining = duration - (time.time() - start_time)
            cv2.putText(frame, f"Time: {remaining:.1f}s", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Red: Raw | Green: Smoothed | Blue: Kalman", (20, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow('Tracking Test', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate and display metrics
        print("\n=== Tracking Test Results ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
        
        if len(self.raw_positions) > 10:
            raw_jitter = self.calculate_jitter(list(self.raw_positions))
            smooth_jitter = self.calculate_jitter(list(self.smoothed_positions))
            kalman_jitter = self.calculate_jitter(list(self.kalman_positions))
            
            print(f"\nJitter Analysis (lower is better):")
            print(f"  Raw tracking:      {raw_jitter:.2f} pixels/frame")
            print(f"  Moving average:    {smooth_jitter:.2f} pixels/frame ({(1-smooth_jitter/raw_jitter)*100:.1f}% improvement)")
            print(f"  Kalman filter:     {kalman_jitter:.2f} pixels/frame ({(1-kalman_jitter/raw_jitter)*100:.1f}% improvement)")
            
            # Plot results
            self.plot_results()
        else:
            print("\nInsufficient data collected for analysis")
    
    def plot_results(self):
        """Plot tracking comparison"""
        if len(self.raw_positions) < 10:
            return
        
        # Extract x and y coordinates
        raw_x = [p[0] for p in self.raw_positions]
        raw_y = [p[1] for p in self.raw_positions]
        smooth_x = [p[0] for p in self.smoothed_positions]
        smooth_y = [p[1] for p in self.smoothed_positions]
        kalman_x = [p[0] for p in self.kalman_positions]
        kalman_y = [p[1] for p in self.kalman_positions]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot X coordinates over time
        axes[0, 0].plot(raw_x, 'r-', alpha=0.5, label='Raw')
        axes[0, 0].plot(smooth_x, 'g-', alpha=0.7, label='Smoothed')
        axes[0, 0].plot(kalman_x, 'b-', alpha=0.7, label='Kalman')
        axes[0, 0].set_title('X Position Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('X Position (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Y coordinates over time
        axes[0, 1].plot(raw_y, 'r-', alpha=0.5, label='Raw')
        axes[0, 1].plot(smooth_y, 'g-', alpha=0.7, label='Smoothed')
        axes[0, 1].plot(kalman_y, 'b-', alpha=0.7, label='Kalman')
        axes[0, 1].set_title('Y Position Over Time')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Y Position (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 2D path
        axes[1, 0].plot(raw_x, raw_y, 'r-', alpha=0.3, label='Raw', linewidth=1)
        axes[1, 0].plot(smooth_x, smooth_y, 'g-', alpha=0.6, label='Smoothed', linewidth=2)
        axes[1, 0].plot(kalman_x, kalman_y, 'b-', alpha=0.6, label='Kalman', linewidth=2)
        axes[1, 0].set_title('2D Tracking Path')
        axes[1, 0].set_xlabel('X Position (pixels)')
        axes[1, 0].set_ylabel('Y Position (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()  # Invert Y to match image coordinates
        
        # Plot jitter (position changes)
        raw_changes = [np.sqrt((raw_x[i]-raw_x[i-1])**2 + (raw_y[i]-raw_y[i-1])**2) 
                      for i in range(1, len(raw_x))]
        smooth_changes = [np.sqrt((smooth_x[i]-smooth_x[i-1])**2 + (smooth_y[i]-smooth_y[i-1])**2) 
                         for i in range(1, len(smooth_x))]
        kalman_changes = [np.sqrt((kalman_x[i]-kalman_x[i-1])**2 + (kalman_y[i]-kalman_y[i-1])**2) 
                         for i in range(1, len(kalman_x))]
        
        axes[1, 1].plot(raw_changes, 'r-', alpha=0.5, label='Raw')
        axes[1, 1].plot(smooth_changes, 'g-', alpha=0.7, label='Smoothed')
        axes[1, 1].plot(kalman_changes, 'b-', alpha=0.7, label='Kalman')
        axes[1, 1].set_title('Frame-to-Frame Movement (Jitter)')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Movement (pixels)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Body Tracking Consistency Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('/home/claude/tracking_analysis.png', dpi=150)
        print("\nAnalysis plot saved as 'tracking_analysis.png'")
        
        plt.show()

def main():
    print("=" * 50)
    print("Body Tracking Consistency Test")
    print("=" * 50)
    print("\nThis test compares different tracking smoothing methods:")
    print("1. Raw MediaPipe output (no smoothing)")
    print("2. Moving average smoothing")
    print("3. Simple Kalman filter")
    print("\nThe test will run for 10 seconds.")
    print("Move your right hand slowly in smooth patterns.\n")
    
    input("Press Enter to start the test...")
    
    test = TrackingTest()
    test.run_test(duration=10)

if __name__ == "__main__":
    main()
