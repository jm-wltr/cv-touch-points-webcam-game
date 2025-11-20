#!/usr/bin/env python3
"""
Two-Player Competitive Body Tracking Game
Players compete to touch targets faster than their opponent
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from enum import Enum
import threading
import queue

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Game Configuration
class GameConfig:
    # Display
    WINDOW_NAME = "2-Player Body Tracking Battle"
    TARGET_FPS = 30
    
    # Game mechanics
    BODY_PARTS = ['Right Hand', 'Left Hand', 'Head']
    TOUCH_THRESHOLD = 50
    TARGET_RADIUS = 35
    TRACKER_RADIUS = 20
    
    # Scoring
    POINTS_PER_TARGET = 10
    SPEED_BONUS_THRESHOLD = 1.5  # seconds for speed bonus
    SPEED_BONUS_POINTS = 5
    WIN_SCORE = 100
    
    # Player colors (BGR format)
    PLAYER_COLORS = {
        1: (255, 100, 100),  # Blue
        2: (100, 255, 100)   # Green
    }
    
    PLAYER_NAMES = {
        1: "Blue Player",
        2: "Green Player"
    }

# Landmark mapping
LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Head': mp_pose.PoseLandmark.NOSE,
}

class GameMode(Enum):
    MENU = 1
    COOPERATIVE = 2
    COMPETITIVE = 3
    RACE = 4
    MIRROR = 5

class Player:
    """Represents a player in the game"""
    def __init__(self, player_id, side='left'):
        self.id = player_id
        self.side = side  # 'left' or 'right' side of screen
        self.score = 0
        self.color = GameConfig.PLAYER_COLORS[player_id]
        self.name = GameConfig.PLAYER_NAMES[player_id]
        self.pose_tracker = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        self.position_history = {part: deque(maxlen=5) for part in GameConfig.BODY_PARTS}
        self.current_target = None
        self.target_spawn_time = 0
        self.combo_multiplier = 1.0
        self.last_touch_time = 0
        self.targets_hit = 0
        self.is_active = False
        self.body_parts_positions = {}
        
    def update_position(self, body_part, position):
        """Update tracked position with smoothing"""
        if position is None:
            return None
        
        self.position_history[body_part].append(position)
        
        if len(self.position_history[body_part]) > 0:
            # Average positions for smoothing
            positions = list(self.position_history[body_part])
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            return (int(avg_x), int(avg_y))
        
        return None
    
    def reset_target(self):
        """Reset player's target"""
        self.current_target = None
        self.target_spawn_time = time.time()
    
    def add_score(self, points):
        """Add points to player's score"""
        self.score += points
        self.targets_hit += 1

class TwoPlayerGame:
    """Main two-player game controller"""
    def __init__(self):
        self.players = {
            1: Player(1, 'left'),
            2: Player(2, 'right')
        }
        self.game_mode = GameMode.COMPETITIVE
        self.game_active = False
        self.winner = None
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.frame_count = 0
        self.start_time = time.time()
        
    def detect_players(self, frame):
        """Detect and separate two players in the frame"""
        h, w = frame.shape[:2]
        mid_x = w // 2
        
        # Split frame for each player
        left_frame = frame[:, :mid_x]
        right_frame = frame[:, mid_x:]
        
        # Process each player's half
        rgb_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        
        # Detect poses
        results_left = self.players[1].pose_tracker.process(rgb_left)
        results_right = self.players[2].pose_tracker.process(rgb_right)
        
        # Update player active status
        self.players[1].is_active = results_left.pose_landmarks is not None
        self.players[2].is_active = results_right.pose_landmarks is not None
        
        return results_left, results_right, mid_x
    
    def generate_target_position(self, player_id, frame_width, frame_height):
        """Generate target position for a specific player"""
        mid_x = frame_width // 2
        margin = 60
        
        if self.players[player_id].side == 'left':
            x = random.randint(margin, mid_x - margin)
        else:
            x = random.randint(mid_x + margin, frame_width - margin)
        
        y = random.randint(margin, frame_height - margin)
        return (x, y)
    
    def generate_targets(self, frame_width, frame_height):
        """Generate new targets for active players"""
        for player_id, player in self.players.items():
            if player.is_active and player.current_target is None:
                position = self.generate_target_position(player_id, frame_width, frame_height)
                body_part = random.choice(GameConfig.BODY_PARTS)
                player.current_target = {
                    'position': position,
                    'body_part': body_part,
                    'spawn_time': time.time()
                }
    
    def check_collision(self, player_id):
        """Check if player touched their target"""
        player = self.players[player_id]
        
        if not player.current_target or not player.is_active:
            return False
        
        target_body_part = player.current_target['body_part']
        if target_body_part not in player.body_parts_positions:
            return False
        
        body_part_pos = player.body_parts_positions[target_body_part]
        target_pos = player.current_target['position']
        
        if body_part_pos is None or target_pos is None:
            return False
        
        distance = np.sqrt((body_part_pos[0] - target_pos[0])**2 + 
                          (body_part_pos[1] - target_pos[1])**2)
        
        return distance < GameConfig.TOUCH_THRESHOLD
    
    def update_game_logic(self):
        """Update game logic for both players"""
        for player_id, player in self.players.items():
            if self.check_collision(player_id):
                # Calculate time taken
                time_taken = time.time() - player.current_target['spawn_time']
                
                # Base points
                points = GameConfig.POINTS_PER_TARGET
                
                # Speed bonus
                if time_taken < GameConfig.SPEED_BONUS_THRESHOLD:
                    points += GameConfig.SPEED_BONUS_POINTS
                
                # Combo multiplier
                current_time = time.time()
                if current_time - player.last_touch_time < 3.0:
                    player.combo_multiplier = min(player.combo_multiplier + 0.2, 2.0)
                else:
                    player.combo_multiplier = 1.0
                
                points = int(points * player.combo_multiplier)
                player.add_score(points)
                player.last_touch_time = current_time
                
                # Reset target
                player.current_target = None
                
                # Check win condition
                if player.score >= GameConfig.WIN_SCORE:
                    self.winner = player_id
                    self.game_active = False
    
    def draw_split_screen(self, frame):
        """Draw split screen divider and labels"""
        h, w = frame.shape[:2]
        mid_x = w // 2
        
        # Draw vertical divider
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 3)
        
        # Draw player zones with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (mid_x, 60), self.players[1].color, -1)
        cv2.rectangle(overlay, (mid_x, 0), (w, 60), self.players[2].color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw player labels
        cv2.putText(frame, self.players[1].name, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, self.players[2].name, (mid_x + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    def draw_player_tracking(self, frame, player_id, landmarks, x_offset=0):
        """Draw player skeleton and tracking points"""
        if landmarks is None:
            return
        
        player = self.players[player_id]
        h, w = frame.shape[:2]
        
        # Adjust width for split screen
        if player_id == 1:
            w = w // 2
        else:
            w = w // 2
        
        # Update and draw body part positions
        for body_part, landmark_id in LANDMARK_MAP.items():
            landmark = landmarks.landmark[landmark_id]
            
            if landmark.visibility > 0.5:
                raw_x = int(landmark.x * w) + x_offset
                raw_y = int(landmark.y * h)
                
                # Update with smoothing
                smoothed_pos = player.update_position(body_part, (raw_x, raw_y))
                
                if smoothed_pos:
                    player.body_parts_positions[body_part] = smoothed_pos
                    
                    # Draw tracker for target body part
                    if (player.current_target and 
                        player.current_target['body_part'] == body_part):
                        # Enhanced tracker
                        cv2.circle(frame, smoothed_pos, GameConfig.TRACKER_RADIUS + 5, 
                                 player.color, -1)
                        cv2.circle(frame, smoothed_pos, GameConfig.TRACKER_RADIUS + 7, 
                                 (255, 255, 255), 2)
                    else:
                        # Normal joint
                        cv2.circle(frame, smoothed_pos, 5, player.color, -1)
        
        # Draw skeleton connections
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_landmark = landmarks.landmark[connection[0]]
            end_landmark = landmarks.landmark[connection[1]]
            
            if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                start_x = int(start_landmark.x * w) + x_offset
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w) + x_offset
                end_y = int(end_landmark.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                        player.color, 2, cv2.LINE_AA)
    
    def draw_targets(self, frame):
        """Draw targets for both players"""
        for player_id, player in self.players.items():
            if player.current_target:
                pos = player.current_target['position']
                body_part = player.current_target['body_part']
                
                # Animated pulse effect
                time_since_spawn = time.time() - player.current_target['spawn_time']
                pulse = abs(np.sin(time_since_spawn * 4)) * 0.3 + 0.7
                radius = int(GameConfig.TARGET_RADIUS * pulse)
                
                # Draw target
                cv2.circle(frame, pos, radius, player.color, 3)
                cv2.circle(frame, pos, 5, player.color, -1)
                
                # Draw crosshair
                cv2.line(frame, (pos[0] - 15, pos[1]), (pos[0] + 15, pos[1]), 
                        player.color, 2)
                cv2.line(frame, (pos[0], pos[1] - 15), (pos[0], pos[1] + 15), 
                        player.color, 2)
                
                # Draw label
                label = f"{body_part}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(frame, label, (pos[0] - label_size[0]//2, pos[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, player.color, 2)
    
    def draw_hud(self, frame):
        """Draw game HUD with scores and info"""
        h, w = frame.shape[:2]
        mid_x = w // 2
        
        # Player 1 score
        score_text = f"Score: {self.players[1].score}"
        cv2.putText(frame, score_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if self.players[1].combo_multiplier > 1:
            combo_text = f"Combo x{self.players[1].combo_multiplier:.1f}"
            cv2.putText(frame, combo_text, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Player 2 score
        score_text = f"Score: {self.players[2].score}"
        cv2.putText(frame, score_text, (mid_x + 20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if self.players[2].combo_multiplier > 1:
            combo_text = f"Combo x{self.players[2].combo_multiplier:.1f}"
            cv2.putText(frame, combo_text, (mid_x + 20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Win progress bar
        bar_width = 200
        bar_height = 20
        bar_y = h - 60
        
        # Player 1 progress
        p1_progress = min(self.players[1].score / GameConfig.WIN_SCORE, 1.0)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (20, bar_y), 
                     (20 + int(bar_width * p1_progress), bar_y + bar_height),
                     self.players[1].color, -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height),
                     (255, 255, 255), 2)
        
        # Player 2 progress
        p2_progress = min(self.players[2].score / GameConfig.WIN_SCORE, 1.0)
        cv2.rectangle(frame, (w - 20 - bar_width, bar_y), (w - 20, bar_y + bar_height),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (w - 20 - bar_width, bar_y), 
                     (w - 20 - bar_width + int(bar_width * p2_progress), bar_y + bar_height),
                     self.players[2].color, -1)
        cv2.rectangle(frame, (w - 20 - bar_width, bar_y), (w - 20, bar_y + bar_height),
                     (255, 255, 255), 2)
        
        # Goal indicator
        goal_text = f"First to {GameConfig.WIN_SCORE} wins!"
        text_size = cv2.getTextSize(goal_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, goal_text, (mid_x - text_size[0]//2, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_winner_screen(self, frame):
        """Draw winner announcement"""
        h, w = frame.shape[:2]
        
        # Darken background
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Winner text
        winner = self.players[self.winner]
        winner_text = f"{winner.name} Wins!"
        text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 50
        
        # Draw text with outline
        cv2.putText(frame, winner_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
        cv2.putText(frame, winner_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, winner.color, 4)
        
        # Final scores
        score_text = f"Final Scores - {self.players[1].name}: {self.players[1].score}  |  {self.players[2].name}: {self.players[2].score}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (text_x, text_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Restart instruction
        restart_text = "Press SPACE to play again or Q to quit"
        text_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, restart_text, (text_x, text_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    def reset_game(self):
        """Reset game state"""
        for player in self.players.values():
            player.score = 0
            player.current_target = None
            player.combo_multiplier = 1.0
            player.targets_hit = 0
            player.last_touch_time = 0
        
        self.winner = None
        self.game_active = True
        self.start_time = time.time()
    
    def run(self):
        """Main game loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow(GameConfig.WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        print(f"\n{GameConfig.WINDOW_NAME}")
        print(f"Resolution: {width}x{height}")
        print("\nInstructions:")
        print("- Stand on opposite sides of the camera")
        print("- Touch targets with the indicated body part")
        print("- First to 100 points wins!")
        print("\nControls:")
        print("  SPACE - Start/Restart game")
        print("  Q - Quit")
        print("\n")
        
        self.game_active = False
        waiting_for_start = True
        fps_counter = deque(maxlen=30)
        
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect players
            results_left, results_right, mid_x = self.detect_players(frame)
            
            # Draw split screen
            self.draw_split_screen(frame)
            
            if not self.game_active and not waiting_for_start and self.winner:
                # Show winner screen
                self.draw_winner_screen(frame)
            elif self.game_active:
                # Draw player tracking
                if results_left.pose_landmarks:
                    self.draw_player_tracking(frame, 1, results_left.pose_landmarks, 0)
                
                if results_right.pose_landmarks:
                    self.draw_player_tracking(frame, 2, results_right.pose_landmarks, mid_x)
                
                # Generate targets
                self.generate_targets(width, height)
                
                # Update game logic
                self.update_game_logic()
                
                # Draw targets
                self.draw_targets(frame)
                
                # Draw HUD
                self.draw_hud(frame)
            else:
                # Waiting to start
                if results_left.pose_landmarks:
                    self.draw_player_tracking(frame, 1, results_left.pose_landmarks, 0)
                
                if results_right.pose_landmarks:
                    self.draw_player_tracking(frame, 2, results_right.pose_landmarks, mid_x)
                
                # Start instruction
                start_text = "Press SPACE to start the game!"
                text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (width - text_size[0]) // 2
                text_y = height // 2
                
                cv2.putText(frame, start_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Player status
                p1_status = "Ready" if self.players[1].is_active else "Not detected"
                p2_status = "Ready" if self.players[2].is_active else "Not detected"
                
                cv2.putText(frame, f"Player 1: {p1_status}", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.players[1].color, 2)
                cv2.putText(frame, f"Player 2: {p2_status}", (mid_x + 20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.players[2].color, 2)
            
            # FPS counter
            fps = 1.0 / (time.time() - frame_start + 1e-6)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (width // 2 - 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(GameConfig.WINDOW_NAME, frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.game_active or self.winner:
                    self.reset_game()
                    waiting_for_start = False
                    print("Game started!")
            elif key == ord('r'):
                self.reset_game()
                waiting_for_start = False
                print("Game restarted!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        for player in self.players.values():
            player.pose_tracker.close()
        self.segmentation.close()
        
        print("\nGame ended!")
        if self.winner:
            print(f"Winner: {self.players[self.winner].name}")
            print(f"Final scores:")
            print(f"  {self.players[1].name}: {self.players[1].score}")
            print(f"  {self.players[2].name}: {self.players[2].score}")

def main():
    game = TwoPlayerGame()
    game.run()

if __name__ == "__main__":
    main()
