#!/usr/bin/env python3
"""
Advanced Multiplayer Body Tracking Game
Supports multiple game modes: Cooperative, Competitive, Race, and Mirror Match
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
from enum import Enum

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Game Configuration
@dataclass
class GameSettings:
    """Game configuration settings"""
    window_name: str = "Multiplayer Body Tracking Arena"
    target_fps: int = 30
    win_score: int = 100
    touch_threshold: int = 45
    target_radius: int = 30
    tracker_radius: int = 18
    
    # Colors (BGR)
    colors = {
        'player1': (255, 100, 100),  # Blue
        'player2': (100, 255, 100),  # Green
        'player3': (100, 100, 255),  # Red
        'player4': (255, 255, 100),  # Cyan
        'target': (0, 200, 255),      # Orange
        'bonus': (255, 0, 255),       # Magenta
    }

class GameMode(Enum):
    """Game modes available"""
    MENU = "menu"
    COOPERATIVE = "coop"      # Players work together
    COMPETITIVE = "versus"    # Players compete
    RACE = "race"            # Race to score
    MIRROR = "mirror"        # Mirror movements
    SURVIVAL = "survival"    # Survival mode

class BodyPart(Enum):
    """Trackable body parts"""
    RIGHT_HAND = "Right Hand"
    LEFT_HAND = "Left Hand"
    HEAD = "Head"
    RIGHT_FOOT = "Right Foot"
    LEFT_FOOT = "Left Foot"
    RIGHT_ELBOW = "Right Elbow"
    LEFT_ELBOW = "Left Elbow"
    RIGHT_KNEE = "Right Knee"
    LEFT_KNEE = "Left Knee"

# Landmark mapping
LANDMARK_MAP = {
    BodyPart.RIGHT_HAND: mp_pose.PoseLandmark.RIGHT_WRIST,
    BodyPart.LEFT_HAND: mp_pose.PoseLandmark.LEFT_WRIST,
    BodyPart.HEAD: mp_pose.PoseLandmark.NOSE,
    BodyPart.RIGHT_FOOT: mp_pose.PoseLandmark.RIGHT_ANKLE,
    BodyPart.LEFT_FOOT: mp_pose.PoseLandmark.LEFT_ANKLE,
    BodyPart.RIGHT_ELBOW: mp_pose.PoseLandmark.RIGHT_ELBOW,
    BodyPart.LEFT_ELBOW: mp_pose.PoseLandmark.LEFT_ELBOW,
    BodyPart.RIGHT_KNEE: mp_pose.PoseLandmark.RIGHT_KNEE,
    BodyPart.LEFT_KNEE: mp_pose.PoseLandmark.LEFT_KNEE,
}

class Target:
    """Represents a game target"""
    def __init__(self, position: Tuple[int, int], body_part: BodyPart, 
                 owner_id: Optional[int] = None, is_bonus: bool = False):
        self.position = position
        self.body_part = body_part
        self.owner_id = owner_id  # Which player owns this target
        self.is_bonus = is_bonus
        self.spawn_time = time.time()
        self.is_hit = False
        self.hit_by = None
        self.points = 20 if is_bonus else 10
        
    def check_hit(self, position: Tuple[int, int], threshold: int) -> bool:
        """Check if target is hit"""
        if position is None or self.is_hit:
            return False
        
        distance = np.sqrt((position[0] - self.position[0])**2 + 
                          (position[1] - self.position[1])**2)
        return distance < threshold

class Player:
    """Enhanced player class with advanced tracking"""
    def __init__(self, player_id: int, name: str, color: Tuple[int, int, int], 
                 zone: Optional[Tuple[int, int, int, int]] = None):
        self.id = player_id
        self.name = name
        self.color = color
        self.zone = zone  # (x1, y1, x2, y2) playing zone
        
        # Tracking
        self.pose_tracker = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=True
        )
        
        # Game state
        self.score = 0
        self.combo = 0
        self.combo_multiplier = 1.0
        self.last_hit_time = 0
        self.targets_hit = 0
        self.accuracy = 0.0
        self.is_active = False
        
        # Position tracking
        self.body_positions = {}
        self.position_history = {part: deque(maxlen=7) for part in BodyPart}
        self.landmarks = None
        
        # Performance metrics
        self.reaction_times = deque(maxlen=10)
        self.average_reaction = 0.0
    
    def update_position(self, body_part: BodyPart, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Update position with advanced smoothing"""
        if position is None:
            return self.body_positions.get(body_part)
        
        # Add to history
        self.position_history[body_part].append(position)
        
        # Weighted moving average
        if len(self.position_history[body_part]) >= 3:
            positions = list(self.position_history[body_part])
            weights = np.linspace(0.3, 1.0, len(positions))
            weights = weights / weights.sum()
            
            avg_x = sum(p[0] * w for p, w in zip(positions, weights))
            avg_y = sum(p[1] * w for p, w in zip(positions, weights))
            
            smoothed = (int(avg_x), int(avg_y))
            self.body_positions[body_part] = smoothed
            return smoothed
        
        self.body_positions[body_part] = position
        return position
    
    def add_score(self, points: int, reaction_time: float = None):
        """Add score with combo calculation"""
        current_time = time.time()
        
        # Update combo
        if current_time - self.last_hit_time < 3.0:
            self.combo += 1
            self.combo_multiplier = min(1.0 + self.combo * 0.2, 3.0)
        else:
            self.combo = 0
            self.combo_multiplier = 1.0
        
        # Apply multiplier
        final_points = int(points * self.combo_multiplier)
        self.score += final_points
        self.targets_hit += 1
        self.last_hit_time = current_time
        
        # Track reaction time
        if reaction_time:
            self.reaction_times.append(reaction_time)
            self.average_reaction = np.mean(self.reaction_times)
        
        return final_points
    
    def reset(self):
        """Reset player state"""
        self.score = 0
        self.combo = 0
        self.combo_multiplier = 1.0
        self.targets_hit = 0
        self.accuracy = 0.0
        self.reaction_times.clear()

class MultiplayerGame:
    """Advanced multiplayer game controller"""
    def __init__(self, num_players: int = 2):
        self.settings = GameSettings()
        self.num_players = min(num_players, 4)
        self.game_mode = GameMode.MENU
        self.game_active = False
        
        # Initialize players
        self.players = self._initialize_players()
        
        # Game state
        self.targets: List[Target] = []
        self.bonus_targets: List[Target] = []
        self.game_start_time = 0
        self.game_duration = 0
        self.winner = None
        self.team_score = 0
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
    
    def _initialize_players(self) -> Dict[int, Player]:
        """Initialize players based on number"""
        players = {}
        
        if self.num_players == 2:
            players[1] = Player(1, "Blue Player", (255, 100, 100))
            players[2] = Player(2, "Green Player", (100, 255, 100))
        elif self.num_players == 3:
            players[1] = Player(1, "Blue Player", (255, 100, 100))
            players[2] = Player(2, "Green Player", (100, 255, 100))
            players[3] = Player(3, "Red Player", (100, 100, 255))
        elif self.num_players == 4:
            players[1] = Player(1, "Blue", (255, 100, 100))
            players[2] = Player(2, "Green", (100, 255, 100))
            players[3] = Player(3, "Red", (100, 100, 255))
            players[4] = Player(4, "Cyan", (255, 255, 100))
        
        return players
    
    def detect_players(self, frame: np.ndarray) -> Dict[int, any]:
        """Detect all players in frame"""
        h, w = frame.shape[:2]
        results = {}
        
        if self.num_players == 2:
            # Split screen vertically
            mid_x = w // 2
            
            # Player 1 (left side)
            left_frame = frame[:, :mid_x]
            rgb_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            results[1] = self.players[1].pose_tracker.process(rgb_left)
            self.players[1].zone = (0, 0, mid_x, h)
            
            # Player 2 (right side)
            right_frame = frame[:, mid_x:]
            rgb_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            results[2] = self.players[2].pose_tracker.process(rgb_right)
            self.players[2].zone = (mid_x, 0, w, h)
            
        elif self.num_players == 3:
            # One player top, two bottom
            mid_y = h // 2
            mid_x = w // 2
            
            # Player 1 (top)
            top_frame = frame[:mid_y, :]
            results[1] = self.players[1].pose_tracker.process(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))
            self.players[1].zone = (0, 0, w, mid_y)
            
            # Player 2 (bottom left)
            bl_frame = frame[mid_y:, :mid_x]
            results[2] = self.players[2].pose_tracker.process(cv2.cvtColor(bl_frame, cv2.COLOR_BGR2RGB))
            self.players[2].zone = (0, mid_y, mid_x, h)
            
            # Player 3 (bottom right)
            br_frame = frame[mid_y:, mid_x:]
            results[3] = self.players[3].pose_tracker.process(cv2.cvtColor(br_frame, cv2.COLOR_BGR2RGB))
            self.players[3].zone = (mid_x, mid_y, w, h)
            
        elif self.num_players == 4:
            # Four quadrants
            mid_x = w // 2
            mid_y = h // 2
            
            # Top left
            tl_frame = frame[:mid_y, :mid_x]
            results[1] = self.players[1].pose_tracker.process(cv2.cvtColor(tl_frame, cv2.COLOR_BGR2RGB))
            self.players[1].zone = (0, 0, mid_x, mid_y)
            
            # Top right
            tr_frame = frame[:mid_y, mid_x:]
            results[2] = self.players[2].pose_tracker.process(cv2.cvtColor(tr_frame, cv2.COLOR_BGR2RGB))
            self.players[2].zone = (mid_x, 0, w, mid_y)
            
            # Bottom left
            bl_frame = frame[mid_y:, :mid_x]
            results[3] = self.players[3].pose_tracker.process(cv2.cvtColor(bl_frame, cv2.COLOR_BGR2RGB))
            self.players[3].zone = (0, mid_y, mid_x, h)
            
            # Bottom right
            br_frame = frame[mid_y:, mid_x:]
            results[4] = self.players[4].pose_tracker.process(cv2.cvtColor(br_frame, cv2.COLOR_BGR2RGB))
            self.players[4].zone = (mid_x, mid_y, w, h)
        
        # Update active status
        for player_id, result in results.items():
            self.players[player_id].is_active = result.pose_landmarks is not None
            self.players[player_id].landmarks = result.pose_landmarks
        
        return results
    
    def generate_targets(self, frame_width: int, frame_height: int):
        """Generate targets based on game mode"""
        if self.game_mode == GameMode.COOPERATIVE:
            # Generate shared targets
            if len(self.targets) == 0:
                num_targets = min(self.num_players, 3)
                for _ in range(num_targets):
                    pos = self._get_random_position(frame_width, frame_height)
                    body_part = random.choice(list(BodyPart))
                    self.targets.append(Target(pos, body_part))
            
        elif self.game_mode == GameMode.COMPETITIVE:
            # Individual targets for each player
            for player_id, player in self.players.items():
                if player.is_active and not any(t.owner_id == player_id for t in self.targets):
                    pos = self._get_random_position_in_zone(player.zone)
                    body_part = random.choice(list(BodyPart))
                    self.targets.append(Target(pos, body_part, owner_id=player_id))
        
        elif self.game_mode == GameMode.RACE:
            # Same target for all players
            if len(self.targets) == 0:
                pos = self._get_center_position(frame_width, frame_height)
                body_part = random.choice(list(BodyPart))
                self.targets.append(Target(pos, body_part))
        
        # Add bonus targets occasionally
        if random.random() < 0.02 and len(self.bonus_targets) < 2:
            pos = self._get_random_position(frame_width, frame_height)
            body_part = random.choice(list(BodyPart))
            self.bonus_targets.append(Target(pos, body_part, is_bonus=True))
    
    def _get_random_position(self, width: int, height: int) -> Tuple[int, int]:
        """Get random position in frame"""
        margin = 60
        x = random.randint(margin, width - margin)
        y = random.randint(margin, height - margin)
        return (x, y)
    
    def _get_random_position_in_zone(self, zone: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get random position within zone"""
        if zone is None:
            return (100, 100)
        
        x1, y1, x2, y2 = zone
        margin = 40
        x = random.randint(x1 + margin, x2 - margin)
        y = random.randint(y1 + margin, y2 - margin)
        return (x, y)
    
    def _get_center_position(self, width: int, height: int) -> Tuple[int, int]:
        """Get center position"""
        return (width // 2, height // 2)
    
    def update_game_logic(self):
        """Update game based on mode"""
        # Check target hits
        for target in self.targets[:]:
            for player_id, player in self.players.items():
                if not player.is_active:
                    continue
                
                # Check if player can hit this target
                if target.owner_id and target.owner_id != player_id:
                    continue
                
                # Get body part position
                if target.body_part in player.body_positions:
                    pos = player.body_positions[target.body_part]
                    
                    if target.check_hit(pos, self.settings.touch_threshold):
                        # Calculate reaction time
                        reaction_time = time.time() - target.spawn_time
                        
                        # Award points
                        points = player.add_score(target.points, reaction_time)
                        
                        # Handle based on mode
                        if self.game_mode == GameMode.COOPERATIVE:
                            self.team_score += points
                        
                        # Mark target as hit
                        target.is_hit = True
                        target.hit_by = player_id
                        
                        # Remove target
                        self.targets.remove(target)
                        break
        
        # Check bonus targets
        for bonus in self.bonus_targets[:]:
            for player_id, player in self.players.items():
                if not player.is_active:
                    continue
                
                if bonus.body_part in player.body_positions:
                    pos = player.body_positions[bonus.body_part]
                    
                    if bonus.check_hit(pos, self.settings.touch_threshold):
                        points = player.add_score(bonus.points)
                        self.bonus_targets.remove(bonus)
                        break
        
        # Check win conditions
        if self.game_mode == GameMode.COMPETITIVE:
            for player in self.players.values():
                if player.score >= self.settings.win_score:
                    self.winner = player.id
                    self.game_active = False
        elif self.game_mode == GameMode.COOPERATIVE:
            if self.team_score >= self.settings.win_score * self.num_players:
                self.winner = "team"
                self.game_active = False
    
    def draw_game_view(self, frame: np.ndarray, results: Dict[int, any]):
        """Draw game visualization"""
        h, w = frame.shape[:2]
        
        # Draw zone dividers
        if self.num_players == 2:
            cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        elif self.num_players == 3:
            cv2.line(frame, (0, h//2), (w, h//2), (255, 255, 255), 2)
            cv2.line(frame, (w//2, h//2), (w//2, h), (255, 255, 255), 2)
        elif self.num_players == 4:
            cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
            cv2.line(frame, (0, h//2), (w, h//2), (255, 255, 255), 2)
        
        # Draw players
        for player_id, result in results.items():
            if result.pose_landmarks:
                self._draw_player(frame, self.players[player_id], result.pose_landmarks)
        
        # Draw targets
        for target in self.targets:
            self._draw_target(frame, target)
        
        for bonus in self.bonus_targets:
            self._draw_bonus_target(frame, bonus)
        
        # Draw HUD
        self._draw_hud(frame)
    
    def _draw_player(self, frame: np.ndarray, player: Player, landmarks):
        """Draw player skeleton and trackers"""
        h, w = frame.shape[:2]
        zone = player.zone
        
        if zone:
            zone_w = zone[2] - zone[0]
            zone_h = zone[3] - zone[1]
            
            # Update and draw body positions
            for body_part, landmark_id in LANDMARK_MAP.items():
                landmark = landmarks.landmark[landmark_id]
                
                if landmark.visibility > 0.5:
                    # Calculate position in zone
                    x = int(landmark.x * zone_w) + zone[0]
                    y = int(landmark.y * zone_h) + zone[1]
                    
                    # Update with smoothing
                    smoothed_pos = player.update_position(body_part, (x, y))
                    
                    if smoothed_pos:
                        # Draw tracker
                        radius = self.settings.tracker_radius
                        
                        # Highlight if this is the target body part
                        is_target = any(t.body_part == body_part for t in self.targets)
                        
                        if is_target:
                            # Glow effect
                            for i in range(3):
                                alpha = 0.3 - i * 0.1
                                overlay = frame.copy()
                                cv2.circle(overlay, smoothed_pos, radius + i*5, player.color, -1)
                                cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0, frame)
                            
                            cv2.circle(frame, smoothed_pos, radius, player.color, -1)
                            cv2.circle(frame, smoothed_pos, radius + 2, (255, 255, 255), 2)
                        else:
                            cv2.circle(frame, smoothed_pos, 5, player.color, -1)
            
            # Draw skeleton
            for connection in mp_pose.POSE_CONNECTIONS:
                start_landmark = landmarks.landmark[connection[0]]
                end_landmark = landmarks.landmark[connection[1]]
                
                if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                    start_x = int(start_landmark.x * zone_w) + zone[0]
                    start_y = int(start_landmark.y * zone_h) + zone[1]
                    end_x = int(end_landmark.x * zone_w) + zone[0]
                    end_y = int(end_landmark.y * zone_h) + zone[1]
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                            player.color, 2, cv2.LINE_AA)
    
    def _draw_target(self, frame: np.ndarray, target: Target):
        """Draw target with effects"""
        # Pulse effect
        time_since_spawn = time.time() - target.spawn_time
        pulse = abs(np.sin(time_since_spawn * 4)) * 0.3 + 0.7
        radius = int(self.settings.target_radius * pulse)
        
        # Color based on owner
        color = self.settings.colors['target']
        if target.owner_id and target.owner_id in self.players:
            color = self.players[target.owner_id].color
        
        # Draw target
        cv2.circle(frame, target.position, radius, color, 3)
        cv2.circle(frame, target.position, 5, color, -1)
        
        # Draw label
        label = target.body_part.value
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_pos = (target.position[0] - text_size[0]//2, target.position[1] - radius - 10)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_bonus_target(self, frame: np.ndarray, target: Target):
        """Draw bonus target with special effects"""
        # Rainbow effect
        time_since_spawn = time.time() - target.spawn_time
        hue = int((time_since_spawn * 60) % 180)
        
        # Create HSV color and convert to BGR
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(int(c) for c in bgr)
        
        # Animated radius
        pulse = abs(np.sin(time_since_spawn * 6)) * 0.4 + 0.6
        radius = int(self.settings.target_radius * 1.5 * pulse)
        
        # Draw with star shape
        angles = np.linspace(0, 2*np.pi, 8)
        outer_points = [(int(target.position[0] + radius * np.cos(a)),
                        int(target.position[1] + radius * np.sin(a))) 
                       for a in angles[::2]]
        inner_points = [(int(target.position[0] + radius*0.5 * np.cos(a)),
                        int(target.position[1] + radius*0.5 * np.sin(a))) 
                       for a in angles[1::2]]
        
        points = []
        for i in range(4):
            points.append(outer_points[i])
            points.append(inner_points[i])
        
        cv2.fillPoly(frame, [np.array(points)], color)
        cv2.polylines(frame, [np.array(points)], True, (255, 255, 255), 2)
        
        # Label
        cv2.putText(frame, "BONUS", 
                   (target.position[0] - 25, target.position[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _draw_hud(self, frame: np.ndarray):
        """Draw game HUD"""
        h, w = frame.shape[:2]
        
        # Draw player scores
        y_offset = 30
        for player_id, player in self.players.items():
            if not player.is_active:
                continue
            
            # Score text
            score_text = f"{player.name}: {player.score}"
            if player.combo > 0:
                score_text += f" (x{player.combo_multiplier:.1f})"
            
            # Position based on player zone
            if player.zone:
                x = player.zone[0] + 10
                y = player.zone[1] + y_offset
                cv2.putText(frame, score_text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, player.color, 2)
        
        # Game mode indicator
        mode_text = f"Mode: {self.game_mode.value.upper()}"
        cv2.putText(frame, mode_text, (w//2 - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Team score for cooperative mode
        if self.game_mode == GameMode.COOPERATIVE:
            team_text = f"Team Score: {self.team_score}/{self.settings.win_score * self.num_players}"
            cv2.putText(frame, team_text, (w//2 - 100, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # FPS
        if len(self.fps_counter) > 0:
            avg_fps = np.mean(self.fps_counter)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
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
        
        cv2.namedWindow(self.settings.window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n{self.settings.window_name}")
        print(f"Players: {self.num_players}")
        print(f"Resolution: {width}x{height}")
        print("\nControls:")
        print("  1-4: Select number of players")
        print("  C: Cooperative mode")
        print("  V: Versus mode")
        print("  R: Race mode")
        print("  SPACE: Start/Restart")
        print("  Q: Quit\n")
        
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect all players
            results = self.detect_players(frame)
            
            # Update game
            if self.game_active:
                self.generate_targets(width, height)
                self.update_game_logic()
            
            # Draw game
            self.draw_game_view(frame, results)
            
            # Show winner screen
            if self.winner:
                self._draw_winner_screen(frame)
            
            cv2.imshow(self.settings.window_name, frame)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - frame_start + 1e-6)
            self.fps_counter.append(fps)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.start_game()
            elif key == ord('c'):
                self.game_mode = GameMode.COOPERATIVE
                print("Mode: Cooperative")
            elif key == ord('v'):
                self.game_mode = GameMode.COMPETITIVE
                print("Mode: Competitive")
            elif key == ord('r'):
                self.game_mode = GameMode.RACE
                print("Mode: Race")
            elif key >= ord('1') and key <= ord('4'):
                self.num_players = key - ord('0')
                self.players = self._initialize_players()
                print(f"Players set to: {self.num_players}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        for player in self.players.values():
            player.pose_tracker.close()
    
    def _draw_winner_screen(self, frame: np.ndarray):
        """Draw winner announcement"""
        h, w = frame.shape[:2]
        
        # Darken background
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        if self.winner == "team":
            text = "Team Victory!"
            color = (0, 255, 0)
        else:
            winner_player = self.players[self.winner]
            text = f"{winner_player.name} Wins!"
            color = winner_player.color
        
        # Draw text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Instructions
        cv2.putText(frame, "Press SPACE to play again", (text_x, text_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def start_game(self):
        """Start or restart game"""
        self.game_active = True
        self.winner = None
        self.targets.clear()
        self.bonus_targets.clear()
        self.team_score = 0
        self.game_start_time = time.time()
        
        for player in self.players.values():
            player.reset()
        
        print(f"Game started - Mode: {self.game_mode.value}")

def main():
    """Main entry point"""
    import sys
    
    # Get number of players from command line
    num_players = 2
    if len(sys.argv) > 1:
        try:
            num_players = int(sys.argv[1])
            num_players = min(max(num_players, 2), 4)
        except:
            pass
    
    game = MultiplayerGame(num_players)
    game.run()

if __name__ == "__main__":
    main()
