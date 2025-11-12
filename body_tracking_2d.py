import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import platform

# Initialize MediaPipe Holistic for better tracking consistency
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Holistic model with optimized settings
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    refine_face_landmarks=False
)

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head', 'Right Shoulder', 'Left Shoulder']
TOUCH_THRESHOLD = 40  # Pixels for 2D distance
TARGET_FPS = 30
BUTTON_WIDTH = 400
BUTTON_HEIGHT = 80
BUTTON_SPACING = 30
TARGET_RADIUS = 25
BODY_PART_RADIUS = 18

# Colors (BGR format for OpenCV)
COLOR_BODY_FILL = (0, 255, 100)  # Green
COLOR_BODY_OUTLINE = (0, 200, 0)  # Darker green
COLOR_TARGET = (0, 0, 255)  # Red
COLOR_TRACKER = (255, 0, 255)  # Magenta
COLOR_HUD_BG = (40, 40, 40)  # Dark gray

# Landmark mapping for holistic model
LANDMARK_MAP = {
    'Right Hand': mp_holistic.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_holistic.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_holistic.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_holistic.PoseLandmark.LEFT_ELBOW,
    'Head': mp_holistic.PoseLandmark.NOSE,
    'Right Shoulder': mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    'Left Shoulder': mp_holistic.PoseLandmark.LEFT_SHOULDER
}

# Smoothing parameters
POSITION_HISTORY_SIZE = 3  # Number of frames to average for smoothing
position_history = {part: deque(maxlen=POSITION_HISTORY_SIZE) for part in BODY_PARTS}

# Game state
score = 0
high_score = 0
show_silhouette = True
debug_mode = False
show_skeleton = False

# Font loading with fallback
def load_fonts():
    """Load fonts with system-specific fallbacks"""
    fonts = {}
    
    # Try different font paths based on OS
    font_paths = []
    if platform.system() == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir.ttc"
        ]
    elif platform.system() == "Windows":
        font_paths = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf"
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
    
    # Try to load fonts
    font_loaded = False
    for font_path in font_paths:
        try:
            fonts['title'] = ImageFont.truetype(font_path, 60)
            fonts['large'] = ImageFont.truetype(font_path, 40)
            fonts['medium'] = ImageFont.truetype(font_path, 28)
            fonts['small'] = ImageFont.truetype(font_path, 20)
            font_loaded = True
            break
        except:
            continue
    
    # Fallback to default font
    if not font_loaded:
        print("Warning: Could not load TrueType fonts, using default")
        default_font = ImageFont.load_default()
        fonts['title'] = default_font
        fonts['large'] = default_font
        fonts['medium'] = default_font
        fonts['small'] = default_font
    
    return fonts

FONTS = load_fonts()

def smooth_position(body_part, new_pos):
    """Apply position smoothing to reduce jitter"""
    if new_pos is not None:
        position_history[body_part].append(new_pos)
    
    if len(position_history[body_part]) == 0:
        return None
    
    # Calculate average position
    positions = list(position_history[body_part])
    avg_x = sum(p[0] for p in positions) / len(positions)
    avg_y = sum(p[1] for p in positions) / len(positions)
    
    return (int(avg_x), int(avg_y))

def calculate_2d_distance(point1, point2):
    """Calculate 2D Euclidean distance"""
    if point1 is None or point2 is None:
        return float('inf')
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return np.sqrt(dx*dx + dy*dy)

def generate_target(frame_width, frame_height):
    """Generate a random target position and body part"""
    margin = 80
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def create_body_mask(segmentation_mask, frame_shape):
    """Create a smooth body mask from segmentation"""
    if segmentation_mask is None:
        return None
    
    # Convert to binary mask
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
    
    # Apply morphological operations for smoother edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur for smoother edges
    binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 2)
    binary_mask = (binary_mask > 128).astype(np.uint8) * 255
    
    return binary_mask

def apply_body_overlay(frame, segmentation_mask, color=COLOR_BODY_FILL, opacity=0.7):
    """Apply a semi-transparent colored overlay for the body"""
    if segmentation_mask is None:
        return frame
    
    # Create smooth body mask
    mask = create_body_mask(segmentation_mask, frame.shape[:2])
    if mask is None:
        return frame
    
    # Create colored overlay
    overlay = frame.copy()
    colored_body = np.zeros_like(frame)
    colored_body[:] = color
    
    # Apply mask to colored body
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    body_overlay = (colored_body * mask_3d).astype(np.uint8)
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - opacity, body_overlay, opacity, 0)
    
    # Draw body outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, COLOR_BODY_OUTLINE, 2)
    
    return result

def draw_text_with_background(image, text, position, font_size='medium', text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=10):
    """Draw text with background using OpenCV"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = {'small': 0.6, 'medium': 0.8, 'large': 1.0, 'title': 1.5}.get(font_size, 0.8)
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Draw background rectangle
    cv2.rectangle(image, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    
    return image

def draw_button(image, text, position, size, color, hover=False, enabled=True):
    """Draw a button with rounded corners"""
    x, y = position
    w, h = size
    
    # Adjust color for hover
    if hover and enabled:
        color = tuple(min(c + 40, 255) for c in color)
    
    # Draw rounded rectangle
    radius = 15
    cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), color, -1)
    
    # Draw corners
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(image, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(image, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(image, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, -1)
    
    # Draw border
    border_color = (255, 255, 255) if hover else (200, 200, 200)
    cv2.rectangle(image, (x + 2, y + 2), (x + w - 2, y + h - 2), border_color, 2)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (255, 255, 255) if enabled else (150, 150, 150)
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x + (w - text_width) // 2
    text_y = y + (h + text_height) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return (x, y, x + w, y + h)

def is_point_in_button(point, button_bounds):
    """Check if a point is inside button bounds"""
    x, y = point
    x1, y1, x2, y2 = button_bounds
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_hud(image, score, high_score, target_body_part, frame_width, frame_height):
    """Draw the game HUD"""
    # Score display
    draw_text_with_background(image, f"Score: {score}", (20, 40), 'large', (0, 255, 0))
    draw_text_with_background(image, f"High Score: {high_score}", (20, 90), 'medium', (255, 255, 0))
    
    # Target display
    target_text = f"Touch with: {target_body_part}"
    draw_text_with_background(image, target_text, (frame_width // 2 - 150, 40), 'large', (255, 255, 255))
    
    # Controls
    controls_text = "ESC: Menu | S: Silhouette | D: Debug | K: Skeleton | Q: Quit"
    cv2.putText(image, controls_text, (20, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Status indicators
    status_y = 140
    if show_silhouette:
        draw_text_with_background(image, "Silhouette: ON", (20, status_y), 'small', (0, 255, 200))
        status_y += 40
    if debug_mode:
        draw_text_with_background(image, "Debug: ON", (20, status_y), 'small', (255, 255, 0))
        status_y += 40
    if show_skeleton:
        draw_text_with_background(image, "Skeleton: ON", (20, status_y), 'small', (0, 255, 255))

def show_menu(cap, frame_width, frame_height):
    """Display the main menu"""
    global show_silhouette, debug_mode, show_skeleton
    
    cv2.namedWindow('Body Tracking 2D Game', cv2.WINDOW_NORMAL)
    
    start_y = 200
    center_x = (frame_width - BUTTON_WIDTH) // 2
    
    buttons = [
        {
            'text': 'Start Game',
            'position': (center_x, start_y),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (50, 150, 50),
            'action': 'play',
            'enabled': True
        },
        {
            'text': 'Settings',
            'position': (center_x, start_y + BUTTON_HEIGHT + BUTTON_SPACING),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (50, 100, 150),
            'action': 'settings',
            'enabled': True
        },
        {
            'text': 'Quit',
            'position': (center_x, start_y + 2 * (BUTTON_HEIGHT + BUTTON_SPACING)),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (150, 50, 50),
            'action': 'quit',
            'enabled': True
        }
    ]
    
    hover_index = None
    mouse_pos = (0, 0)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, hover_index
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, button in enumerate(buttons):
                bounds = draw_button(np.zeros((1, 1, 3)), button['text'], button['position'], 
                                   button['size'], button['color'])
                if is_point_in_button((x, y), bounds):
                    return i
        return None
    
    cv2.setMouseCallback('Body Tracking 2D Game', mouse_callback)
    
    print("Menu loaded!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # Apply body overlay if enabled
        if results.segmentation_mask is not None and show_silhouette:
            frame = apply_body_overlay(frame, results.segmentation_mask)
        
        # Draw title
        title_text = "Body Tracking 2D"
        cv2.putText(frame, title_text, (frame_width // 2 - 200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Check hover state
        hover_index = None
        for i, button in enumerate(buttons):
            bounds = draw_button(frame, button['text'], button['position'], 
                               button['size'], button['color'])
            if is_point_in_button(mouse_pos, bounds):
                hover_index = i
                draw_button(frame, button['text'], button['position'], 
                          button['size'], button['color'], hover=True, enabled=button['enabled'])
            else:
                draw_button(frame, button['text'], button['position'], 
                          button['size'], button['color'], enabled=button['enabled'])
        
        # Track hand for gesture control
        if results.pose_landmarks:
            right_hand = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            hand_x = int(right_hand.x * frame_width)
            hand_y = int(right_hand.y * frame_height)
            
            # Draw hand cursor
            cv2.circle(frame, (hand_x, hand_y), 15, COLOR_TRACKER, -1)
            cv2.circle(frame, (hand_x, hand_y), 17, (255, 255, 255), 2)
            
            # Check if hand is over a button
            for i, button in enumerate(buttons):
                bounds = (button['position'][0], button['position'][1],
                         button['position'][0] + button['size'][0],
                         button['position'][1] + button['size'][1])
                if is_point_in_button((hand_x, hand_y), bounds):
                    hover_index = i
                    draw_button(frame, button['text'], button['position'], 
                              button['size'], button['color'], hover=True, enabled=button['enabled'])
        
        cv2.imshow('Body Tracking 2D Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Handle keyboard input
        if key == ord('q') or key == 27:  # ESC
            return None
        elif key == ord(' ') or key == 13:  # Space or Enter
            if hover_index is not None:
                action = buttons[hover_index]['action']
                if action == 'play':
                    return 'play'
                elif action == 'settings':
                    return 'settings'
                elif action == 'quit':
                    return None
        elif key == ord('s'):
            show_silhouette = not show_silhouette
            print(f"Silhouette: {'ON' if show_silhouette else 'OFF'}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('k'):
            show_skeleton = not show_skeleton
            print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
        
        # Check mouse clicks
        for i, button in enumerate(buttons):
            bounds = (button['position'][0], button['position'][1],
                     button['position'][0] + button['size'][0],
                     button['position'][1] + button['size'][1])
            if is_point_in_button(mouse_pos, bounds):
                # This would be triggered by mouse callback
                pass
    
    return None

def play_game(cap, frame_width, frame_height):
    """Main game loop"""
    global score, high_score, show_silhouette, debug_mode, show_skeleton, position_history
    
    # Reset position history
    position_history = {part: deque(maxlen=POSITION_HISTORY_SIZE) for part in BODY_PARTS}
    
    score = 0
    target_pos, target_body_part = generate_target(frame_width, frame_height)
    target_spawn_time = time.time()
    combo_multiplier = 1
    last_touch_time = 0
    
    cv2.namedWindow('Body Tracking 2D Game', cv2.WINDOW_NORMAL)
    
    print("Game started! Touch the targets with the specified body parts.")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with holistic model
        results = holistic.process(rgb_frame)
        
        # Apply body overlay if enabled
        if results.segmentation_mask is not None and show_silhouette:
            frame = apply_body_overlay(frame, results.segmentation_mask, opacity=0.6)
        
        # Track body part position
        body_part_pos = None
        if results.pose_landmarks and target_body_part in LANDMARK_MAP:
            landmark = LANDMARK_MAP[target_body_part]
            body_part_landmark = results.pose_landmarks.landmark[landmark]
            
            # Get raw position
            raw_x = int(body_part_landmark.x * frame_width)
            raw_y = int(body_part_landmark.y * frame_height)
            
            # Apply smoothing
            body_part_pos = smooth_position(target_body_part, (raw_x, raw_y))
            
            if body_part_pos:
                # Draw body part tracker
                cv2.circle(frame, body_part_pos, BODY_PART_RADIUS, COLOR_TRACKER, -1)
                cv2.circle(frame, body_part_pos, BODY_PART_RADIUS + 2, (255, 255, 255), 2)
                
                # Calculate distance to target
                distance = calculate_2d_distance(body_part_pos, target_pos)
                
                # Check if target is touched
                if distance < TOUCH_THRESHOLD:
                    # Calculate combo bonus
                    current_time = time.time()
                    if current_time - last_touch_time < 2.0:  # Within 2 seconds
                        combo_multiplier = min(combo_multiplier + 0.5, 5.0)
                    else:
                        combo_multiplier = 1
                    
                    points = int(10 * combo_multiplier)
                    score += points
                    high_score = max(high_score, score)
                    last_touch_time = current_time

                    # Visual feedback
                    cv2.putText(frame, f"+{points}", target_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Generate new target
                    target_pos, target_body_part = generate_target(frame_width, frame_height)
                    target_spawn_time = time.time()
                    
                    
        
        # Draw target with pulsing effect
        time_since_spawn = time.time() - target_spawn_time
        pulse = abs(np.sin(time_since_spawn * 3)) * 0.3 + 0.7
        target_radius = int(TARGET_RADIUS * pulse)
        
        cv2.circle(frame, target_pos, target_radius, COLOR_TARGET, 3)
        cv2.circle(frame, target_pos, 5, COLOR_TARGET, -1)
        
        # Draw target body part label
        cv2.putText(frame, target_body_part, 
                   (target_pos[0] - 50, target_pos[1] - target_radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw skeleton if enabled
        if show_skeleton and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        
        # Draw HUD
        draw_hud(frame, score, high_score, target_body_part, frame_width, frame_height)
        
        # Draw combo indicator
        if combo_multiplier > 1:
            combo_text = f"Combo x{combo_multiplier:.1f}"
            draw_text_with_background(frame, combo_text, (frame_width - 200, 40), 
                                     'large', (255, 100, 0))
        
        # Debug information
        if debug_mode:
            debug_y = 200
            if body_part_pos:
                debug_texts = [
                    f"Target: {target_pos}",
                    f"Body Part: {body_part_pos}",
                    f"Distance: {calculate_2d_distance(body_part_pos, target_pos):.1f}px",
                    f"Threshold: {TOUCH_THRESHOLD}px",
                    f"FPS: {fps:.1f}"
                ]
                for text in debug_texts:
                    draw_text_with_background(frame, text, (20, debug_y), 'small', (255, 255, 0))
                    debug_y += 35
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        cv2.imshow('Body Tracking 2D Game', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC
            return 'menu'
        elif key == ord('s'):
            show_silhouette = not show_silhouette
            print(f"Silhouette: {'ON' if show_silhouette else 'OFF'}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('k'):
            show_skeleton = not show_skeleton
            print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
    
    return 'quit'

def main():
    """Main application entry point"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Body Tracking 2D Game")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Controls:")
    print(f"  S - Toggle silhouette")
    print(f"  D - Toggle debug mode")
    print(f"  K - Toggle skeleton")
    print(f"  ESC - Return to menu")
    print(f"  Q - Quit game")
    
    while True:
        mode = show_menu(cap, frame_width, frame_height)
        
        if mode is None or mode == 'quit':
            break
        elif mode == 'play':
            result = play_game(cap, frame_width, frame_height)
            if result == 'quit':
                break
        elif mode == 'settings':
            # Settings menu can be implemented here
            continue
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("\nThanks for playing Body Tracking 2D!")
    print(f"Your high score: {high_score}")

if __name__ == "__main__":
    main()
