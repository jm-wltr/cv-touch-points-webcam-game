import cv2
import mediapipe as mp
import random
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Segmentation model for body silhouette
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50
FIST_HOLD_FRAMES = 15
TARGET_FPS = 30
BUTTON_WIDTH = 400
BUTTON_HEIGHT = 80
BUTTON_SPACING = 30
CURSOR_RADIUS = 20
TARGET_RADIUS = 30
BODY_PART_RADIUS = 15

score = 0
show_silhouette = True  # Always show silhouette by default
debug_mode = False  # Separate debug mode for skeleton

LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

# Load fonts (system fonts or use default)
try:
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    FONT_LARGE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    FONT_MEDIUM = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except:
    # Fallback for non-Mac systems
    try:
        FONT_TITLE = ImageFont.truetype("arial.ttf", 60)
        FONT_LARGE = ImageFont.truetype("arial.ttf", 40)
        FONT_MEDIUM = ImageFont.truetype("arial.ttf", 28)
        FONT_SMALL = ImageFont.truetype("arial.ttf", 20)
    except:
        print("Warning: Could not load TrueType fonts, using default fonts")
        FONT_TITLE = ImageFont.load_default()
        FONT_LARGE = ImageFont.load_default()
        FONT_MEDIUM = ImageFont.load_default()
        FONT_SMALL = ImageFont.load_default()

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
    
    # Adjust color for hover
    if hover and enabled:
        color = tuple(min(c + 30, 255) for c in color)
    
    # Draw rounded rectangle (button background)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=color)
    
    # Draw border
    border_color = (255, 255, 255) if hover else (200, 200, 200)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, outline=border_color, width=3)
    
    # Draw text centered
    font = FONT_MEDIUM
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (w - text_w) // 2
    text_y = y + (h - text_h) // 2
    
    text_color = (255, 255, 255) if enabled else (150, 150, 150)
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    return (x, y, x + w, y + h)

def apply_body_silhouette(frame, segmentation_result, color=(0, 255, 0)):
    """Apply a solid color silhouette of the body"""
    # Get the segmentation mask
    mask = segmentation_result.segmentation_mask
    
    # Create a condition where mask > 0.5 (person detected)
    condition = mask > 0.5
    
    # Create solid color overlay
    colored_overlay = np.zeros_like(frame)
    colored_overlay[:] = color
    
    # Apply the mask: show colored silhouette where person is detected
    output_image = np.where(condition[:, :, np.newaxis], colored_overlay, frame)
    
    return output_image.astype(np.uint8)

def is_fist_closed(hand_landmarks):
    """Detect if hand is making a fist"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    fingertips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    closed_count = 0
    for tip, pip in zip(fingertips, finger_pips):
        tip_point = hand_landmarks.landmark[tip]
        pip_point = hand_landmarks.landmark[pip]
        
        tip_dist = ((tip_point.x - wrist.x)**2 + (tip_point.y - wrist.y)**2)**0.5
        pip_dist = ((pip_point.x - wrist.x)**2 + (pip_point.y - wrist.y)**2)**0.5
        
        if tip_dist < pip_dist * 1.1:
            closed_count += 1
    
    return closed_count >= 3

def is_point_in_button(point, button_bounds):
    """Check if a point is inside button bounds"""
    x, y = point
    x1, y1, x2, y2 = button_bounds
    return x1 <= x <= x2 and y1 <= y <= y2

def show_menu(cap, frame_width, frame_height):
    """Display menu with beautiful PIL rendering"""
    global show_silhouette, debug_mode

    cv2.namedWindow('Body Part Touch Game', cv2.WINDOW_NORMAL)
    
    start_y = 200
    center_x = (frame_width - BUTTON_WIDTH) // 2
    
    buttons = [
        {
            'text': 'Single Player',
            'position': (center_x, start_y),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (50, 150, 50),
            'mode': 'single',
            'enabled': True
        },
        {
            'text': 'Two Players',
            'position': (center_x, start_y + BUTTON_HEIGHT + BUTTON_SPACING),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (100, 100, 100),
            'mode': 'two',
            'enabled': False
        },
        {
            'text': 'Crazy Multiplayer',
            'position': (center_x, start_y + 2 * (BUTTON_HEIGHT + BUTTON_SPACING)),
            'size': (BUTTON_WIDTH, BUTTON_HEIGHT),
            'color': (100, 100, 100),
            'mode': 'crazy',
            'enabled': False
        }
    ]
    
    hover_index = None
    fist_timer = 0
    
    mouse_click_button = [None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, bounds in enumerate(param):
                if is_point_in_button((x, y), bounds):
                    mouse_click_button[0] = i
                    break
    
    # Set up mouse callback once
    cv2.setMouseCallback('Body Part Touch Game', mouse_callback, [])
    
    print("Menu loaded!")
    
    while cap.isOpened():
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Process on RAW frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        # Process pose for debug skeleton
        pose_results = None
        if debug_mode:
            pose_results = pose.process(rgb_frame)
        
        # Always process segmentation for body silhouette
        seg_results = segmentation.process(rgb_frame)
        
        # Apply body silhouette
        if seg_results and show_silhouette:
            frame = apply_body_silhouette(frame, seg_results, color=(0, 200, 200))
        
        # Convert to PIL for drawing (RGBA mode for transparency)
        rgb_frame_for_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame_for_pil).convert('RGBA')
        overlay = Image.new('RGBA', pil_img.size, (20, 20, 20, 180))
        pil_img = Image.alpha_composite(pil_img, overlay)
        draw = ImageDraw.Draw(pil_img)
        
        # Title
        title = "BODY PART TOUCH GAME"
        bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
        title_w = bbox[2] - bbox[0]
        title_x = (frame_width - title_w) // 2
        draw.text((title_x, 80), title, font=FONT_TITLE, fill=(0, 255, 255))
        
        # Draw buttons
        button_bounds = []
        for i, button in enumerate(buttons):
            hover = (i == hover_index)
            bounds = draw_button_pil(
                draw,
                button['text'],
                button['position'],
                button['size'],
                button['color'],
                hover,
                button['enabled']
            )
            button_bounds.append(bounds)
        
        # Update mouse callback with current button bounds
        cv2.setMouseCallback('Body Part Touch Game', mouse_callback, button_bounds)
        
        # Coming soon labels
        for i in range(1, 3):
            button = buttons[i]
            x, y = button['position']
            w, h = button['size']
            draw.text((x + w + 20, y + h // 2), "Coming Soon!", font=FONT_SMALL, fill=(150, 150, 150))
        
        # Instructions
        instructions = [
            "Hover your hand over a button",
            "Close your fist, click mouse, or press SPACE to select",
            "Press 's' to toggle silhouette, 'd' for debug mode"
        ]
        y_pos = frame_height - 100
        for instruction in instructions:
            draw.text((30, y_pos), instruction, font=FONT_SMALL, fill=(200, 200, 200))
            y_pos += 25
        
        # Status indicators
        status_y = 30
        if show_silhouette:
            draw.text((frame_width - 250, status_y), "Silhouette: ON", font=FONT_SMALL, fill=(0, 255, 200))
            status_y += 30
        if debug_mode:
            draw.text((frame_width - 250, status_y), "DEBUG MODE: ON", font=FONT_SMALL, fill=(255, 255, 0))
        
        # Convert back to OpenCV (RGB)
        frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Draw skeleton if debug mode (on top of everything)
        if debug_mode and pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        
        # Process hand cursor
        hover_index = None
        current_fist_closed = False
        hand_x, hand_y = None, None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                hand_x = int(wrist.x * frame_width)
                hand_y = int(wrist.y * frame_height)
                
                current_fist_closed = is_fist_closed(hand_landmarks)
                
                # Draw hand cursor
                cursor_color = (0, 255, 0) if current_fist_closed else (255, 0, 255)
                cv2.circle(frame, (hand_x, hand_y), CURSOR_RADIUS, cursor_color, -1)
                cv2.circle(frame, (hand_x, hand_y), CURSOR_RADIUS + 5, (255, 255, 255), 2)
                
                # Status text
                fist_text = "FIST!" if current_fist_closed else "OPEN"
                cv2.putText(frame, fist_text, (hand_x + 30, hand_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Check hover
                for i, bounds in enumerate(button_bounds):
                    if is_point_in_button((hand_x, hand_y), bounds):
                        hover_index = i
                        break
        
        # Handle fist selection
        if current_fist_closed and hover_index == 0:
            fist_timer += 1
            
            button = buttons[0]
            x, y = button['position']
            w, h = button['size']
            
            progress = min(fist_timer / FIST_HOLD_FRAMES, 1.0)
            bar_width = int(w * progress)
            cv2.rectangle(frame, (x, y + h + 10), (x + bar_width, y + h + 20), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 20), (255, 255, 255), 2)
            
            if fist_timer >= FIST_HOLD_FRAMES:
                return 'single'
        else:
            fist_timer = 0
        
        if mouse_click_button[0] == 0:
            return 'single'
        
        cv2.imshow('Body Part Touch Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        elif key == ord(' ') and hover_index == 0:
            return 'single'
        elif key == ord('s'):
            show_silhouette = not show_silhouette
            print(f"Silhouette view: {'ON' if show_silhouette else 'OFF'}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode (skeleton): {'ON' if debug_mode else 'OFF'}")
        
        # FPS limiting
        frame_time = time.time() - frame_start
        wait_time = max(1, int((1.0 / TARGET_FPS - frame_time) * 1000))
        cv2.waitKey(wait_time)
    
    return None

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generate_target(frame_width, frame_height):
    margin = 100
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def play_game(cap, frame_width, frame_height):
    global score, show_silhouette, debug_mode
    score = 0
    
    target_pos, target_body_part = generate_target(frame_width, frame_height)
    
    window_name = 'Body Part Touch Game'
    cv2.namedWindow(window_name)
    
    print("Game Started!")
    
    while cap.isOpened():
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Always process segmentation for body silhouette
        seg_results = segmentation.process(rgb_frame)
        
        # Apply body silhouette
        if seg_results and show_silhouette:
            frame = apply_body_silhouette(frame, seg_results, color=(0, 200, 200))
        
        # Convert to PIL (RGBA mode)
        rgb_frame_for_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame_for_pil).convert('RGBA')
        draw = ImageDraw.Draw(pil_img)
        
        body_part_x, body_part_y = 0, 0
        distance = 0
        
        if results.pose_landmarks:
            landmark = LANDMARK_MAP[target_body_part]
            body_part_landmark = results.pose_landmarks.landmark[landmark]
            
            body_part_x = int(body_part_landmark.x * frame_width)
            body_part_y = int(body_part_landmark.y * frame_height)
            

            
            # Draw body part tracker
            draw.ellipse([body_part_x - BODY_PART_RADIUS, body_part_y - BODY_PART_RADIUS, 
                         body_part_x + BODY_PART_RADIUS, body_part_y + BODY_PART_RADIUS], 
                        fill=(255, 0, 255))
            
            distance = calculate_distance((body_part_x, body_part_y), target_pos)
            
            if distance < TOUCH_THRESHOLD:
                score += 1
                target_pos, target_body_part = generate_target(frame_width, frame_height)
        
        # Draw target
        draw.ellipse([target_pos[0] - TARGET_RADIUS, target_pos[1] - TARGET_RADIUS, 
                     target_pos[0] + TARGET_RADIUS, target_pos[1] + TARGET_RADIUS], 
                    outline=(255, 0, 0), width=3)
        draw.ellipse([target_pos[0] - 5, target_pos[1] - 5, target_pos[0] + 5, target_pos[1] + 5], 
                    fill=(255, 0, 0))
        
        # Beautiful HUD
        draw_text_with_background(draw, f"Touch with: {target_body_part}", (20, 20), 
                                 FONT_LARGE, (255, 255, 255), (0, 0, 0, 180), padding=15)
        draw_text_with_background(draw, f"Score: {score}", (20, 80), 
                                 FONT_LARGE, (0, 255, 0), (0, 0, 0, 180), padding=15)
        
        # Status indicators
        status_y = 20
        if show_silhouette:
            draw_text_with_background(draw, "Silhouette: ON", (frame_width - 250, status_y), 
                                     FONT_SMALL, (0, 255, 200), (0, 0, 0, 180), padding=8)
            status_y += 35
        
        # Debug info
        if debug_mode:
            debug_texts = [
                "DEBUG MODE ON",
                f"Target: {target_body_part}",
                f"Target Pos: ({target_pos[0]}, {target_pos[1]})",
                f"Body Part Pos: ({body_part_x}, {body_part_y})",
                f"Distance: {distance:.1f} px",
                f"Threshold: {TOUCH_THRESHOLD} px"
            ]
            debug_y = 120
            for i, text in enumerate(debug_texts):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                draw_text_with_background(draw, text, (frame_width - 450, debug_y + i * 30), 
                                         FONT_SMALL, color, (0, 0, 0, 200), padding=10)
        
        # Controls
        draw.text((20, frame_height - 40), "Press 'ESC' for menu, 's' silhouette, 'd' debug, or 'q' to quit", 
                 font=FONT_SMALL, fill=(255, 255, 255))
        
        # Convert back to OpenCV
        frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Draw skeleton if debug mode (on top of everything)
        if debug_mode and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC key
            return 'menu'
        elif key == ord('s'):
            show_silhouette = not show_silhouette
            print(f"Silhouette view: {'ON' if show_silhouette else 'OFF'}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode (skeleton): {'ON' if debug_mode else 'OFF'}")
        
        # FPS limiting
        frame_time = time.time() - frame_start
        wait_time = max(1, int((1.0 / TARGET_FPS - frame_time) * 1000))
        cv2.waitKey(wait_time)
    
    return 'quit'

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam resolution: {frame_width}x{frame_height}")
    print(f"Body silhouette is ON by default. Press 's' to toggle.")
    print(f"Press 'd' for debug mode (skeleton view).")
    
    while True:
        mode = show_menu(cap, frame_width, frame_height)
        
        if mode is None:
            break
        
        if mode == 'single':
            result = play_game(cap, frame_width, frame_height)
            if result == 'quit':
                break
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    segmentation.close()
    print("Game closed. Thanks for playing!")

if __name__ == "__main__":
    main()