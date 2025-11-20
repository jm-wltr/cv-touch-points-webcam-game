import cv2
import mediapipe as mp
import random
import numpy as np
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

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50
score = 0
debug_mode = False

LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

# Load fonts (Windows-compatible)
try:
    # Try Windows fonts first
    FONT_TITLE = ImageFont.truetype("arial.ttf", 60)
    FONT_LARGE = ImageFont.truetype("arial.ttf", 40)
    FONT_MEDIUM = ImageFont.truetype("arial.ttf", 28)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 20)
except:
    try:
        # Try system fonts path
        FONT_TITLE = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
        FONT_LARGE = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        FONT_MEDIUM = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        FONT_SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        # Fallback to default
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
    global debug_mode
    
    button_width = 400
    button_height = 80
    button_spacing = 30
    start_y = 200
    center_x = (frame_width - button_width) // 2
    
    buttons = [
        {
            'text': 'Single Player',
            'position': (center_x, start_y),
            'size': (button_width, button_height),
            'color': (50, 150, 50),
            'mode': 'single',
            'enabled': True
        },
        {
            'text': 'Two Players',
            'position': (center_x, start_y + button_height + button_spacing),
            'size': (button_width, button_height),
            'color': (100, 100, 100),
            'mode': 'two',
            'enabled': False
        },
        {
            'text': 'Crazy Multiplayer',
            'position': (center_x, start_y + 2 * (button_height + button_spacing)),
            'size': (button_width, button_height),
            'color': (100, 100, 100),
            'mode': 'crazy',
            'enabled': False
        }
    ]
    
    hover_index = None
    fist_timer = 0
    FIST_HOLD_FRAMES = 15
    
    mouse_click_button = [None]
    callback_set = False
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, bounds in enumerate(param):
                if is_point_in_button((x, y), bounds):
                    mouse_click_button[0] = i
                    break
    
    print("Menu loaded!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Process on RAW frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        pose_results = None
        if debug_mode:
            pose_results = pose.process(rgb_frame)
        
        # Convert to PIL for drawing
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        
        # Semi-transparent dark overlay (much lighter for Windows)
        overlay = Image.new('RGBA', pil_img.size, (20, 20, 20, 100))
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
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
            "Press 'd' to toggle debug mode"
        ]
        y_pos = frame_height - 100
        for instruction in instructions:
            draw.text((30, y_pos), instruction, font=FONT_SMALL, fill=(200, 200, 200))
            y_pos += 25
        
        # Debug indicator
        if debug_mode:
            draw.text((frame_width - 250, 30), "DEBUG MODE: ON", font=FONT_MEDIUM, fill=(0, 255, 255))
        
        # Convert back to OpenCV (RGB for compatibility)
        frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Draw skeleton if debug
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
                cv2.circle(frame, (hand_x, hand_y), 20, cursor_color, -1)
                cv2.circle(frame, (hand_x, hand_y), 25, (255, 255, 255), 2)
                
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
        
        # Show window first
        cv2.imshow('Body Part Touch Game', frame)
        
        # Set mouse callback only once after window is created
        if not callback_set:
            cv2.setMouseCallback('Body Part Touch Game', mouse_callback, button_bounds)
            callback_set = True
        
        if mouse_click_button[0] == 0:
            return 'single'
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        elif key == ord(' ') and hover_index == 0:
            return 'single'
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
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
    global score, debug_mode
    score = 0
    
    target_pos, target_body_part = generate_target(frame_width, frame_height)
    
    print("Game Started!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Convert to PIL
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)
        
        body_part_x, body_part_y = 0, 0
        distance = 0
        
        if results.pose_landmarks:
            if debug_mode:
                # Draw on opencv frame first
                frame_temp = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame_temp,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                pil_img = Image.fromarray(cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
            
            landmark = LANDMARK_MAP[target_body_part]
            body_part_landmark = results.pose_landmarks.landmark[landmark]
            
            body_part_x = int(body_part_landmark.x * frame_width)
            body_part_y = int(body_part_landmark.y * frame_height)
            
            # Draw body part tracker
            draw.ellipse([body_part_x - 15, body_part_y - 15, body_part_x + 15, body_part_y + 15], 
                        fill=(255, 0, 255))
            
            distance = calculate_distance((body_part_x, body_part_y), target_pos)
            
            if distance < TOUCH_THRESHOLD:
                score += 1
                target_pos, target_body_part = generate_target(frame_width, frame_height)
        
        # Draw target
        draw.ellipse([target_pos[0] - 30, target_pos[1] - 30, target_pos[0] + 30, target_pos[1] + 30], 
                    outline=(255, 0, 0), width=3)
        draw.ellipse([target_pos[0] - 5, target_pos[1] - 5, target_pos[0] + 5, target_pos[1] + 5], 
                    fill=(255, 0, 0))
        
        # Beautiful HUD
        draw_text_with_background(draw, f"Touch with: {target_body_part}", (20, 20), 
                                 FONT_LARGE, (255, 255, 255), (0, 0, 0, 180), padding=15)
        draw_text_with_background(draw, f"Score: {score}", (20, 80), 
                                 FONT_LARGE, (0, 255, 0), (0, 0, 0, 180), padding=15)
        
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
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                draw_text_with_background(draw, text, (frame_width - 400, debug_y + i * 30), 
                                         FONT_SMALL, color, (0, 0, 0, 200), padding=10)
        
        # Controls
        draw.text((20, frame_height - 40), "Press 'ESC' for menu, 'd' for debug, or 'q' to quit", 
                 font=FONT_SMALL, fill=(255, 255, 255))
        
        # Convert back to BGR for OpenCV
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Body Part Touch Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC key
            return 'menu'
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    return 'quit'

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    # Set camera resolution (helps with Windows compatibility)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera initialized: {frame_width}x{frame_height}")
    
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
    print("Game closed successfully!")

if __name__ == "__main__":
    main()