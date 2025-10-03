import cv2
import mediapipe as mp
import random
import numpy as np

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
TOUCH_THRESHOLD = 50  # pixels distance to consider "touching"
score = 0
debug_mode = False  # Global debug mode flag

# Mapping body parts to MediaPipe landmarks
LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

def draw_button(frame, text, position, size, color, hover=False):
    """Draw a button on the frame"""
    x, y = position
    w, h = size
    
    # Draw button background
    button_color = tuple(min(c + 30, 255) for c in color) if hover else color
    cv2.rectangle(frame, (x, y), (x + w, y + h), button_color, -1)
    
    # Draw button border
    border_color = (255, 255, 255) if hover else (200, 200, 200)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 3)
    
    # Draw text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return (x, y, x + w, y + h)

def is_fist_closed(hand_landmarks):
    """Detect if hand is making a fist by checking finger positions"""
    # Get wrist and fingertip landmarks
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
        
        # Calculate distance from wrist
        tip_dist = ((tip_point.x - wrist.x)**2 + (tip_point.y - wrist.y)**2)**0.5
        pip_dist = ((pip_point.x - wrist.x)**2 + (pip_point.y - wrist.y)**2)**0.5
        
        # Finger is closed if tip is closer to wrist than the PIP joint
        if tip_dist < pip_dist * 1.1:
            closed_count += 1
    
    # Consider it a fist if at least 3 fingers are closed
    return closed_count >= 3

def is_point_in_button(point, button_bounds):
    """Check if a point is inside button bounds"""
    x, y = point
    x1, y1, x2, y2 = button_bounds
    return x1 <= x <= x2 and y1 <= y <= y2

def show_menu(cap, frame_width, frame_height):
    """Display the menu screen and return selected mode"""
    global debug_mode
    
    # Button settings
    button_width = 400
    button_height = 80
    button_spacing = 30
    start_y = 200
    center_x = (frame_width - button_width) // 2
    
    # Define buttons
    buttons = [
        {
            'text': 'Single Player',
            'position': (center_x, start_y),
            'size': (button_width, button_height),
            'color': (50, 150, 50),
            'mode': 'single'
        },
        {
            'text': 'Two Players',
            'position': (center_x, start_y + button_height + button_spacing),
            'size': (button_width, button_height),
            'color': (100, 100, 100),
            'mode': 'two'
        },
        {
            'text': 'Crazy Multiplayer',
            'position': (center_x, start_y + 2 * (button_height + button_spacing)),
            'size': (button_width, button_height),
            'color': (100, 100, 100),
            'mode': 'crazy'
        }
    ]
    
    hover_index = None
    fist_timer = 0
    FIST_HOLD_FRAMES = 15  # Hold fist for ~0.5 seconds at 30fps
    
    # Mouse click handler
    mouse_click_button = [None]  # Use list to modify in nested function
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check which button was clicked
            for i, bounds in enumerate(param):
                if is_point_in_button((x, y), bounds):
                    mouse_click_button[0] = i
                    break
    
    print("Menu loaded - make a fist to select!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Process hand and pose detection on RAW frame BEFORE drawing UI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        # Also process pose if debug mode is on to show skeleton in menu
        pose_results = None
        if debug_mode:
            pose_results = pose.process(rgb_frame)
        
        # NOW create dark overlay and draw UI elements
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Draw title
        title = "BODY PART TOUCH GAME"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        title_x = (frame_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 120), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw buttons
        button_bounds = []
        for i, button in enumerate(buttons):
            hover = (i == hover_index)
            bounds = draw_button(
                frame,
                button['text'],
                button['position'],
                button['size'],
                button['color'],
                hover
            )
            button_bounds.append(bounds)
        
        # Set mouse callback with button bounds
        cv2.setMouseCallback('Body Part Touch Game', mouse_callback, button_bounds)
        
        # Check if mouse clicked on single player button
        if mouse_click_button[0] == 0:
            return buttons[0]['mode']
        
        # Draw "Coming Soon" labels for disabled modes
        for i in range(1, 3):
            button = buttons[i]
            x, y = button['position']
            w, h = button['size']
            cv2.putText(frame, "Coming Soon!", (x + w + 20, y + h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        
        # Draw instructions
        instructions = [
            "Hover your hand over a button",
            "Close your fist, click mouse, or press SPACE to select",
            "Press 'd' to toggle debug mode"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (50, frame_height - 110 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Debug mode indicator in menu
        if debug_mode:
            cv2.putText(frame, "DEBUG MODE: ON", (frame_width - 250, 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            # Draw skeleton if pose was detected
            if pose_results and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        
        # Process hand detection results for cursor and fist detection
        hover_index = None
        current_fist_closed = False
        hand_x, hand_y = None, None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand position (wrist for cursor)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                hand_x = int(wrist.x * frame_width)
                hand_y = int(wrist.y * frame_height)
                
                # Check if fist is closed
                current_fist_closed = is_fist_closed(hand_landmarks)
                
                # Draw hand cursor
                cursor_color = (0, 255, 0) if current_fist_closed else (255, 0, 255)
                cv2.circle(frame, (hand_x, hand_y), 20, cursor_color, -1)
                cv2.circle(frame, (hand_x, hand_y), 25, (255, 255, 255), 2)
                
                # Debug text
                fist_text = "FIST!" if current_fist_closed else "OPEN"
                cv2.putText(frame, fist_text, (hand_x + 30, hand_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Check which button is being hovered
                for i, bounds in enumerate(button_bounds):
                    if is_point_in_button((hand_x, hand_y), bounds):
                        hover_index = i
                        break
        
        # Handle fist selection
        if current_fist_closed and hover_index is not None:
            fist_timer += 1
            
            # Draw progress indicator
            if hover_index == 0:  # Single player
                button = buttons[hover_index]
                x, y = button['position']
                w, h = button['size']
                
                # Draw progress bar
                progress = min(fist_timer / FIST_HOLD_FRAMES, 1.0)
                bar_width = int(w * progress)
                cv2.rectangle(frame, (x, y + h + 10), (x + bar_width, y + h + 20), 
                             (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 20), 
                             (255, 255, 255), 2)
                
                # Selection complete
                if fist_timer >= FIST_HOLD_FRAMES:
                    print("Selection complete!")
                    return buttons[0]['mode']
        else:
            fist_timer = 0
        
        cv2.imshow('Body Part Touch Game', frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        elif key == ord(' ') and hover_index == 0:  # Only single player works
            return buttons[0]['mode']
        elif key == ord('d'):  # Toggle debug mode with 'd' key
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    return None

def generate_target(frame_width, frame_height):
    """Generate a random target position and body part"""
    margin = 100
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def play_game(cap, frame_width, frame_height):
    """Main game loop"""
    global score, debug_mode
    score = 0
    
    target_pos, target_body_part = generate_target(frame_width, frame_height)
    
    print("Game Started!")
    print("Press 'ESC' to return to menu, 'd' to toggle debug, or 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process pose detection on RAW frame BEFORE drawing UI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Variables for debugging
        body_part_x, body_part_y = 0, 0
        distance = 0
        detection_confidence = 0
        
        if results.pose_landmarks:
            # Always draw landmarks if debug mode is on
            if debug_mode:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
            
            landmark = LANDMARK_MAP[target_body_part]
            body_part_landmark = results.pose_landmarks.landmark[landmark]
            
            body_part_x = int(body_part_landmark.x * frame_width)
            body_part_y = int(body_part_landmark.y * frame_height)
            
            # Draw circle on tracked body part
            cv2.circle(frame, (body_part_x, body_part_y), 15, (255, 0, 255), -1)
            
            distance = calculate_distance((body_part_x, body_part_y), target_pos)
            
            if distance < TOUCH_THRESHOLD:
                score += 1
                target_pos, target_body_part = generate_target(frame_width, frame_height)
        
        # Draw target point
        cv2.circle(frame, target_pos, 30, (0, 0, 255), 3)
        cv2.circle(frame, target_pos, 5, (0, 0, 255), -1)
        
        # Draw instruction text
        instruction_text = f"Touch with: {target_body_part}"
        cv2.putText(frame, instruction_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw score
        score_text = f"Score: {score}"
        cv2.putText(frame, score_text, (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Debug information overlay
        if debug_mode:
            debug_y = 120
            debug_texts = [
                f"DEBUG MODE ON",
                f"Target: {target_body_part}",
                f"Target Pos: ({target_pos[0]}, {target_pos[1]})",
                f"Body Part Pos: ({body_part_x}, {body_part_y})",
                f"Distance: {distance:.1f} px",
                f"Threshold: {TOUCH_THRESHOLD} px",
                f"Frame: {frame_width}x{frame_height}",
                f"Pose Detected: {results.pose_landmarks is not None}",
            ]
            
            # Draw semi-transparent background for debug info
            debug_bg_height = len(debug_texts) * 25 + 20
            overlay = frame.copy()
            cv2.rectangle(overlay, (frame_width - 420, debug_y - 20), 
                         (frame_width - 10, debug_y + debug_bg_height), 
                         (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            for i, text in enumerate(debug_texts):
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(frame, text, (frame_width - 410, debug_y + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(frame, "Press 'ESC' for menu, 'd' for debug, or 'q' to quit", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Body Part Touch Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == 27:  # ESC key
            return 'menu'
        elif key == ord('d'):  # Toggle debug mode
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    print(f"\nGame Over! Final Score: {score}")
    return 'quit'

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Main loop - allows returning to menu
    while True:
        # Show menu
        mode = show_menu(cap, frame_width, frame_height)
        
        if mode is None:
            break
        
        if mode == 'single':
            result = play_game(cap, frame_width, frame_height)
            if result == 'quit':
                break
            # If result is 'menu', loop continues to show menu again
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()

if __name__ == "__main__":
    main()