import cv2
import mediapipe as mp
import random
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Game settings
BODY_PARTS = ['Right Hand', 'Left Hand', 'Right Elbow', 'Left Elbow', 'Head']
TOUCH_THRESHOLD = 50  # pixels distance to consider "touching"
score = 0

# Mapping body parts to MediaPipe landmarks
LANDMARK_MAP = {
    'Right Hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'Left Hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'Head': mp_pose.PoseLandmark.NOSE
}

def generate_target(frame_width, frame_height):
    """Generate a random target position and body part"""
    margin = 100 # Keep targets away from edges
    x = random.randint(margin, frame_width - margin)
    y = random.randint(margin, frame_height - margin)
    body_part = random.choice(BODY_PARTS)
    return (x, y), body_part

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
    
    # Generate first target
    target_pos, target_body_part = generate_target(frame_width, frame_height)
    
    print("Body Part Touch Game Started!")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose detection
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            # Get the current target body part position
            landmark = LANDMARK_MAP[target_body_part]
            body_part_landmark = results.pose_landmarks.landmark[landmark]
            
            # Convert normalized coordinates to pixel coordinates
            body_part_x = int(body_part_landmark.x * frame_width)
            body_part_y = int(body_part_landmark.y * frame_height)
            
            # Draw a circle on the tracked body part
            cv2.circle(frame, (body_part_x, body_part_y), 15, (255, 0, 255), -1)
            
            # Calculate distance to target
            distance = calculate_distance((body_part_x, body_part_y), target_pos)
            
            # Check if body part touches the target
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
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Body Part Touch Game', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    print(f"\nGame Over! Final Score: {score}")

if __name__ == "__main__":
    main()
