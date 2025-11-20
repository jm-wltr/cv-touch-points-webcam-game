# Body Part Touch Game - Version Comparison

## Overview
This document compares three different implementations of the body part touch game, each with unique approaches to pose detection, rendering, and game mechanics.

---

## Version Summary Table

| Feature | Jaime's Version | Kristian's Version | Jay's Version |
|---------|----------------|-------------------|---------------|
| **Model** | MediaPipe Pose + Hands | MediaPipe Holistic | YOLOv8n-pose + InsightFace |
| **Players** | Single Player | Single Player | Multi-Player (Unlimited) |
| **UI Quality** | High (PIL-based) | Medium (OpenCV-based) | Basic (OpenCV-based) |
| **Key Feature** | Fist detection menu | Body silhouetting | Multi-person tracking |
| **Tracking** | Basic frame-by-frame | Smoothed positions | Advanced re-identification |
| **Performance** | Good | Good | Moderate (heavy models) |
| **Complexity** | Low-Medium | Medium | High |

---

## 1. JAIME'S VERSION (jaime.py)

### Architecture
- **Primary Focus**: Polished UI/UX with gesture-based menu interaction
- **Target Audience**: Single player with intuitive controls

### Models Used
1. **MediaPipe Pose** (`mp_pose.Pose`)
   - Detection confidence: 0.5
   - Tracking confidence: 0.5
   - Used for: Body part tracking during gameplay

2. **MediaPipe Hands** (`mp_hands.Hands`)
   - Detection confidence: 0.7
   - Tracking confidence: 0.7
   - Max hands: 1
   - Used for: Fist detection in menu navigation

### Key Features
- PIL/ImageDraw for high-quality text rendering
- Fist gesture detection for menu selection
- Mouse, keyboard, and gesture control support
- Debug mode with skeleton overlay
- Beautiful button rendering with rounded corners

### High-Level Pseudocode

```
INITIALIZATION:
    Initialize MediaPipe Pose model
    Initialize MediaPipe Hands model
    Load fonts (Helvetica/Arial) for PIL rendering
    Define body parts: Right Hand, Left Hand, Right Elbow, Left Elbow, Head
    Set touch threshold: 50 pixels

MAIN LOOP:
    Open webcam
    While running:
        mode = show_menu()
        If mode == 'single':
            play_game()
        Else:
            break
    Cleanup and close

SHOW_MENU:
    Initialize buttons:
        - Single Player (enabled)
        - Two Players (coming soon)
        - Crazy Multiplayer (coming soon)

    While in menu:
        Capture frame from webcam
        Flip frame horizontally

        PROCESS:
            Run hands model on frame (for gesture detection)
            If debug_mode: run pose model on frame

        RENDER:
            Convert frame to PIL Image
            Draw dark overlay (semi-transparent)
            Draw title text with PIL fonts
            Draw buttons with rounded corners
            Draw instructions at bottom
            If debug_mode: draw pose skeleton
            Convert back to OpenCV format

        HAND TRACKING:
            If hand detected:
                Get wrist position
                Check if fist is closed (compare fingertip vs PIP distances)
                Draw hand cursor (green if fist, magenta if open)
                Check which button hand hovers over
                If fist held for 15 frames on enabled button:
                    Return 'single'

        KEYBOARD INPUT:
            'q' = quit
            'd' = toggle debug mode
            'space' = select hovered button

        MOUSE INPUT:
            Click on button = select mode

PLAY_GAME:
    Reset score to 0
    Generate random target (position + body part)

    While playing:
        Capture frame from webcam
        Flip frame horizontally

        PROCESS:
            Run pose model on frame
            Extract target body part landmark
            Convert to pixel coordinates

        COLLISION DETECTION:
            Calculate Euclidean distance between:
                - Body part position
                - Target position
            If distance < TOUCH_THRESHOLD (50px):
                Increment score
                Generate new target

        RENDER (using PIL):
            Convert frame to PIL Image
            If debug_mode: draw pose skeleton
            Draw target circle (30px radius, red outline)
            Draw body part tracker (15px circle, magenta)
            Draw HUD with backgrounds:
                - Current target body part
                - Score
            If debug_mode:
                - Show target/body part coordinates
                - Show distance to target
                - Show threshold
            Draw controls at bottom
            Convert back to OpenCV format

        DISPLAY:
            Show frame

        KEYBOARD INPUT:
            'q' = quit game
            'ESC' = return to menu
            'd' = toggle debug mode

IS_FIST_CLOSED:
    Get wrist landmark
    For each finger (index, middle, ring, pinky):
        Get fingertip and PIP joint positions
        Calculate distance from wrist to fingertip
        Calculate distance from wrist to PIP
        If fingertip_distance < PIP_distance * 1.1:
            Increment closed_count
    Return True if closed_count >= 3
```

### Strengths
- **Excellent UI/UX**: PIL rendering provides crisp, beautiful text
- **Intuitive controls**: Fist detection is creative and fun
- **Multiple input methods**: Mouse, keyboard, and gestures
- **Clean code structure**: Well-organized with helper functions

### Weaknesses
- **Single player only**: No multiplayer support
- **No position smoothing**: Can be jittery
- **Basic game mechanics**: Simple touch-and-score system

---

## 2. KRISTIAN'S VERSION (kristian.py)

### Architecture
- **Primary Focus**: Body silhouetting and visual feedback
- **Target Audience**: Single player with emphasis on visual immersion

### Models Used
1. **MediaPipe Holistic** (`mp_holistic.Holistic`)
   - Detection confidence: 0.7
   - Tracking confidence: 0.8
   - Model complexity: 1 (medium)
   - Smooth landmarks: True
   - Enable segmentation: True
   - Smooth segmentation: True
   - Used for: Full body tracking + segmentation mask

### Key Features
- **Body segmentation**: Creates silhouette overlay with smooth edges
- **Position smoothing**: Uses deque to average last 3 frames
- **Combo system**: Multiplier increases with quick consecutive touches (up to 5x)
- **High score tracking**: Persistent across game sessions
- **Pulsing targets**: Visual feedback with sine wave animation
- **Multiple toggles**: Silhouette, skeleton, debug modes

### High-Level Pseudocode

```
INITIALIZATION:
    Initialize MediaPipe Holistic model with segmentation
    Load fonts with OS-specific fallbacks (macOS/Windows/Linux)
    Define body parts: Right Hand, Left Hand, Elbows, Head, Shoulders
    Set touch threshold: 40 pixels
    Initialize position history (deque with size 3 for smoothing)

MAIN LOOP:
    Open webcam with 1280x720 @ 30fps
    While running:
        mode = show_menu()
        If mode == 'play':
            play_game()
        Else:
            break
    Cleanup and close

SHOW_MENU:
    Initialize buttons:
        - Start Game
        - Settings (enabled)
        - Quit

    While in menu:
        Capture frame from webcam
        Flip frame horizontally

        PROCESS:
            Run holistic model on frame
            Get pose landmarks
            Get segmentation mask

        SEGMENTATION:
            If silhouette mode enabled:
                Create binary mask from segmentation
                Apply morphological operations (close + open)
                Apply Gaussian blur for smooth edges
                Create colored overlay (green)
                Blend with original frame (70% opacity)
                Find and draw contours

        RENDER:
            Draw title text
            Draw buttons with rounded corners
            If pose detected:
                Get right hand position
                Draw hand cursor
                Check button hover

        KEYBOARD INPUT:
            'q'/'ESC' = quit
            's' = toggle silhouette
            'd' = toggle debug
            'k' = toggle skeleton
            'space'/'enter' = select button

PLAY_GAME:
    Reset score to 0
    Initialize combo multiplier = 1
    Clear position history
    Generate random target

    While playing:
        Capture frame from webcam
        Flip frame horizontally

        PROCESS:
            Run holistic model on frame
            Get pose landmarks
            Get segmentation mask

        SEGMENTATION:
            If silhouette mode enabled:
                Apply body overlay (60% opacity)

        POSITION SMOOTHING:
            For target body part:
                Get raw landmark position
                Add to position history (deque)
                Calculate smoothed position = average of last 3 positions

        COLLISION DETECTION:
            Calculate 2D Euclidean distance
            If distance < TOUCH_THRESHOLD (40px):
                Check time since last touch
                If < 2 seconds:
                    combo_multiplier = min(combo_multiplier + 0.5, 5.0)
                Else:
                    combo_multiplier = 1.0

                points = 10 * combo_multiplier
                Add points to score
                Update high_score if needed
                Generate new target

        RENDER:
            Draw pulsing target:
                pulse = abs(sin(time * 3)) * 0.3 + 0.7
                radius = BASE_RADIUS * pulse
            Draw body part tracker (magenta, 18px)
            If skeleton mode: draw pose connections
            Draw HUD:
                - Score (green)
                - High score (yellow)
                - Target body part (white)
                - Status indicators (Silhouette/Debug/Skeleton)
            If combo > 1:
                Draw combo multiplier indicator (orange)
            If debug mode:
                - Target position
                - Body part position
                - Distance
                - Threshold
                - FPS
            Draw controls at bottom

        FPS CALCULATION:
            Count frames
            Every 30 frames: calculate average FPS

        KEYBOARD INPUT:
            'q' = quit
            'ESC' = return to menu
            's' = toggle silhouette
            'd' = toggle debug
            'k' = toggle skeleton

SMOOTH_POSITION(body_part, new_position):
    Add new_position to position_history[body_part]
    If history empty: return None
    Calculate average X = sum(all X) / count
    Calculate average Y = sum(all Y) / count
    Return (avg_X, avg_Y)

CREATE_BODY_MASK(segmentation_mask):
    Convert to binary (threshold 0.5)
    Apply morphological closing (ellipse kernel 5x5)
    Apply morphological opening (ellipse kernel 5x5)
    Apply Gaussian blur (7x7, sigma=2)
    Re-threshold at 128
    Return smooth binary mask

APPLY_BODY_OVERLAY(frame, segmentation_mask, opacity):
    Create smooth mask
    Create colored overlay (green)
    Apply mask to overlay
    Blend with original frame using opacity
    Find contours on mask
    Draw contours as outline
    Return result
```

### Strengths
- **Best visual feedback**: Silhouette overlay is unique and immersive
- **Smooth tracking**: Position averaging reduces jitter significantly
- **Engaging mechanics**: Combo system encourages quick play
- **High polish**: Pulsing targets, smooth edges, good feedback
- **Multiple display modes**: Skeleton, silhouette, debug options

### Weaknesses
- **Single player only**: No multiplayer support
- **Complex dependencies**: Requires segmentation-capable model
- **Performance overhead**: Segmentation and morphological operations add latency

---

## 3. JAY'S VERSION (jay.py)

### Architecture
- **Primary Focus**: Multi-person tracking with re-identification
- **Target Audience**: Multiple players simultaneously

### Models Used
1. **YOLOv8n-pose** (Ultralytics)
   - Model: `yolov8n-pose.pt` (nano size for speed)
   - Used for: Real-time multi-person pose detection
   - Outputs: Bounding boxes + 17 keypoints per person

2. **InsightFace** (ArcFace recognition)
   - Provider: CPUExecutionProvider
   - Detection size: 640x640
   - Used for: Face recognition and person re-identification
   - Outputs: 512-dimensional face embeddings

### Key Features
- **Multi-person tracking**: Unlimited simultaneous players
- **Re-identification**: Persistent IDs using face recognition
- **Registration window**: 30-frame period to register players
- **Hungarian algorithm**: Optimal detection-to-track matching
- **Team scoring**: All players must touch targets to score
- **Consistent colors**: Each player gets unique color across frames

### High-Level Pseudocode

```
INITIALIZATION:
    Load YOLOv8n-pose model
    Initialize InsightFace recognizer
    Define body parts: Right Hand, Left Hand, Elbows, Head
    Set touch threshold: 50 pixels
    Initialize MultiPersonTracker:
        - track_thresh: 0.5 (minimum detection confidence)
        - match_thresh: 0.65 (IoU + pose similarity for matching)
        - reid_thresh: 0.55 (face similarity for re-ID)
        - track_buffer: 90 frames (keep lost tracks)
        - registration_window: 30 frames (accept new players)

MAIN LOOP:
    Open webcam
    While running:
        Capture frame
        Flip horizontally

        DETECTION:
            Run YOLO model on frame
            Extract detections:
                - Bounding boxes
                - Confidence scores
                - 17 keypoints per person

        TRACKING:
            tracks = tracker.update(detections, frame)

        TARGET MANAGEMENT:
            Get current player IDs from tracks
            If registration window just ended:
                Generate targets for all registered players
            Add targets for new players
            Remove targets for lost players

        GAME LOGIC:
            For each tracked player:
                Get target body part landmark
                If landmark visible (confidence > 0.5):
                    Calculate distance to target
                    If distance < TOUCH_THRESHOLD:
                        Mark target as touched

            If ALL players touched their targets:
                Increment team score
                Generate new targets for all players

        RENDERING:
            For each track:
                Draw bounding box (player's color)
                Draw player label
                Draw skeleton with keypoints
                Draw body part tracker (15px circle)

            For each player target:
                Draw target circle (filled if touched, hollow if not)
                Draw body part label

            Draw UI:
                - Team score
                - Progress (X/Y targets touched)
                - Registration status or player count
                - FPS counter
                - Instructions

        KEYBOARD INPUT:
            'q' = quit
            'r' = reset tracker and game

        Display frame

MULTI_PERSON_TRACKER CLASS:

    ATTRIBUTES:
        tracked_tracks: Dictionary of currently visible tracks
        lost_tracks: Dictionary of temporarily lost tracks
        face_database: Dictionary of face embeddings per track_id
        registered_ids: Set of player IDs allowed in game
        frame_id: Current frame counter
        track_id_count: Global counter for new IDs

    UPDATE(detections, frame):
        Increment frame_id

        FILTERING:
            high_dets = detections with score >= track_thresh
            allow_new_registrations = (frame_id <= registration_window)

        STEP 1: MATCH EXISTING TRACKS TO DETECTIONS
            Initialize unmatched_tracks = all tracked track IDs

            matches, unmatched_tracks, unmatched_dets =
                match_detections_to_tracks(
                    track_ids=unmatched_tracks,
                    detections=high_dets,
                    threshold=match_thresh
                )

            For each match (track_id, det_idx):
                Update tracked_tracks[track_id] with detection data

                If enough frames passed since last face extract:
                    Extract face region from keypoints
                    Get face embedding from InsightFace
                    Update face_database with EMA

        STEP 2: HANDLE UNMATCHED DETECTIONS
            Extract face features for all unmatched detections

            RE-IDENTIFICATION:
                For each unmatched detection:
                    If face features available:
                        Find best match in face_database
                        If similarity >= reid_thresh:
                            Recover track from lost_tracks or create new
                            Update with detection data
                            Update face_database
                            Remove from unmatched_dets

            NEW REGISTRATIONS:
                If allow_new_registrations:
                    For each remaining unmatched detection:
                        Create new track with new ID
                        Add to tracked_tracks
                        If face features available:
                            Add to face_database
                        Add ID to registered_ids
                        Increment track_id_count

        STEP 3: TRACK LIFECYCLE MANAGEMENT
            For each unmatched track:
                Move from tracked_tracks to lost_tracks

            For each lost track:
                If frame_id - last_update > track_buffer:
                    Delete from lost_tracks

        Return list of currently tracked tracks

    MATCH_DETECTIONS_TO_TRACKS(track_ids, detections, threshold):
        If empty: return no matches

        BUILD SIMILARITY MATRIX:
            For each (track, detection) pair:
                iou = compute_iou(track.bbox, det.bbox)
                pose_sim = compute_pose_similarity(track.kps, det.kps)
                similarity = 0.5 * iou + 0.5 * pose_sim

        HUNGARIAN ALGORITHM:
            Maximize: Sum of similarities for assigned pairs
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        FILTER LOW SIMILARITIES:
            matches = pairs where similarity >= threshold
            unmatched_tracks = tracks not in matches
            unmatched_dets = detections not in matches

        Return matches, unmatched_tracks, unmatched_dets

    COMPUTE_IOU(box1, box2):
        Calculate intersection area
        Calculate union area
        Return intersection / union

    COMPUTE_POSE_SIMILARITY(kps1, kps2):
        Find keypoints visible in both poses
        Calculate distances between corresponding keypoints
        Normalize by bbox scale
        Use sigmas (Object Keypoint Similarity formula)
        Return mean OKS score

    GET_FACE_BBOX_FROM_KEYPOINTS(keypoints, person_bbox):
        Get head keypoints (indices 0-4: nose, eyes, ears)
        Find visible head keypoints
        If >= 2 visible:
            Get bounding box of head keypoints
            Add padding (80% of size)
        Else:
            Use top 1/3 of person bounding box
        Return face region

INSIGHTFACE_RECOGNIZER CLASS:

    EXTRACT_FACE_FEATURES(frame, bbox):
        Crop face region from frame
        Run InsightFace face detection on crop
        If faces found:
            Select face with highest detection score
            Return 512D embedding vector
        Else:
            Return None

    COMPARE_FEATURES(feat1, feat2):
        Calculate cosine similarity:
            similarity = dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        Normalize to [0, 1]:
            similarity = (similarity + 1) / 2
        Return similarity

GENERATE_TARGETS_FOR_PLAYERS(frame_width, frame_height, player_ids):
    targets = {}
    For each player_id:
        Generate random position (with margins)
        Select random body part
        targets[player_id] = {pos, body_part, touched=False}
    Return targets

GET_TRACK_COLOR(track_id):
    If color not assigned:
        Seed random with track_id
        Generate random RGB color
        Cache color
    Return cached color
```

### Strengths
- **Multi-player support**: Truly revolutionary for this game type
- **Robust tracking**: Face re-identification handles occlusions and re-entries
- **Scalable**: Can handle many players simultaneously
- **Team mechanics**: Creates cooperative gameplay
- **Advanced CV**: Demonstrates state-of-the-art tracking techniques
- **Persistent IDs**: Players maintain identity across frames

### Weaknesses
- **High complexity**: Hardest to understand and maintain
- **Heavy dependencies**: Requires YOLO + InsightFace (large models)
- **Performance**: Slower due to face recognition (especially on CPU)
- **Limited UI**: Basic text rendering, no polished menus
- **Registration window**: Requires players to be present at start

---

## Detailed Feature Comparison

### Pose Detection Models

#### MediaPipe Pose (Jaime)
- **Type**: BlazePose (Google)
- **Keypoints**: 33 landmarks
- **Speed**: ~30-60 FPS on CPU
- **Accuracy**: Good for single person
- **Pros**: Fast, accurate, built-in tracking
- **Cons**: Single person only, struggles with occlusions

#### MediaPipe Holistic (Kristian)
- **Type**: BlazePose + BlazeFace + BlazePalm
- **Keypoints**: 33 pose + 21 per hand + 468 face
- **Speed**: ~20-30 FPS on CPU
- **Accuracy**: Excellent for single person
- **Pros**: Includes segmentation, very detailed
- **Cons**: Slower, single person only

#### YOLOv8n-pose (Jay)
- **Type**: CNN-based object detection + pose estimation
- **Keypoints**: 17 COCO keypoints
- **Speed**: ~15-25 FPS on CPU (varies with # of people)
- **Accuracy**: Good for multiple people
- **Pros**: Multi-person, robust to occlusions
- **Cons**: Fewer keypoints, needs more compute

### Rendering Approaches

#### PIL/ImageDraw (Jaime)
- High-quality text with TrueType fonts
- Smooth anti-aliasing
- Rounded rectangles
- Text bounding boxes
- Best for UI elements
- Requires format conversions (RGB ↔ BGR)

#### OpenCV (Kristian & Jay)
- Direct drawing on frame
- No format conversions needed
- Faster but lower quality text
- Good enough for game overlay
- Built-in shape primitives

### Game Mechanics

#### Jaime's Approach
```
Single target at a time
Single player
Instant respawn on touch
Simple scoring (1 point per touch)
```

#### Kristian's Approach
```
Single target at a time
Single player
Instant respawn on touch
Combo multiplier (1x to 5x)
Points = 10 * multiplier
Combo resets after 2 seconds
```

#### Jay's Approach
```
One target per player
Multiple players
Respawn when ALL players touch
Team scoring
Cooperative gameplay
Progress tracking (X/Y complete)
```

---

## Performance Analysis

### Frame Processing Pipeline

#### Jaime
```
Capture (640x480) → Flip
    ↓
MediaPipe Pose (10-15ms)
    ↓
RGB→BGR conversion
    ↓
PIL rendering (5-10ms)
    ↓
RGB←BGR conversion
    ↓
Display
Total: ~20-30ms (30-50 FPS)
```

#### Kristian
```
Capture (1280x720) → Flip
    ↓
MediaPipe Holistic (25-40ms)
    ↓
Segmentation processing (5-10ms)
  - Morphological ops
  - Gaussian blur
  - Contour finding
    ↓
Position smoothing (1ms)
    ↓
OpenCV rendering (2-5ms)
    ↓
Display
Total: ~35-60ms (15-30 FPS)
```

#### Jay
```
Capture (640x480) → Flip
    ↓
YOLO inference (30-60ms)
    ↓
InsightFace (per person, 10-30ms)
    ↓
Tracking update (5-10ms)
  - IoU computation
  - Pose similarity
  - Hungarian algorithm
  - Face matching
    ↓
OpenCV rendering (2-5ms)
    ↓
Display
Total: ~50-110ms (9-20 FPS)
```

### Memory Usage

| Version | Model Size | RAM Usage | VRAM Usage (GPU) |
|---------|-----------|-----------|------------------|
| Jaime | ~50 MB | ~200 MB | N/A |
| Kristian | ~60 MB | ~250 MB | N/A |
| Jay | ~400 MB | ~800 MB | ~1-2 GB |

---

## Code Quality & Maintainability

### Jaime's Version
- **Structure**: Functional, well-organized
- **Comments**: Adequate
- **Modularity**: Good separation of concerns
- **Dependencies**: Minimal (cv2, mediapipe, PIL, numpy)
- **Error Handling**: Basic try-catch for fonts
- **Configuration**: Constants at top
- **Extensibility**: Easy to add game modes

### Kristian's Version
- **Structure**: Functional, clean
- **Comments**: Good documentation
- **Modularity**: Excellent helper functions
- **Dependencies**: Moderate (adds platform)
- **Error Handling**: OS-specific font fallbacks
- **Configuration**: Well-organized constants
- **Extensibility**: Easy to add features

### Jay's Version
- **Structure**: Object-oriented (tracker class)
- **Comments**: Detailed documentation
- **Modularity**: Excellent (separate tracker class)
- **Dependencies**: Heavy (ultralytics, insightface, scipy)
- **Error Handling**: Try-catch in critical sections
- **Configuration**: Flexible tracker parameters
- **Extensibility**: Tracker can be reused

---

## Use Case Recommendations

### Choose Jaime's Version If:
- You want the best UI/UX
- Single player is sufficient
- You need gesture-based controls
- You want fast, responsive gameplay
- You're deploying on limited hardware

### Choose Kristian's Version If:
- You want visual immersion (silhouette)
- You need smooth, jitter-free tracking
- You want engaging scoring mechanics (combos)
- You need multiple display modes
- You want to showcase segmentation

### Choose Jay's Version If:
- You need multi-player support
- You want persistent player identities
- You're building a competitive/cooperative game
- You have access to good hardware
- You need state-of-the-art tracking

---

## Technical Innovations

### Jaime
- **Fist detection for UI**: Creative use of hand landmarks for gesture control
- **PIL text rendering**: High-quality fonts in OpenCV application
- **Multi-input support**: Mouse, keyboard, and gestures seamlessly integrated

### Kristian
- **Position smoothing**: Deque-based averaging for stable tracking
- **Body silhouetting**: Morphological operations for smooth segmentation
- **Combo system**: Time-based multiplier for engaging gameplay
- **Pulsing targets**: Sine wave animation for visual feedback

### Jay
- **Multi-person tracking**: Hungarian algorithm for optimal assignment
- **Face re-identification**: InsightFace for persistent identities
- **Registration window**: Smart system for player onboarding
- **OKS scoring**: Object Keypoint Similarity for pose matching
- **EMA face features**: Exponential moving average for stable embeddings

---

## Conclusion

All three versions demonstrate strong computer vision skills with different priorities:

- **Jaime**: Best for single-player experience with polished UI
- **Kristian**: Best for visual immersion and smooth gameplay
- **Jay**: Best for multi-player scenarios and advanced tracking

For a production game, the ideal approach would combine:
- Jaime's UI/UX polish
- Kristian's position smoothing and visual effects
- Jay's multi-player architecture (simplified)

This comparison shows the team's diverse skill set covering UI design, signal processing, and advanced CV algorithms.
