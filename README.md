# Body Tracking 2D Game

An interactive 2D body tracking game that uses computer vision to detect your body movements and turn them into game controls. The game removes the z-axis complexity from the original 3D version and focuses on precise 2D tracking with improved consistency.

## Features

### Core Improvements from Original
- **2D-Only Tracking**: Removed z-axis calculations for simpler, more reliable tracking
- **Enhanced Body Detection**: Multiple smoothing techniques for consistent body outline
- **Better Visual Feedback**: Smooth body silhouette with gradient effects and anti-aliased outlines
- **Advanced Tracking**: Kalman filtering and position buffering for jitter-free movement
- **Responsive Gameplay**: Combo system and visual effects for engaging experience

### Game Mechanics
- Touch targets with specific body parts (hands, elbows, head, shoulders, knees)
- Score points with combo multipliers for quick successive hits
- Real-time body silhouette visualization
- Debug mode for tracking diagnostics

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/camera connected to your computer
- Adequate lighting for body detection

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
# For the standard version with menu system
python body_tracking_2d.py

# For the enhanced version with advanced smoothing
python body_tracking_enhanced.py
```

## Game Versions

### 1. Standard Version (`body_tracking_2d.py`)
- Full menu system with mouse and gesture control
- MediaPipe Holistic model for unified tracking
- Position history smoothing
- Combo scoring system
- Multiple game modes support structure

### 2. Enhanced Version (`body_tracking_enhanced.py`)
- Kalman filtering for ultra-smooth tracking
- Advanced contour processing with spline interpolation
- Gradient body fill effects
- Glow effects on trackers and targets
- Confidence-based tracking
- Bilateral filtering for edge-preserving smoothing

## Controls

| Key | Action |
|-----|--------|
| S | Toggle body silhouette overlay |
| D | Toggle debug mode |
| K | Toggle skeleton view (standard version) |
| ESC | Return to menu / Pause |
| Q | Quit game |
| Space | Select menu option (standard version) |

## Technical Details

### Body Tracking Technology
- **MediaPipe Pose**: Detects 33 body landmarks in real-time
- **MediaPipe Holistic**: Unified model for pose, face, and hand tracking
- **Selfie Segmentation**: Separates person from background for silhouette

### Smoothing Techniques Applied
1. **Kalman Filtering**: Predicts and corrects position measurements
2. **Position Buffering**: Averages recent positions with weighted importance
3. **Spline Interpolation**: Smooths body contours for natural appearance
4. **Morphological Operations**: Cleans up segmentation masks
5. **Bilateral Filtering**: Preserves edges while smoothing surfaces

### Performance Optimizations
- Efficient frame processing pipeline
- Optimized OpenCV operations
- Smart caching of calculations
- Configurable model complexity

## Customization

### Adjusting Game Difficulty
Edit the configuration values in either game file:

```python
TOUCH_THRESHOLD = 45  # Decrease for harder, increase for easier
TARGET_RADIUS = 30    # Size of targets
TRACKER_RADIUS = 20   # Size of body part trackers
```

### Changing Visual Style
Modify color values in the GameConfig class:

```python
BODY_COLOR = (100, 255, 100)  # RGB values
TARGET_COLOR = (0, 100, 255)
TRACKER_COLOR = (255, 100, 255)
```

### Adding New Body Parts
Add entries to the BODY_PARTS list and corresponding LANDMARK_MAP/POSE_LANDMARKS dictionaries.

## Troubleshooting

### Poor Tracking Performance
- Ensure good lighting (avoid backlighting)
- Stand 4-8 feet from camera
- Wear contrasting clothing to background
- Check camera resolution (1280x720 recommended)

### Jittery Movement
- Try the enhanced version for better smoothing
- Increase POSITION_BUFFER_SIZE
- Adjust Kalman filter parameters

### Slow Frame Rate
- Reduce model_complexity in pose initialization
- Lower camera resolution
- Close other applications

## System Requirements

### Minimum
- CPU: Dual-core 2.0GHz
- RAM: 4GB
- Camera: 720p webcam
- OS: Windows 10, macOS 10.15, Ubuntu 20.04

### Recommended
- CPU: Quad-core 2.5GHz
- RAM: 8GB
- Camera: 1080p webcam with good low-light performance
- OS: Latest versions

## Known Improvements from Original

1. **Removed Z-axis complexity**: No more depth estimation issues
2. **Better body outline**: Smooth, consistent silhouette filling
3. **Improved tracking**: Multiple filtering layers prevent jitter
4. **Enhanced visuals**: Gradient effects, glow, and animations
5. **More body parts**: Added shoulders and knees for variety
6. **Cleaner code structure**: Modular design with classes
7. **Better performance**: Optimized processing pipeline

## Future Enhancements

- Multiplayer support over network
- Recording and replay system
- Custom gesture recognition
- Power-ups and special effects
- Difficulty progression
- Achievement system
- Background music and sound effects

## License

This project is provided as-is for educational and entertainment purposes.

## Credits

Built using:
- OpenCV for image processing
- MediaPipe by Google for pose detection
- NumPy for numerical operations
- SciPy for advanced filtering
- Pillow for enhanced graphics

## Comparison with Original

| Feature | Original | 2D Version |
|---------|----------|------------|
| Depth Tracking | 3D with z-axis | 2D only |
| Body Outline | Basic segmentation | Smooth, filtered contours |
| Position Tracking | Basic smoothing | Kalman + buffering |
| Visual Effects | Simple circles | Gradients, glow, animations |
| Code Structure | Single file, functional | Modular, object-oriented |
| Performance | Good | Optimized |
| Consistency | Variable | High |
