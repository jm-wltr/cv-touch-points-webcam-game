# Multiplayer Body Tracking Games Guide

## Overview

I've created two enhanced multiplayer versions of the body tracking game that allow players to compete or cooperate using computer vision. These versions improve upon the original by providing better player separation, competitive gameplay, and multiple game modes.

## Files Created

### 1. **body_tracking_2player.py** - Two-Player Competitive Game
A focused 2-player competitive experience with split-screen gameplay.

### 2. **body_tracking_multiplayer_advanced.py** - Advanced Multiplayer System
Supports 2-4 players with multiple game modes and advanced features.

## Two-Player Version Features

### Core Gameplay
- **Split-Screen Display**: Players use left and right halves of the screen
- **Individual Targets**: Each player gets their own targets in their zone
- **Competitive Scoring**: First to 100 points wins
- **Visual Separation**: Different colors for each player (Blue vs Green)

### Enhanced Features
- **Combo System**: Chain hits for multiplier bonuses (up to 2x)
- **Speed Bonuses**: Extra points for quick reactions
- **Progress Bars**: Visual indication of winning progress
- **Winner Celebration**: Victory screen with final scores

### Player Tracking
- Each player has independent pose detection
- Smoothed position tracking to reduce jitter
- Color-coded skeletons and trackers
- Active player detection

## Advanced Multiplayer Version Features

### Supported Player Configurations

#### 2 Players
- Vertical split screen (left/right)
- Best for competitive play

#### 3 Players  
- One player on top, two on bottom
- Great for team vs single player

#### 4 Players
- Four quadrants layout
- Maximum chaos and fun

### Game Modes

#### 1. **Cooperative Mode**
- Players work together toward a shared goal
- Combined team score
- Shared targets that anyone can hit
- Goal: Reach team score threshold together

#### 2. **Competitive Mode** 
- Each player has individual targets
- Only you can hit your own targets
- First to win score wins
- Pure competition

#### 3. **Race Mode**
- Same target appears for all players
- First to hit gets the points
- Tests reaction speed
- Most intense competition

#### 4. **Mirror Mode** (Planned)
- Players must mirror each other's movements
- Synchronization challenge

#### 5. **Survival Mode** (Planned)
- Increasingly difficult targets
- Last player standing wins

### Visual Enhancements

#### Player Identification
- Unique colors per player:
  - Player 1: Blue
  - Player 2: Green  
  - Player 3: Red
  - Player 4: Cyan

#### Target Types
- **Normal Targets**: Standard points (10)
- **Bonus Targets**: Star-shaped, rainbow colors (20 points)
- **Speed Targets**: Time-based scoring

#### Visual Effects
- Glow effects on active trackers
- Pulsing target animations
- Smooth skeleton rendering
- Zone dividers

### Scoring System

#### Base Points
- Normal target: 10 points
- Bonus target: 20 points

#### Multipliers
- Combo multiplier: Up to 3x
- Speed bonus: +5 for quick hits
- Accuracy tracking

#### Performance Metrics
- Reaction time tracking
- Average response time
- Hit accuracy percentage
- Targets per minute

## Technical Improvements

### Better Player Separation
- **Zone-based detection**: Each player has a defined play area
- **Independent trackers**: Separate MediaPipe instances per player
- **No interference**: Players can't affect each other's tracking

### Optimized Performance
- Efficient frame splitting
- Parallel pose detection capability
- Optimized rendering pipeline
- 30 FPS target with multiple players

### Robust Tracking
- Weighted moving average smoothing
- Position history buffers
- Confidence-based filtering
- Landmark visibility checks

## How to Play

### Setup Instructions

1. **Physical Setup**:
   - Players should stand apart from each other
   - Ensure good lighting for all players
   - Camera should capture all play areas
   - Recommended: 6+ feet between players

2. **Starting the Game**:
   ```bash
   # Two-player version
   python body_tracking_2player.py
   
   # Advanced multiplayer (2 players default)
   python body_tracking_multiplayer_advanced.py
   
   # Advanced multiplayer (3 players)
   python body_tracking_multiplayer_advanced.py 3
   
   # Advanced multiplayer (4 players)
   python body_tracking_multiplayer_advanced.py 4
   ```

3. **Controls**:

   **Two-Player Version**:
   - SPACE: Start/Restart game
   - Q: Quit
   - R: Reset game
   
   **Advanced Multiplayer**:
   - 1-4: Set number of players
   - C: Cooperative mode
   - V: Versus (competitive) mode
   - R: Race mode
   - SPACE: Start/Restart
   - Q: Quit

### Gameplay Tips

#### For Best Tracking
- Wear contrasting colors to background
- Keep movements smooth and deliberate
- Stay within your designated zone
- Maintain consistent distance from camera

#### For High Scores
- Build combos by hitting targets quickly
- Prioritize bonus targets when they appear
- Learn to anticipate target positions
- Practice smooth, efficient movements

#### Strategic Tips
- In competitive: Focus on your zone
- In cooperative: Communicate with teammates
- In race: Position yourself centrally
- Watch for bonus targets (star shapes)

## Comparison with Original Multiplayer

| Feature | Original (YOLO) | New 2-Player | Advanced Multi |
|---------|----------------|--------------|----------------|
| Player Detection | YOLO + Face Recognition | MediaPipe Split | MediaPipe Zones |
| Max Players | Unlimited | 2 | 4 |
| Player Separation | AI-based tracking | Screen split | Zone-based |
| Game Modes | Cooperative only | Competitive | 5 modes |
| Visual Quality | Basic | Enhanced | Premium |
| Performance | Heavy | Optimized | Scalable |
| Setup Complexity | High | Low | Low |
| Reliability | Variable | High | High |

## Key Improvements

### Over Original 3D Version
- Removed z-axis complexity
- Better visual separation
- Competitive gameplay
- Multiple game modes

### Over Original Multiplayer
- No need for YOLO or face recognition
- Simpler setup and dependencies
- Better performance
- More reliable player separation
- Enhanced visual effects
- Multiple game modes

## System Requirements

### Minimum
- 2+ players in frame
- 720p webcam
- Good lighting
- 4GB RAM
- Dual-core CPU

### Recommended  
- 1080p webcam
- Bright, even lighting
- 8GB RAM
- Quad-core CPU
- Large play area (8x6 feet minimum)

## Troubleshooting

### Players Not Detected
- Check lighting conditions
- Ensure players are in their zones
- Verify camera can see full body
- Adjust detection confidence in code

### Overlapping Zones
- Players too close together
- Move further apart
- Use 2-player mode for better separation
- Ensure camera is positioned correctly

### Performance Issues
- Reduce number of players
- Lower camera resolution
- Close other applications
- Disable visual effects in code

## Future Enhancements

### Planned Features
- Network multiplayer support
- Tournament mode
- Player profiles and stats
- Customizable targets
- Power-ups and obstacles
- Background music/sounds
- Replay system
- Leaderboards

### Possible Expansions
- Team battles (2v2)
- King of the Hill mode
- Capture the Flag variant
- Simon Says mode
- Dance battle mode
- Fitness challenges

## Code Customization

### Adjusting Difficulty
```python
# In GameConfig class
TOUCH_THRESHOLD = 45  # Decrease for harder
WIN_SCORE = 100      # Increase for longer games
SPEED_BONUS_THRESHOLD = 1.5  # Decrease for harder speed bonus
```

### Changing Colors
```python
PLAYER_COLORS = {
    1: (255, 100, 100),  # BGR format
    2: (100, 255, 100),
    3: (100, 100, 255),
    4: (255, 255, 100)
}
```

### Adding Body Parts
```python
class BodyPart(Enum):
    RIGHT_HAND = "Right Hand"
    LEFT_HAND = "Left Hand"
    HEAD = "Head"
    # Add more here
    HIPS = "Hips"
    CHEST = "Chest"
```

## Conclusion

These multiplayer versions transform the body tracking game into a social, competitive experience. The improved player separation, multiple game modes, and enhanced visuals create an engaging multiplayer game that's easy to set up and fun to play. The removal of complex dependencies (YOLO, face recognition) makes it more accessible while maintaining high-quality gameplay.

Enjoy your multiplayer body tracking battles!
