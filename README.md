# Body Game

A simple webcam-based body movement game using **OpenCV** and **MediaPipe**.  
Touch the target points with the correct body part to score points — but be quick, or you lose!

## Setup

```bash
# 1. Create a new conda environment with Python 3.11
conda create -n body_game python=3.11 -y

# 2. Activate the environment
conda activate body_game

# 3. Install required packages
pip install opencv-python mediapipe numpy

# 4. Run the game
python main.py
```