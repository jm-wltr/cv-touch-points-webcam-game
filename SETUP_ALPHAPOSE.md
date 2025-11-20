# AlphaPose Game Setup Guide

## What is AlphaPose?

AlphaPose is a state-of-the-art multi-person pose estimation system from SJTU.

**Advantages:**
- ✅ Industry-leading accuracy
- ✅ Robust multi-person tracking
- ✅ Real-time performance (with GPU)
- ✅ Better than OpenPose in many scenarios
- ✅ Active development and support

**Performance:**
- **With GPU**: 20-30 FPS (2 players)
- **CPU only**: 5-10 FPS (2 players) - Not recommended

---

## Installation

### Option 1: Quick Install (Recommended for Testing)

The game includes a **fallback mode** using MediaPipe if AlphaPose isn't installed:

```bash
# Install basic dependencies
pip install opencv-python numpy torch torchvision mediapipe

# Run the game (will use MediaPipe fallback)
python alphapose_game.py
```

This lets you test the game structure while setting up AlphaPose.

### Option 2: Full AlphaPose Installation

AlphaPose requires specific setup:

#### Step 1: Install PyTorch

```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slow, not recommended)
pip install torch torchvision
```

Check CUDA: `nvcc --version` or `nvidia-smi`

#### Step 2: Install AlphaPose

**Method A: From PyPI (easiest)**
```bash
pip install alphapose
```

**Method B: From Source (recommended for latest)**
```bash
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
pip install -r requirements.txt
python setup.py install
```

#### Step 3: Download Pre-trained Models

```bash
# Download fast ResNet-50 model (~200MB)
cd AlphaPose
python scripts/download_models.py

# Or download manually:
# https://github.com/MVIG-SJTU/AlphaPose/releases
```

Models needed:
- `fast_res50_256x192.pth` - Pose model
- `yolov3-spp.weights` - Person detector (or use built-in YOLO)

#### Step 4: Update Paths in alphapose_game.py

```python
# Edit these lines:
ALPHAPOSE_CFG = "path/to/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
ALPHAPOSE_CHECKPOINT = "path/to/AlphaPose/pretrained_models/fast_res50_256x192.pth"
```

---

## Quick Start

### With AlphaPose Installed
```bash
python alphapose_game.py
```

### With Fallback Mode (MediaPipe)
```bash
# Just run it - will auto-detect and use MediaPipe
python alphapose_game.py
```

---

## Expected Output

### If AlphaPose Found:
```
======================================================================
Multi-Person Body Touch Game - AlphaPose Version
======================================================================

✓ AlphaPose found!
✓ Using device: cuda
✓ AlphaPose model loaded successfully
Camera: 1920×1080
Max players: 2

======================================================================
Game starting! Stand in front of camera...
======================================================================
```

### If Using Fallback:
```
======================================================================
Multi-Person Body Touch Game - AlphaPose Version
======================================================================

✓ AlphaPose found!
⚠ Using simplified AlphaPose initialization
  For full AlphaPose, provide config and checkpoint paths
  Falling back to MediaPipe for pose estimation
✓ Using device: cuda
Camera: 1920×1080
Max players: 2
```

---

## Troubleshooting

### "AlphaPose not installed"

**Solution:**
```bash
pip install alphapose
# Or use fallback mode (auto-enabled)
```

### "CUDA out of memory"

**Solution 1: Reduce batch size**
```python
# In alphapose_game.py, add:
MAX_PLAYERS = 1  # Reduce from 2
```

**Solution 2: Use smaller model**
Download the mobile model instead:
```bash
# Use mobilenet instead of resnet50
# Update ALPHAPOSE_CFG to point to mobile config
```

### "Cannot find config file"

The game will fall back to MediaPipe automatically. To use full AlphaPose:

1. Download AlphaPose from GitHub
2. Update paths in the Python file:
```python
ALPHAPOSE_CFG = "/full/path/to/AlphaPose/configs/..."
ALPHAPOSE_CHECKPOINT = "/full/path/to/fast_res50_256x192.pth"
```

### Low FPS (<10 FPS)

**If using GPU:**
- Check GPU usage: `nvidia-smi`
- Ensure PyTorch uses CUDA: `torch.cuda.is_available()`
- Reduce MAX_PLAYERS to 1

**If using CPU:**
- AlphaPose is very slow on CPU
- Use hybrid.py or jaime_hog.py instead
- Or accept lower FPS (5-10)

---

## Performance Comparison

| Method | GPU FPS | CPU FPS | Accuracy | Setup |
|--------|---------|---------|----------|-------|
| **AlphaPose** | 20-30 | 5-10 | ★★★★★ | Complex |
| **Hybrid (YOLO+MP)** | 25-30 | 13-15 | ★★★★☆ | Easy |
| **HOG+MediaPipe** | N/A | 15-18 | ★★★☆☆ | Easy |
| **MediaPipe only** | N/A | 30-40 | ★★★★☆ | Easy |

**Recommendation:**
- **Have GPU + want best accuracy**: AlphaPose
- **CPU only**: Use hybrid.py or jaime_hog.py instead
- **Quick test**: Run alphapose_game.py in fallback mode

---

## File Structure

If installing full AlphaPose:

```
AlphaPose/
├── configs/
│   └── coco/
│       └── resnet/
│           └── 256x192_res50_lr1e-3_1x.yaml  ← Config
├── pretrained_models/
│   ├── fast_res50_256x192.pth  ← Pose model
│   └── yolov3-spp.weights      ← Detector
└── detector/
    └── yolo/

Your project/
└── alphapose_game.py  ← Update paths to above
```

---

## Advanced Configuration

### Use Different Detector

AlphaPose supports multiple detectors:

```python
# In the config, change detector:
# - yolo (default, good balance)
# - yolox (faster)
# - tracker (best for tracking)
# - efficientdet (lighter)
```

### Adjust Confidence Thresholds

```python
# In alphapose_game.py:
DETECTION_CONFIDENCE = 0.5  # Lower = more detections
POSE_CONFIDENCE = 0.05      # Lower = more visible keypoints
```

### Multi-GPU Support

```python
# AlphaPose supports multi-GPU, but our game doesn't need it
# (only 2 players max)
```

---

## Why AlphaPose?

**Accuracy:**
- State-of-the-art on COCO dataset
- Better than OpenPose in crowded scenes
- Robust to occlusions

**Speed:**
- Real-time with GPU (30+ FPS)
- Optimized inference

**Tracking:**
- Built-in person tracking
- Consistent IDs across frames

**Industry Use:**
- Used in production systems
- Active community
- Well-maintained

---

## Fallback Mode Details

If AlphaPose isn't fully set up, the game automatically uses MediaPipe:

**What you get:**
- ✅ Game works immediately
- ✅ Good pose estimation (33 landmarks)
- ✅ Single-person focus
- ⚠️ Not true AlphaPose performance

**To upgrade to full AlphaPose:**
1. Follow installation steps above
2. Update config paths
3. Re-run the game

---

## Alternative: Use Our Other Versions

If AlphaPose setup is too complex:

| File | Method | FPS | Setup |
|------|--------|-----|-------|
| [hybrid.py](hybrid.py) | YOLO + MediaPipe | 13-15 | `pip install ultralytics mediapipe` |
| [jaime_hog.py](jaime_hog.py) | HOG + MediaPipe | 15-18 | `pip install opencv-python mediapipe` |
| [kristian.py](kristian.py) | MediaPipe Holistic | 25-30 | `pip install mediapipe` |

All provide good multi-person support without complex setup.

---

## Resources

- **AlphaPose GitHub**: https://github.com/MVIG-SJTU/AlphaPose
- **Documentation**: https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md
- **Pre-trained Models**: https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
- **Paper**: https://arxiv.org/abs/1612.00137

---

## Summary

**For Testing:**
```bash
# Just run it - uses MediaPipe fallback
python alphapose_game.py
```

**For Full AlphaPose:**
```bash
# 1. Install AlphaPose
pip install alphapose

# 2. Download models
python scripts/download_models.py

# 3. Update paths in alphapose_game.py
# 4. Run
python alphapose_game.py
```

**If Too Complex:**
Use [hybrid.py](hybrid.py) or [jaime_hog.py](jaime_hog.py) instead - they work great!

Happy gaming! 🎮
