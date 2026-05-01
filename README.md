# REU-HumanFollowing

**Robust Human-Following Robot Using Multimodal Interaction**  
Color-Based Tracking · Hand Gesture Control · LiDAR Distance Regulation · Finite State Machine

> REU Research · Department of Computer Science · University of South Florida · Spring 2026  
> Author: SY, LL

---

## Overview

This project implements a real-time human-following system on a differential-drive robot (HamBot) using a Raspberry Pi. The robot follows a target person wearing a distinctively colored shirt using HSV color segmentation for lateral alignment and LiDAR for distance regulation. Hand gestures provide high-level state control (follow / stop). The system is designed to be robust to intermittent gesture signal loss through a multimodal fallback state machine.

---

## Architecture

```
Pi (HamBot)                              Mac (Client)
┌──────────────────────┐                ┌─────────────────────────┐
│ Pi Camera            │ -- Ch1 Video ->│ HSV Color Detection     │
│ RPLidar              │                │ MediaPipe Hand Gesture  │
│ FSM + PID Control    │ <- Ch2 JSON  --│ x_error + gesture JSON  │
└──────────────────────┘                └─────────────────────────┘
       port 5000 (video)                        port 5001 (cmd)
```

---

## State Machine

| State | Condition | Behavior |
|---|---|---|
| `IDLE` | No color detected | Stop |
| `COLOR_ONLY` | Color detected, no hand | PID active at 50% speed |
| `FOLLOWING` | Color detected + hand visible | PID active at 100% speed |
| `STOP` | CLOSE gesture | Immediate stop, ignores all inputs |

---

## LiDAR Safety

| Condition | Action |
|---|---|
| Front dist < 500 mm | Emergency stop (overrides PID) |
| Rear dist < 1000 mm | Block reverse only (lateral PID still active) |

---

## Folder Structure

```
REU-HumanFollowing/
├── server/
│   ├── socket_server.py        # Run on Pi (actual motor control)
│   └── socket_server_test.py   # Run on Pi (debug + metric mode, no motors)
├── client/
│   └── socket_client.py        # Run on Mac
└── README.md
```

### Reference Repos (clone separately, do not modify)

```
Pi:  ~/Desktop/REU-HumanFollowing/Hambot/
     -> git clone https://github.com/biorobaw/HamBot.git

Mac: ~/hand-gesture-recognition-mediapipe/
     -> git clone https://github.com/kinivi/hand-gesture-recognition-mediapipe.git
```

---

## Setup

### Pi

```bash
# Clone HamBot library (one time)
cd ~/Desktop/REU-HumanFollowing
git clone https://github.com/biorobaw/HamBot.git Hambot

# Set up virtual environment (one time)
cd Hambot
python3 -m venv --system-site-packages hambot_venv
source hambot_venv/bin/activate
pip install -e .

# Clone this repo
cd ~/Desktop/REU-HumanFollowing
git clone https://github.com/ekdus9968/REU-HumanFollowing.git Controller
```

### Mac

```bash
# Clone gesture recognition repo (one time)
git clone https://github.com/kinivi/hand-gesture-recognition-mediapipe.git

# Install dependencies (one time)
cd hand-gesture-recognition-mediapipe
pip install mediapipe opencv-python tensorflow-macos tensorflow-metal

# Clone this repo
git clone https://github.com/ekdus9968/REU-HumanFollowing.git
```

---

## Demo Video
https://usfedu-my.sharepoint.com/:v:/g/personal/skan_usf_edu/IQCdr5uyBGobSpwhCNhgeNZyAT5W1CWhDGgQ6uPIxfM-iis?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=KcCejw 

---

## Running

### Step 1 — Pi (SSH terminal)

```bash
cd ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing
source ~/Desktop/REU-HumanFollowing/Hambot/hambot_venv/bin/activate

# Actual robot control
python server/socket_server.py

# Debug + metric mode (no motors)
python server/socket_server_test.py
```

Wait for:
```
[VIDEO] Waiting... port 5000
[CMD]   Waiting... port 5001
```

### Step 2 — Mac (new terminal)

```bash
cd /Users/<your_username>/Documents/.../REU-HumanFollowing/Hambot
python3 client/socket_client.py --host <Pi_IP>
```

Pi IP is shown on the HamBot OLED display.

Press `q` in the video window to quit.

---

## Gesture Control

| Gesture | Action |
|---|---|
| `OPEN` | Following mode (PID active) |
| `CLOSE` | Stop immediately |

Custom gesture dataset trained with MediaPipe keypoint classifier:  
**5 classes** — Open · Close · Pointer · Open Upside Down · Close Upside Down

---

## PID Parameters

| Controller | Kp | Ki | Kd | Input | Output |
|---|---|---|---|---|---|
| Lateral | 15.0 | 0.0 | 3.0 | color x_error | turn correction |
| Longitudinal | 0.02 | 0.0 | 0.005 | LiDAR distance error | forward speed |

Target distance: **500 mm**  
Max motor speed: **75 RPM**

---

## Evaluation Metrics (socket_server_test.py)

| Metric | Description |
|---|---|
| Tracking Success Rate | % of frames where color is detected |
| Mean Distance Error | Mean \|dist - 500mm\| in FOLLOWING state only |
| State Transition Accuracy | % of correct state transitions on input change |

Metrics are printed every 2 seconds and summarized on Ctrl+C.

---

## Hardware

| Component | Spec |
|---|---|
| Compute | Raspberry Pi + Build HAT |
| Camera | Pi Camera Board v2 (8 MP) |
| LiDAR | Slamtec RPLidar (360°, planar) |
| Motors | LEGO Technic Large Angular |
| Wheel Diameter | 90 mm |
| Axle Length | 184 mm |
| Max Speed | 75 RPM |

---

## License

MIT License
