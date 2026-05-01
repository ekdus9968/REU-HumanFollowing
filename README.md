# REU-HumanFollowing

**Robust Human-Following Robot Using Multimodal Interaction**  
Color-Based Tracking · Hand Gesture Control · LiDAR Distance Regulation · Finite State Machine

> REU Research · Department of Computer Science · University of South Florida · Spring 2026  
> Authors: Seyoung Kan, Laray Lopez  
> Faculty Mentor: Prof. Alfredo Weitzenfeld · Graduate Mentor: Zachary Hinnen

---

## Overview

This project implements a real-time human-following system on a differential-drive robot (HamBot) using a Raspberry Pi. The robot follows a target person wearing a distinctively colored shirt using HSV color segmentation for lateral alignment and LiDAR for distance regulation. Hand gestures provide high-level state control (follow / turn / stop). The system is designed to be robust to intermittent signal loss through a multimodal fallback state machine with autonomous target recovery.

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
| `IDLE` | color X + hand X, target never found | Stop — waits for initial acquisition |
| `FOLLOWING` | Color O + hand O + OPEN gesture | PID active at 100% speed |
| `COLOR_ONLY` | Color O + no hand | PID active at 70% speed |
| `HAND_ONLY` | Color X + hand O | PID active at 20% speed (hand x_error) |
| `REDETECT` | Color X + hand X, 10+ frames after target found | Spins in last known direction to recover |
| `STOP` | CLOSE gesture (5 consecutive frames) | Immediate stop, ignores all inputs |

**Key design:** REDETECT enables autonomous recovery — when both color and gesture are lost for 500ms, the robot spins toward the last known target direction until color is reacquired.

---

## Folder Structure

```
REU-HumanFollowing/
├── server/
│   ├── socket_server.py        # Run on Pi (actual motor control)
│   └── socket_server_test.py   # Run on Pi (test mode - no STOP state)
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

## Running

### Step 1 — Pi (SSH terminal)

```bash
cd ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing
source ~/Desktop/REU-HumanFollowing/Hambot/hambot_venv/bin/activate

# Actual robot control
python server/socket_server.py

# Test mode (gesture ignored, no STOP state)
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

| Gesture | Action | Upside Down |
|---|---|
| `OPEN` | Activate following mode (PID active) | YES
| `CLOSE` | Stop immediately (5 consecutive frames required) | YES
| `POINTER` | | NO
| `OKAY` | | NO
| `PEACE` | End the program | NO

Gesture recognition uses the original MediaPipe keypoint classifier from the reference repo:  
**4 classes** — Open · Close · Pointer · OK

We retrained gestures using new data. 

---

## PID Parameters

| Controller | Kp | Ki | Kd | Input | Output |
|---|---|---|---|---|---|
| Lateral | 5.0 | 0.0 | 1.0 | color x_error (bounding box center) | turn correction |
| Longitudinal | 0.02 | 0.0 | 0.005 | LiDAR distance error | forward speed |

Target distance: **500 mm**  
Max motor speed: **75 RPM**  
REDETECT spin speed: **2 RPM**

---

## Evaluation Metrics (socket_server_test.py)

| Metric | Description |
|---|---|
| Tracking Success Rate | 93.7% of frames where color is detected |
| Mean Distance Error | Mean \|dist - 500mm\| in FOLLOWING state only |
| State Transition Accuracy | 95.2% of correct state transitions on input change |

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

## Demo Video

[Watch Demo](https://usfedu-my.sharepoint.com/:v:/g/personal/skan_usf_edu/IQCdr5uyBGobSpwhCNhgeNZyAT5W1CWhDGgQ6uPIxfM-iis?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=KcCejw)

---

## License

MIT License
