"""
socket_client.py - Run on Mac
Path: ~/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/Hambot/client/socket_client.py

Channel 1 (port 5000): Receive Pi camera stream -> color detection + hand gesture recognition
Channel 2 (port 5001): Send JSON control payload -> Pi state machine

JSON payload sent to Pi:
    {
        "gesture"       : "OPEN" | "CLOSE" | "POINTER" | "OK" | "PEACE" | "UD_OPEN" | "UD_CLOSE" | "NONE",
        "color_x_error" : float (-1.0 ~ 1.0),
        "hand_x_error"  : float (-1.0 ~ 1.0),
        "color_detected": bool,
        "hand_detected" : bool
    }

Gesture actions:
    UP_OPEN    -> FOLLOWING mode
    UP_CLOSE   -> STOP
    PEACE   -> Shutdown client program
    Others  -> passed to Pi for future use

Ref repo (do not modify):
    ~/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/hand-gesture-recognition-mediapipe/

Usage:
    python3 client/socket_client.py --host <Pi IP>
    ex)  python3 client/socket_client.py --host 172.20.10.11
"""

import sys
import os
import json
import time

# hand-gesture-recognition-mediapipe clone path (do not modify)
GESTURE_REPO = '/Users/seyoung/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/hand-gesture-recognition-mediapipe'
sys.path.append(GESTURE_REPO)
os.chdir(GESTURE_REPO)

import socket
import threading
import struct
import argparse
import cv2
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from utils import CvFpsCalc


# ── Argument Parsing ───────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True,
                    help='Pi IP address (ex: 172.20.10.11)')
parser.add_argument('--video_port', type=int, default=5000)
parser.add_argument('--cmd_port',   type=int, default=5001)
parser.add_argument('--min_detection_confidence', type=float, default=0.7)
parser.add_argument('--min_tracking_confidence',  type=float, default=0.5)
args = parser.parse_args()
# ──────────────────────────────────────────────────────


# ── Red Color HSV Range ────────────────────────────────
RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_COLOR_AREA = 3000
# ──────────────────────────────────────────────────────


# ── Gesture Labels (must match keypoint_classifier_label.csv) ──
GESTURE_LABELS = {
    0: "OPEN",
    1: "CLOSE",
    2: "POINTER",
    3: "OK",
}
# ──────────────────────────────────────────────────────


# ── MediaPipe Initialization ───────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
)
keypoint_classifier = KeyPointClassifier()
# ──────────────────────────────────────────────────────


# ── Global State ───────────────────────────────────────
current_payload = {
    "gesture":        "NONE",
    "color_x_error":  0.0,
    "hand_x_error":   0.0,
    "color_detected": False,
    "hand_detected":  False,
}
lock = threading.Lock()
# ──────────────────────────────────────────────────────


def detect_red_color(frame):
    """
    Detect red color in frame using HSV segmentation.
    Returns (color_detected, x_error, bbox).
    x_error normalized to [-1.0, +1.0] relative to frame center.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0.0, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_COLOR_AREA:
        return False, 0.0, None

    x, y, w, h = cv2.boundingRect(largest)
    center_x    = x + w / 2.0
    frame_w     = frame.shape[1]
    x_error     = (center_x / frame_w - 0.5) * 2.0

    return True, round(x_error, 3), (x, y, w, h)


def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to normalized relative coordinate list."""
    h, w = image.shape[:2]
    pts = [[min(int(lm.x * w), w - 1),
             min(int(lm.y * h), h - 1)]
            for lm in landmarks.landmark]

    base_x, base_y = pts[0]
    relative = [[p[0] - base_x, p[1] - base_y] for p in pts]
    flat     = [v for xy in relative for v in xy]
    max_val  = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]


def calc_hand_x_error(frame, landmarks):
    """
    Compute hand center x_error from bounding box of all 21 landmarks.
    Returns float in [-1.0, +1.0].
    """
    frame_w = frame.shape[1]
    x_coords = [lm.x for lm in landmarks.landmark]
    hand_center_x = (min(x_coords) + max(x_coords)) / 2.0
    return round((hand_center_x - 0.5) * 2.0, 3)


# ── Channel 1: Video Receive + Perception (main thread) ──
def video_client():
    global current_payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.video_port))
    print(f"[VIDEO] Connected to Pi ({args.host}:{args.video_port})")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    data_buf  = b""

    try:
        while True:
            fps = cvFpsCalc.get()

            # Receive 4-byte size header
            while len(data_buf) < 4:
                data_buf += sock.recv(4096)
            frame_size = struct.unpack('>I', data_buf[:4])[0]
            data_buf = data_buf[4:]

            # Receive frame body
            while len(data_buf) < frame_size:
                data_buf += sock.recv(65536)
            frame_data = data_buf[:frame_size]
            data_buf   = data_buf[frame_size:]

            # Decode JPEG
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Red color detection
            color_detected, color_x_error, bbox = detect_red_color(frame)

            if color_detected and bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.circle(frame, (bx + bw // 2, by + bh // 2), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"color_err: {color_x_error:.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Hand gesture recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture       = "NONE"
            hand_detected = False
            hand_x_error  = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list  = calc_landmark_list(frame, hand_landmarks)
                hand_sign_id   = keypoint_classifier(landmark_list)
                gesture        = GESTURE_LABELS.get(hand_sign_id, "NONE")
                hand_detected  = True
                hand_x_error   = calc_hand_x_error(frame, hand_landmarks)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(frame, f"Gesture: {gesture}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # PEACE -> shutdown client
            if gesture == "PEACE":
                print("[EXIT] POINTER gesture detected. Shutting down.")
                cv2.destroyAllWindows()
                sock.close()
                sys.exit(0)

            # Update shared payload
            payload = {
                "gesture":        gesture,
                "color_x_error":  color_x_error,
                "hand_x_error":   hand_x_error,
                "color_detected": color_detected,
                "hand_detected":  hand_detected,
            }
            with lock:
                current_payload = payload

            # Overlay UI
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("REU-HumanFollowing | Pi Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()


# ── Channel 2: Command Send (sub thread, 20 Hz) ────────
def command_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.cmd_port))
    print(f"[CMD]   Connected to Pi ({args.host}:{args.cmd_port})")

    try:
        while True:
            with lock:
                payload = current_payload.copy()

            message = json.dumps(payload) + '\n'
            sock.sendall(message.encode('utf-8'))
            time.sleep(0.05)  # 20 Hz

    except (BrokenPipeError, ConnectionResetError):
        print("[CMD] Pi disconnected")
    finally:
        sock.close()


# ── Main ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=== REU-HumanFollowing | Gesture Client Start ===")
    print(f"Pi: {args.host} | video:{args.video_port} | cmd:{args.cmd_port}")
    print("Gestures: UD_OPEN=follow | UD_CLOSE=stop | POINTER=quit")
    print("Press 'q' in video window to quit")

    t_cmd = threading.Thread(target=command_client, daemon=True)
    t_cmd.start()

    video_client()