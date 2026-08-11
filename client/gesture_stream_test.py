"""
gesture_stream_test.py - Run on Mac [GESTURE ACCURACY over Pi camera stream]
Path: ~/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/Hambot/client/gesture_stream_test.py

Based directly on socket_client.py. Measures gesture classification accuracy
under REAL operating conditions: the Pi camera streams frames over the socket,
and the model runs on the Mac. Only the video channel is used (no command sent).

Usage:
    python3 client/gesture_stream_test.py --host <Pi IP> --label OPEN --frames 100
    ex)  python3 client/gesture_stream_test.py --host 172.20.10.11 --label UD_OPEN --frames 100

Only frames where a hand is detected are counted toward --frames.
Auto-stops when the target count is reached.

Output (logs/ folder):
    gesture_stream_<label>_raw.csv     : every counted frame (predicted, correct)
    gesture_stream_<label>_summary.csv : accuracy + misclassification breakdown
"""

import sys
import os
import csv
import time

# hand-gesture-recognition-mediapipe clone path (do not modify)
GESTURE_REPO = '/Users/seyoung/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/hand-gesture-recognition-mediapipe'
sys.path.append(GESTURE_REPO)
os.chdir(GESTURE_REPO)

import socket
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
parser.add_argument('--label', type=str, required=True,
                    help='Ground-truth gesture being shown (OPEN/CLOSE/POINTER/OK/PEACE/UD_OPEN/UD_CLOSE)')
parser.add_argument('--frames', type=int, default=100,
                    help='Number of hand-detected frames to collect before stopping')
parser.add_argument('--min_detection_confidence', type=float, default=0.7)
parser.add_argument('--min_tracking_confidence',  type=float, default=0.5)
args = parser.parse_args()
# ──────────────────────────────────────────────────────


# ── Gesture Labels (must match keypoint_classifier_label.csv) ──
GESTURE_LABELS = {
    0: "OPEN",
    1: "CLOSE",
    2: "POINTER",
    3: "OK",
    4: "PEACE",
    5: "UD_OPEN",
    6: "UD_CLOSE",
}
LABEL_TO_ID = {v: k for k, v in GESTURE_LABELS.items()}

GROUND_TRUTH = args.label.upper()
if GROUND_TRUTH not in LABEL_TO_ID:
    print(f"[ERROR] --label must be one of: {list(LABEL_TO_ID.keys())}")
    sys.exit(1)
# ──────────────────────────────────────────────────────


# ── Output paths ───────────────────────────────────────
LOG_DIR = os.path.join(
    '/Users/seyoung/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/Hambot/client',
    "logs"
)
os.makedirs(LOG_DIR, exist_ok=True)
RAW_PATH     = os.path.join(LOG_DIR, f"gesture_stream_{GROUND_TRUTH}_raw.csv")
SUMMARY_PATH = os.path.join(LOG_DIR, f"gesture_stream_{GROUND_TRUTH}_summary.csv")
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


# ── Channel 1: Video Receive + Gesture Eval (main thread) ──
def video_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.video_port))
    print(f"[VIDEO] Connected to Pi ({args.host}:{args.video_port})")
    print(f"[TEST]  Ground truth: {GROUND_TRUTH} | Target frames: {args.frames}")
    print("Hold the gesture from different positions. Auto-stops when target reached.")
    print("Press 'q' to stop early.\n")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    data_buf  = b""

    collected   = 0
    correct     = 0
    pred_counts = {name: 0 for name in GESTURE_LABELS.values()}
    samples     = []  # (frame_index, predicted, correct)

    try:
        while collected < args.frames:
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

            # Hand gesture recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            pred_name = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list  = calc_landmark_list(frame, hand_landmarks)
                hand_sign_id   = keypoint_classifier(landmark_list)
                pred_name      = GESTURE_LABELS.get(hand_sign_id, "NONE")

                # Count this frame
                collected += 1
                is_correct = (pred_name == GROUND_TRUTH)
                if is_correct:
                    correct += 1
                if pred_name in pred_counts:
                    pred_counts[pred_name] += 1
                samples.append((collected, pred_name, is_correct))

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Live overlay
            acc_so_far = (correct / collected * 100) if collected else 0
            color = (0, 255, 0) if pred_name == GROUND_TRUTH else (0, 0, 255)
            cv2.putText(frame, f"GT: {GROUND_TRUTH}  Pred: {pred_name if pred_name else '--'}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"{collected}/{args.frames}  acc={acc_so_far:.1f}%",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Stream Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()

    # ── Save results ──
    with open(RAW_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "ground_truth", "predicted", "correct"])
        for idx, pred, ok in samples:
            wr.writerow([idx, GROUND_TRUTH, pred, ok])

    accuracy = (correct / collected * 100) if collected else 0
    with open(SUMMARY_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "value"])
        wr.writerow(["ground_truth", GROUND_TRUTH])
        wr.writerow(["total_frames", collected])
        wr.writerow(["correct", correct])
        wr.writerow(["accuracy_pct", f"{accuracy:.2f}"])
        wr.writerow([])
        wr.writerow(["predicted_as", "count"])
        for name, cnt in pred_counts.items():
            wr.writerow([name, cnt])

    print(f"\n{'='*55}")
    print(f"[DONE] Ground truth: {GROUND_TRUTH}")
    print(f"  Frames counted : {collected}")
    print(f"  Correct        : {correct}")
    print(f"  Accuracy       : {accuracy:.1f}%")
    misclass = ", ".join(f'{k}:{v}' for k, v in pred_counts.items() if k != GROUND_TRUTH and v > 0)
    print(f"  Misclassified  : {{{misclass}}}")
    print(f"  Raw:     {RAW_PATH}")
    print(f"  Summary: {SUMMARY_PATH}")
    print(f"{'='*55}")


# ── Main ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=== REU-HumanFollowing | Gesture Stream Accuracy Test ===")
    print(f"Pi: {args.host} | video:{args.video_port}")
    video_client()
