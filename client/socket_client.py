"""
socket_client.py - Run on Mac
Path: ~/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/Hambot/client/socket_client.py

Channel 1 (port 5000): Receive Pi camera → Color detection + Hand gesture
Channel 2 (port 5001): Send JSON → Pi state machine

State Machine:
    IDLE         : 색 감지 X → 정지
    COLOR_ONLY   : 색 감지 O + 손 없음 → 50% 속도
    FOLLOWING    : 색 감지 O + 손 있음 → 100% 속도
    STOP         : CLOSE gesture → 정지

JSON:
    {
        "gesture"       : "OPEN" | "CLOSE" | "NONE",
        "color_x_error" : float (-1.0 ~ 1.0),
        "color_detected": bool,
        "hand_detected" : bool
    }

ref repo (no pull/push here):
    ~/Documents/USF/CLASS/Spring2026/CIS4915/REU-HumanFollowing/hand-gesture-recognition-mediapipe/

Usage:
    python3 client/socket_client.py --host <Pi IP>
    ex) python3 client/socket_client.py --host 172.20.10.11
"""

import sys
import os
import json
import time

# hand-gesture-recognition-mediapipe clone 경로 (건드리지 않음)
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


# ── 인자 파싱 ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True,
                    help='Pi IP address (ex: 172.20.10.11)')
parser.add_argument('--video_port', type=int, default=5000)
parser.add_argument('--cmd_port',   type=int, default=5001)
parser.add_argument('--min_detection_confidence', type=float, default=0.7)
parser.add_argument('--min_tracking_confidence',  type=float, default=0.5)
args = parser.parse_args()
# ──────────────────────────────────────────────────────


# ── 빨간색 HSV 범위 ────────────────────────────────────
# 빨강은 HSV에서 두 범위로 나뉨 (0~10, 170~180)
RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_COLOR_AREA = 3000  # 최소 감지 면적 (노이즈 제거)
# ──────────────────────────────────────────────────────


# ── Gesture 라벨 ───────────────────────────────────────
GESTURE_LABELS = {
    0: "OPEN",
    1: "CLOSE",
    2: "POINTER",
    3: "OK",
}
# ──────────────────────────────────────────────────────


# ── MediaPipe 초기화 ───────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
)
keypoint_classifier = KeyPointClassifier()
# ──────────────────────────────────────────────────────


# ── 전역 상태 ──────────────────────────────────────────
current_payload = {
    "gesture":        "NONE",
    "color_x_error":  0.0,
    "color_detected": False,
    "hand_detected":  False,
}
lock = threading.Lock()
# ──────────────────────────────────────────────────────


def detect_red_color(frame):
    """
    빨간색 감지 → 가장 큰 컨투어의 바운딩박스 중심 x_error 반환
    Returns:
        (color_detected, x_error, bbox)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 두 범위 마스크 합치기
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0.0, None

    # 가장 큰 컨투어 선택
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_COLOR_AREA:
        return False, 0.0, None

    x, y, w, h = cv2.boundingRect(largest)
    center_x = x + w / 2.0

    # 화면 중앙(0.5) 기준 정규화 (-1.0 ~ 1.0)
    frame_w = frame.shape[1]
    x_error = (center_x / frame_w - 0.5) * 2.0

    return True, round(x_error, 3), (x, y, w, h)


def calc_landmark_list(image, landmarks):
    """MediaPipe landmark → 정규화된 좌표 리스트 (KeyPointClassifier 입력용)"""
    h, w = image.shape[:2]
    pts = [[min(int(lm.x * w), w - 1),
             min(int(lm.y * h), h - 1)]
            for lm in landmarks.landmark]

    base_x, base_y = pts[0]
    relative = [[p[0] - base_x, p[1] - base_y] for p in pts]
    flat = [v for xy in relative for v in xy]
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]


# ── 채널 1: 영상 수신 + 감지 (메인 스레드) ───────────────
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

            # 4바이트 크기 헤더 수신
            while len(data_buf) < 4:
                data_buf += sock.recv(4096)
            frame_size = struct.unpack('>I', data_buf[:4])[0]
            data_buf = data_buf[4:]

            # 프레임 본문 수신
            while len(data_buf) < frame_size:
                data_buf += sock.recv(65536)
            frame_data = data_buf[:frame_size]
            data_buf   = data_buf[frame_size:]

            # JPEG 디코딩
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # ── 빨간색 감지 ──
            color_detected, color_x_error, bbox = detect_red_color(frame)

            if color_detected and bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.circle(frame, (bx + bw // 2, by + bh // 2), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"color_err: {color_x_error:.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ── 손 제스처 감지 ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture       = "NONE"
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list  = calc_landmark_list(frame, hand_landmarks)
                hand_sign_id   = keypoint_classifier(landmark_list)
                gesture        = GESTURE_LABELS.get(hand_sign_id, "NONE")
                hand_detected  = True

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(frame, f"Gesture: {gesture}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ── payload 업데이트 ──
            payload = {
                "gesture":        gesture,
                "color_x_error":  color_x_error,
                "color_detected": color_detected,
                "hand_detected":  hand_detected,
            }
            with lock:
                current_payload = payload

            # ── 화면 표시 ──
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("REU-HumanFollowing | Pi Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()


# ── 채널 2: JSON 송신 (서브 스레드, 20Hz) ─────────────────
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

            time.sleep(0.05)  # 20Hz

    except (BrokenPipeError, ConnectionResetError):
        print("[CMD] Pi disconnected")
    finally:
        sock.close()


# ── 메인 ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=== REU-HumanFollowing | Gesture Client Start ===")
    print(f"Pi: {args.host} | video:{args.video_port} | cmd:{args.cmd_port}")
    print("END: ENTER 'q' Key")

    t_cmd = threading.Thread(target=command_client, daemon=True)
    t_cmd.start()

    video_client()