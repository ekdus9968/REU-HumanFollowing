"""
socket_client.py - Run on Mac
path : ~/REU-HumanFollowing/client/socket_client.py

Channel 1 (port 5000): Pi camera send video → detect gesture
Channel 2 (port 5001): gesture command (OPEN/CLOSE/POINTER/OK) → recieved Pi 
 
ref repo (no push/pull):
    ~/hand-gesture-recognition-mediapipe/  ← import -> sys.path
 
How to USe:
    python socket_client.py --host <Pi's IP>
    ex) python socket_client.py --host 192.168.1.42
"""

import sys
import os

# hand-gesture-recognition-mediapipe clone path(no change)
GESTURE_REPO = os.path.expanduser('~/hand-gesture-recognition-mediapipe')
sys.path.append(GESTURE_REPO)

import socket
import threading
import struct
import argparse
import time
import cv2
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from utils import CvFpsCalc


# =====================
#       SETTING         
# =====================

PI_IP = '10.42.0.1'


# =====================
#       PARSER         
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True, help=PI_IP)
parser.add_argument('--viedo_port', type=int, default=5000)
parser.add_argument('--cmd_port', type=int, default=5001)
parser.add_argument('--min_detection_confidence', type=float, default=0.7)
parser.add_argument('--min_tracking_confidence',  type=float, default=0.5)
args = parser.parse_args()


# ==========================
#       GESTURE LABEL         
# ==========================

GESTURE_LABELS = {
    0 : "OPEN",
    1 : "CLOSE",
    2 : "POINTER",
    3 : "OK",
}

# ===========================
#       MEDIAPIPE RESET         
# ===========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
)
keypoint_classifier = KeyPointClassifier()


# =======================
#       PARMETER        
# =======================

current_gesture   = None
last_sent_gesture = None
lock = threading.Lock()



# =======================
#       FUNCTION        
# =======================

def calc_landmark_list(image, landmarks):
    """MediaPipe landmark → normalizated x,y axis"""
    h, w = image.shape[:2]
    pts = [[min(int(lm.x * w), w - 1),
             min(int(lm.y * h), h - 1)]
            for lm in landmarks.landmark]
 
    base_x, base_y = pts[0]
    relative = [[p[0] - base_x, p[1] - base_y] for p in pts]
    flat = [v for xy in relative for v in xy]
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]
 


# =======================================================================
#       CHANNEL 1: SEND VIDEO + RECOGNISED GESTURE (MAIN THREADS)        
# =======================================================================
def video_client():
    global current_gesture
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.video_port))
    print(f"[VIDEO] Pi CONNECTED ({args.host}:{args.video_port})")
 
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
 
            # MediaPipe 추론
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
 
            gesture_label = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list  = calc_landmark_list(frame, hand_landmarks)
                hand_sign_id   = keypoint_classifier(landmark_list)
                gesture_label  = GESTURE_LABELS.get(hand_sign_id, "UNKNOWN")
 
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(frame, f"Gesture: {gesture_label}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2)
 
            with lock:
                current_gesture = gesture_label
 
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)
 
            cv2.imshow("REU-HumanFollowing | Pi Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
    finally:
        sock.close()
        cv2.destroyAllWindows()
 
 

# =======================================================================
#       CHANNEL 2: RECEIVED COMMAND (SERVE THREADS)        
# =======================================================================
def command_client():
    global last_sent_gesture
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.cmd_port))
    print(f"[CMD]   Pi CONNECTED ({args.host}:{args.cmd_port})")
 
    try:
        while True:
            with lock:
                gesture = current_gesture
 
            # 변경 있을 때만 전송 (중복 방지)
            if gesture and gesture != last_sent_gesture:
                sock.sendall((gesture + '\n').encode('utf-8'))
                print(f"[CMD]   SENDED: {gesture}")
                last_sent_gesture = gesture
 
            time.sleep(0.1)  # 100ms 주기
 
    except (BrokenPipeError, ConnectionResetError):
        print("[CMD] Pi DISCONNECTED")
    finally:
        sock.close()
 
 

# =================
#       MAIN      
# =================
if __name__ == '__main__':
    print("=== REU-HumanFollowing | Gesture Client 시작 ===")
    print(f"Pi: {args.host} | video:{args.video_port} | cmd:{args.cmd_port}")
    print("END: ENTER 'q' Key")
 
    # 명령 송신은 서브 스레드
    t_cmd = threading.Thread(target=command_client, daemon=True)
    t_cmd.start()
 
    # 영상 수신 + 제스처 인식은 메인 스레드 (OpenCV 창 요구사항)
    video_client()