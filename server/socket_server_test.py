"""
socket_server_test.py - Run on Pi (HamBot) [DEBUG ONLY]
Path: ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing/server/socket_server_test.py

모터 제어 없이 모든 값 디버그 출력만 함.
실제 구동은 socket_server.py 사용.

State Machine:
    IDLE       : 색 감지 X                  → 정지 (print only)
    COLOR_ONLY : 색 감지 O + 손 없음         → 50% 속도 (print only)
    FOLLOWING  : 색 감지 O + 손 있음         → 100% 속도 (print only)
    STOP       : CLOSE gesture              → 정지 (print only)
"""
"""
socket_server.py - Run on Pi (HamBot)
Path: ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing/server/socket_server.py

Channel 1 (port 5000): Pi camera → send to Mac
Channel 2 (port 5001): Receive JSON → State Machine + PID control

State Machine:
    IDLE       : 색 감지 X                  → 정지
    COLOR_ONLY : 색 감지 O + 손 없음         → 50% 속도 + PID
    FOLLOWING  : 색 감지 O + 손 있음         → 100% 속도 + PID
    STOP       : CLOSE gesture              → 정지 (색/손 무시)

Lidar 우선순위 (모든 state):
    dist < 500mm          → 즉시 정지
    500mm ~ 1000mm        → 최대 30% 속도
    1000mm ~ 2000mm       → PID 정상
    2000mm 이상            → 전진

ref repo (no pull/push here):
    ~/Desktop/REU-HumanFollowing/Hambot/  ← only import using sys.path
    
Predicable Issue:
1. 시작할 때 색 인식이 관건
카메라 각도랑 거리가 맞아야 빨간 옷이 잡혀. 너무 가까우면 옷이 화면을 꽉 채워서 오히려 노이즈로 튈 수 있어.
2. 1000m 이내에서 시작하면 longitudinal PID가 후진 명령을 냄
"""

import sys
sys.path.append('/home/hambot/Desktop/REU-HumanFollowing/Hambot/src')

import socket
import threading
import struct
import time
import json
import cv2
import numpy as np
from picamera2 import Picamera2
from robot_systems.robot import HamBot


# ── 설정 ──────────────────────────────────────────────
HOST         = '0.0.0.0'
VIDEO_PORT   = 5000
CMD_PORT     = 5001

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60

TARGET_DISTANCE = 1000   # mm (2m)
MAX_SPEED       = 75     # 최대 모터 속도 RPM

# Lidar 거리 임계값 (mm)
DIST_STOP       = 500    # 즉시 정지

# State별 속도 비율
SPEED_FOLLOWING  = 1.0   # 100%
SPEED_COLOR_ONLY = 0.5   # 50%
# ──────────────────────────────────────────────────────


# ── State 정의 ─────────────────────────────────────────
class State:
    IDLE       = "IDLE"
    COLOR_ONLY = "COLOR_ONLY"
    FOLLOWING  = "FOLLOWING"
    STOP       = "STOP"
# ──────────────────────────────────────────────────────

# ── PID 클래스 ─────────────────────────────────────────
class PID:
    def __init__(self, Kp, Ki, Kd, output_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        self._prev_error = 0.0
        self._integral   = 0.0
        self._prev_time  = time.time()

    def compute(self, error):
        now = time.time()
        dt  = max(now - self._prev_time, 1e-6)

        self._integral += error * dt
        derivative      = (error - self._prev_error) / dt
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        if self.output_limit:
            output = max(-self.output_limit, min(self.output_limit, output))

        self._prev_error = error
        self._prev_time  = now
        return output

    def reset(self):
        self._prev_error = 0.0
        self._integral   = 0.0
        self._prev_time  = time.time()
# ──────────────────────────────────────────────────────


# ── HamBot / 카메라 초기화 ─────────────────────────────
bot = HamBot(lidar_enabled=True, camera_enabled=False)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
))
picam2.start()
# ──────────────────────────────────────────────────────


# ── PID 초기화 ─────────────────────────────────────────
lateral_pid = PID(Kp=30.0, Ki=0.0, Kd=5.0,   output_limit=MAX_SPEED)
forward_pid = PID(Kp=0.02, Ki=0.0, Kd=0.005, output_limit=MAX_SPEED)
# ──────────────────────────────────────────────────────


# ── 전역 상태 ──────────────────────────────────────────
current_state   = State.IDLE
color_x_error   = 0.0
color_detected  = False
hand_detected   = False
current_gesture = "NONE"
lock = threading.Lock()
# ──────────────────────────────────────────────────────

# ── 전역 변수 ──────────────────────────────────────────
total_frames        = 0
tracked_frames      = 0
distance_errors     = []
state_transitions   = 0
correct_transitions = 0
prev_gesture        = "NONE"
prev_color_detected = False
prev_hand_detected  = False
# ──────────────────────────────────────────────────────


def get_front_distance():
    """Lidar 정면(175~185도) 최솟값 반환 (mm)"""
    try:
        scan = bot.get_range_image()
        if scan is not None and len(scan) > 0:
            dist = np.min(scan[175:185])
            if dist > 0 and not np.isnan(dist) and not np.isinf(dist):
                return dist
    except Exception as e:
        print(f"[LIDAR] Error: {e}")
    return None


def get_speed_ratio_from_distance(dist):
    if dist is None:
        return 1.0

    if dist < DIST_STOP:
        return None  # 긴급 정지

    return 1.0  # PID에게 맡김


def determine_state(gesture, color_det, hand_det):
    """State 전환 로직"""
    if gesture == "CLOSE":
        return State.STOP

    if not color_det:
        return State.IDLE

    if color_det and not hand_det:
        return State.COLOR_ONLY

    if color_det and hand_det:
        return State.FOLLOWING

    return State.IDLE


def motor_control_loop():
    """State Machine + PID 모터 제어 (20Hz)"""
    global current_state

    print("[MOTOR] Control loop started")

    while True:
        with lock:
            gesture  = current_gesture
            c_x_err  = color_x_error
            c_det    = color_detected
            h_det    = hand_detected

        # ── State 결정 ──
        state = determine_state(gesture, c_det, h_det)

        with lock:
            current_state = state

        print(f"[STATE] {state} | color={c_det} hand={h_det} gesture={gesture}")

        # ── IDLE / STOP → 정지 ──
        if state in (State.IDLE, State.STOP):
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            time.sleep(0.05)
            continue

        # ── COLOR_ONLY / FOLLOWING → PID ──

        # Lidar 거리 확인 (우선순위)
        dist        = get_front_distance()
        speed_ratio = get_speed_ratio_from_distance(dist)

        if speed_ratio is None:
            # Lidar 긴급 정지
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[LIDAR] Emergency stop! dist={dist}mm")
            time.sleep(0.05)
            continue

        # State별 속도 비율 적용
        if state == State.COLOR_ONLY:
            speed_ratio *= SPEED_COLOR_ONLY   # 50%
        elif state == State.FOLLOWING:
            speed_ratio *= SPEED_FOLLOWING    # 100%

        # 전후 PID (Lidar)
        if dist is not None:
            distance_error = dist - TARGET_DISTANCE
            forward_speed  = forward_pid.compute(distance_error) * speed_ratio
        else:
            forward_speed = 0.0
            forward_pid.reset()

        # 좌우 PID (색 x_error)
        turn_correction = lateral_pid.compute(c_x_err) * speed_ratio

        # 최종 모터 속도
        left_speed  = forward_speed - turn_correction
        right_speed = forward_speed + turn_correction
                
        total_frames += 1
        if c_det:
            tracked_frames += 1

        # Mean Distance Error → FOLLOWING일 때만
        if state == State.FOLLOWING and dist is not None:
            distance_errors.append(abs(dist - TARGET_DISTANCE))

        # State Transition Accuracy
        if gesture != prev_gesture or c_det != prev_color_detected or h_det != prev_hand_detected:
            state_transitions += 1
            if state == determine_state(gesture, c_det, h_det):
                correct_transitions += 1
            prev_gesture        = gesture
            prev_color_detected = c_det
            prev_hand_detected  = h_det

        # 출력
        tracking_rate = (tracked_frames / total_frames * 100) if total_frames > 0 else 0
        mean_dist_err = (sum(distance_errors) / len(distance_errors)) if distance_errors else 0
        state_acc     = (correct_transitions / state_transitions * 100) if state_transitions > 0 else 0

        print(f"[METRIC] Track={tracking_rate:.1f}% | DistErr={mean_dist_err:.1f}mm | StateAcc={state_acc:.1f}%")

        # 클램핑
        left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        bot.set_left_motor_speed(left_speed)
        bot.set_right_motor_speed(right_speed)

        print(f"[MOTOR] dist={dist}mm x_err={c_x_err:.2f} L={left_speed:.1f} R={right_speed:.1f} ratio={speed_ratio:.2f}")

        time.sleep(0.05)  # 20Hz


# ── 채널 1: 영상 송신 ─────────────────────────────────
def video_stream_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, VIDEO_PORT))
    server.listen(1)
    print(f"[VIDEO] Waiting... port {VIDEO_PORT}")

    conn, addr = server.accept()
    print(f"[VIDEO] Connected: {addr}")

    try:
        while True:
            frame = picam2.capture_array()
            ret, encoded = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ret:
                continue
            data = encoded.tobytes()
            conn.sendall(struct.pack('>I', len(data)) + data)

    except (BrokenPipeError, ConnectionResetError):
        print("[VIDEO] Mac disconnected")
    finally:
        conn.close()
        server.close()


# ── 채널 2: JSON 명령 수신 ────────────────────────────
def command_server():
    global color_x_error, color_detected, hand_detected, current_gesture

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, CMD_PORT))
    server.listen(1)
    print(f"[CMD]   Waiting... port {CMD_PORT}")

    conn, addr = server.accept()
    print(f"[CMD]   Connected: {addr}")

    buffer = ""
    try:
        while True:
            data = conn.recv(256).decode('utf-8')
            if not data:
                break
            buffer += data

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    payload = json.loads(line)
                    with lock:
                        current_gesture = payload.get('gesture',        'NONE')
                        color_x_error   = payload.get('color_x_error',  0.0)
                        color_detected  = payload.get('color_detected',  False)
                        hand_detected   = payload.get('hand_detected',   False)
                except json.JSONDecodeError:
                    print(f"[CMD] JSON parse error: {line}")

    except ConnectionResetError:
        print("[CMD] Mac disconnected")
    finally:
        with lock:
            color_detected  = False
            hand_detected   = False
            current_gesture = "NONE"
        bot.stop_motors()
        conn.close()
        server.close()


# ── 메인 ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=== REU-HumanFollowing | HamBot Server Start ===")
    print("Run Pi IP on MAC -- Enter host IP")

    t_video = threading.Thread(target=video_stream_server, daemon=True)
    t_cmd   = threading.Thread(target=command_server,      daemon=True)
    t_motor = threading.Thread(target=motor_control_loop,  daemon=True)

    t_video.start()
    t_cmd.start()
    t_motor.start()

    try:
        t_video.join()
        t_cmd.join()
        t_motor.join()
    except KeyboardInterrupt:
        print("\n[STOP] Server shutdown")
        bot.stop_motors()
        picam2.stop()