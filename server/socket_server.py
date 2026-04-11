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

Lidar 우선순위:
    dist < 500mm  → 즉시 정지 (PID 무시)
    dist > 500mm  → PID가 알아서 (후진/전진/속도 조절)

ref repo (no pull/push here):
    ~/Desktop/REU-HumanFollowing/Hambot/  ← only import using sys.path
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

TARGET_DISTANCE  = 500   # mm (3m)
MAX_SPEED        = 75     # 최대 모터 속도 RPM
DIST_STOP        = 500    # 긴급 정지 임계값 (mm)

SPEED_FOLLOWING  = 1.0    # 100%
SPEED_COLOR_ONLY = 0.5    # 50%
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
lateral_pid = PID(Kp=5.0, Ki=0.0, Kd=5.0,   output_limit=MAX_SPEED)
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
            gesture = current_gesture
            c_x_err = color_x_error
            c_det   = color_detected
            h_det   = hand_detected

        # ── State 결정 ──
        state = determine_state(gesture, c_det, h_det)
        with lock:
            current_state = state

        # ── IDLE / STOP → 정지 ──
        if state in (State.IDLE, State.STOP):
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[STATE] {state:12s} | → STOP")
            time.sleep(0.05)
            continue

        # ── Lidar 거리 확인 ──
        dist = get_front_distance()

        # 긴급 정지 (500mm 이하)
        if dist is not None and dist < DIST_STOP:
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[LIDAR] Emergency stop! dist={dist:.0f}mm")
            time.sleep(0.05)
            continue

        # ── State별 속도 비율 ──
        if state == State.COLOR_ONLY:
            speed_ratio = SPEED_COLOR_ONLY
        else:
            speed_ratio = SPEED_FOLLOWING

        # ── 전후 PID (Lidar) ──
        if dist is not None:
            distance_error = dist - TARGET_DISTANCE
            forward_speed  = forward_pid.compute(distance_error) * speed_ratio
        else:
            forward_speed = 0.0
            forward_pid.reset()

        # ── 좌우 PID (색 x_error) ──
        turn_correction = lateral_pid.compute(c_x_err) * speed_ratio

        # ── 최종 모터 속도 ──
        left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_correction))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_correction))

        bot.set_left_motor_speed(left_speed)
        bot.set_right_motor_speed(right_speed)

        print(f"[STATE] {state:12s} | dist={str(round(dist)) if dist else 'None':6s}mm | "
              f"x_err={c_x_err:+.3f} | fwd={forward_speed:+6.1f} | "
              f"turn={turn_correction:+6.1f} | L={left_speed:+6.1f} R={right_speed:+6.1f}")

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