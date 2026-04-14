"""
socket_server.py - Run on Pi (HamBot)
Path: ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing/server/socket_server.py

Channel 1 (port 5000): Pi camera → send to Mac
Channel 2 (port 5001): Receive JSON → State Machine + PID control

State Machine:
    IDLE       : No color detected               → Stop
    COLOR_ONLY : Color detected, no hand         → 50% speed + PID
    FOLLOWING  : Color detected + hand visible   → 100% speed + PID
    STOP       : CLOSE gesture                   → Stop (ignore color/hand)

LiDAR priority:
    Front dist < 500mm   → Emergency stop
    Rear  dist < 1000mm  → Block reverse (lateral PID still active)
    Otherwise            → PID handles normally

ref repo (no pull/push here):
    ~/Desktop/REU-HumanFollowing/Hambot/  <- only import using sys.path
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


# ── Configuration ─────────────────────────────────────
HOST         = '0.0.0.0'
VIDEO_PORT   = 5000
CMD_PORT     = 5001

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60

TARGET_DISTANCE  = 500    # mm (0.5m)
MAX_SPEED        = 75     # max motor speed (RPM)
DIST_FRONT_STOP  = 500    # front emergency stop threshold (mm)
DIST_REAR_STOP   = 1000   # rear obstacle reverse block threshold (mm)

SPEED_FOLLOWING  = 1.0    # 100%
SPEED_COLOR_ONLY = 1.5    # 50%
# ──────────────────────────────────────────────────────


# ── State Definition ───────────────────────────────────
class State:
    IDLE       = "IDLE"
    COLOR_ONLY = "COLOR_ONLY"
    FOLLOWING  = "FOLLOWING"
    STOP       = "STOP"
# ──────────────────────────────────────────────────────


# ── PID Controller ─────────────────────────────────────
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


# ── HamBot / Camera Initialization ────────────────────
bot = HamBot(lidar_enabled=True, camera_enabled=False)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
))
picam2.start()
# ──────────────────────────────────────────────────────


# ── PID Initialization ─────────────────────────────────
lateral_pid = PID(Kp=5.0, Ki=0.0, Kd=1.0,   output_limit=MAX_SPEED)
forward_pid = PID(Kp=0.02, Ki=0.0, Kd=0.005, output_limit=MAX_SPEED)
# ──────────────────────────────────────────────────────


# ── Global State ───────────────────────────────────────
current_state   = State.IDLE
color_x_error   = 0.0
color_detected  = False
hand_detected   = False
current_gesture = "NONE"
lock = threading.Lock()
# ──────────────────────────────────────────────────────


def get_front_distance():
    """Return minimum LiDAR distance in frontal arc (175-185 deg) in mm."""
    try:
        scan = bot.get_range_image()
        if scan is not None and len(scan) > 0:
            dist = np.min(scan[175:185])
            if dist > 0 and not np.isnan(dist) and not np.isinf(dist):
                return dist, scan
    except Exception as e:
        print(f"[LIDAR] Error: {e}")
    return None, None


def get_rear_distance(scan):
    """Return minimum LiDAR distance in rear arc (355-5 deg) in mm."""
    try:
        if scan is not None and len(scan) > 0:
            rear  = np.concatenate([scan[355:360], scan[0:5]])
            valid = rear[rear > 0]
            if len(valid) > 0:
                dist = np.min(valid)
                if not np.isnan(dist) and not np.isinf(dist):
                    return dist
    except Exception as e:
        print(f"[LIDAR REAR] Error: {e}")
    return None


def determine_state(gesture, color_det, hand_det):
    """State transition logic based on gesture, color, and hand detection."""
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
    """State machine + PID motor control loop at 20Hz."""
    global current_state

    print("[MOTOR] Control loop started")

    while True:
        with lock:
            gesture = current_gesture
            c_x_err = color_x_error
            c_det   = color_detected
            h_det   = hand_detected

        # ── Determine current state ──
        state = determine_state(gesture, c_det, h_det)
        with lock:
            current_state = state

        # ── IDLE / STOP → halt motors ──
        if state in (State.IDLE, State.STOP):
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[STATE] {state:12s} | -> STOP")
            time.sleep(0.05)
            continue

        # ── Read LiDAR distances ──
        dist, scan = get_front_distance()
        rear_dist  = get_rear_distance(scan)

        # Front emergency stop
        if dist is not None and dist < DIST_FRONT_STOP:
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[LIDAR] Front emergency stop! dist={dist:.0f}mm")
            time.sleep(0.05)
            continue

        # ── Speed ratio by state ──
        speed_ratio = SPEED_COLOR_ONLY if state == State.COLOR_ONLY else SPEED_FOLLOWING

        # ── Longitudinal PID (LiDAR) ──
        if dist is not None:
            distance_error = dist - TARGET_DISTANCE
            forward_speed  = forward_pid.compute(distance_error) * speed_ratio
        else:
            forward_speed = 0.0
            forward_pid.reset()

        # ── Rear obstacle → block reverse, keep lateral PID ──
        rear_blocked = False
        if rear_dist is not None and rear_dist < DIST_REAR_STOP:
            if forward_speed < 0:  # only block when reverse command issued
                forward_speed = 0.0
                rear_blocked  = True

        # ── Lateral PID (color x_error) ──
        turn_correction = lateral_pid.compute(c_x_err) * speed_ratio

        # ── Final motor speeds ──
        left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_correction))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_correction))

        bot.set_left_motor_speed(left_speed)
        bot.set_right_motor_speed(right_speed)

        rear_info = f"rear={rear_dist:.0f}mm {'BLOCKED' if rear_blocked else 'OK'}" if rear_dist else "rear=None"
        print(f"[STATE] {state:12s} | dist={str(round(dist)) if dist else 'None':6s}mm | {rear_info} | "
              f"x_err={c_x_err:+.3f} | fwd={forward_speed:+6.1f} | "
              f"turn={turn_correction:+6.1f} | L={left_speed:+6.1f} R={right_speed:+6.1f}")

        time.sleep(0.05)  # 20Hz


# ── Channel 1: Video Stream ────────────────────────────
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


# ── Channel 2: Command Receive ────────────────────────
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


# ── Main ───────────────────────────────────────────────
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