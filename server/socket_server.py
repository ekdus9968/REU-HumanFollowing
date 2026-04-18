"""
socket_server.py - Run on Pi (HamBot)
Path: ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing/server/socket_server.py

Channel 1 (port 5000): Pi camera -> send to Mac
Channel 2 (port 5001): Receive JSON -> State Machine + PID control

State Machine:
    IDLE       : Initial state, color+UD_OPEN not yet detected -> Stop
    FOLLOWING  : color O + hand O + gesture UD_OPEN            -> 100% speed PID
    COLOR_ONLY : color O + (no hand or non-UD_OPEN gesture)    -> 70% speed PID
    HAND_ONLY  : color X + hand O                              -> 20% speed PID (hand x_error)
    REDETECT   : color X + hand X, lost for 10+ frames         -> spin speed=2
    STOP       : UD_CLOSE gesture                              -> Stop (overrides all)

Ref repo (do not modify):
    ~/Desktop/REU-HumanFollowing/Hambot/  <- import via sys.path only
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


# ── Configuration ──────────────────────────────────────
HOST         = '0.0.0.0'
VIDEO_PORT   = 5000
CMD_PORT     = 5001

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60

TARGET_DISTANCE      = 500   # mm (0.5m)
MAX_SPEED            = 10    # max motor speed (RPM)
SPIN_SPEED           = 2     # REDETECT spin speed (RPM)
COLOR_LOST_THRESHOLD = 10    # frames before entering REDETECT

SPEED_FOLLOWING  = 1.0       # 100%
SPEED_COLOR_ONLY = 0.7       # 70%
SPEED_HAND_ONLY  = 0.2       # 20%
# ──────────────────────────────────────────────────────


# ── State Definition ───────────────────────────────────
class State:
    IDLE       = "IDLE"
    FOLLOWING  = "FOLLOWING"
    COLOR_ONLY = "COLOR_ONLY"
    HAND_ONLY  = "HAND_ONLY"
    REDETECT   = "REDETECT"
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
lateral_pid = PID(Kp=5.0,  Ki=0.0, Kd=1.0,   output_limit=MAX_SPEED)
forward_pid = PID(Kp=0.02, Ki=0.0, Kd=0.005, output_limit=MAX_SPEED)
# ──────────────────────────────────────────────────────


# ── Global State ───────────────────────────────────────
current_state     = State.IDLE
color_x_error     = 0.0
hand_x_error      = 0.0
color_detected    = False
hand_detected     = False
current_gesture   = "NONE"
last_color_x_err  = 0.0
target_ever_found = False  # True only after color + UD_OPEN detected together
color_lost_count  = 0      # frames color has been lost consecutively
stop_gesture_count = 0
lock = threading.Lock()
# ──────────────────────────────────────────────────────


def get_front_distance():
    """Return minimum LiDAR distance in front arc (175~185 deg) in mm."""
    try:
        scan = bot.get_range_image()
        if scan is not None and len(scan) > 0:
            dist = np.min(scan[175:185])
            if dist > 0 and not np.isnan(dist) and not np.isinf(dist):
                return dist
    except Exception as e:
        print(f"[LIDAR] Error: {e}")
    return None


def determine_state(gesture, color_det, hand_det, target_found):
    """State transition logic."""
    # STOP always takes priority
    if gesture == "UD_CLOSE":
        return State.STOP

    # Before target is found: stay IDLE no matter what
    if not target_found:
        return State.IDLE

    # After target found: allow state transitions
    if color_det and hand_det and gesture == "UD_OPEN":
        return State.FOLLOWING
    if color_det:
        return State.COLOR_ONLY
    if not color_det and hand_det:
        return State.HAND_ONLY
    if color_lost_count >= COLOR_LOST_THRESHOLD:
        return State.REDETECT
    return State.IDLE


ddef motor_control_loop():
    """State machine + PID motor control loop at 20 Hz."""
    global current_state, last_color_x_err, target_ever_found, color_lost_count, stop_gesture_count

    print("[MOTOR] Control loop started")

    while True:
        with lock:
            gesture = current_gesture
            c_x_err = color_x_error
            h_x_err = hand_x_error
            c_det   = color_detected
            h_det   = hand_detected
            t_found = target_ever_found
            last_x  = last_color_x_err

        # UD_CLOSE debounce - 5 consecutive frames to trigger STOP
        if gesture == "UD_CLOSE":
            stop_gesture_count += 1
        else:
            stop_gesture_count = 0
        filtered_gesture = "UD_CLOSE" if stop_gesture_count >= 5 else gesture

        # Set target_ever_found only when color + UD_OPEN detected together
        if c_det and gesture == "UD_OPEN":
            target_ever_found = True

        # Update color lost count and last known direction
        if c_det:
            last_color_x_err = c_x_err
            color_lost_count = 0
        else:
            color_lost_count += 1

        # Determine state using filtered gesture
        state = determine_state(filtered_gesture, c_det, h_det, t_found)
        with lock:
            current_state = state

        # ── IDLE / STOP -> stop ────────────────────────
        if state in (State.IDLE, State.STOP):
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()
            print(f"[STATE] {state:12s} | -> STOP")
            time.sleep(0.05)
            continue

        # ── REDETECT -> spin ───────────────────────────
        if state == State.REDETECT:
            lateral_pid.reset()
            forward_pid.reset()

            if c_det:
                bot.stop_motors()
                time.sleep(0.05)
                continue

            if last_x >= 0:
                left_speed  =  SPIN_SPEED
                right_speed = -SPIN_SPEED
            else:
                left_speed  = -SPIN_SPEED
                right_speed =  SPIN_SPEED

            bot.set_left_motor_speed(left_speed)
            bot.set_right_motor_speed(right_speed)
            print(f"[STATE] REDETECT     | spinning {'RIGHT' if last_x >= 0 else 'LEFT'} speed={SPIN_SPEED}")
            time.sleep(0.05)
            continue

        # ── Read LiDAR ────────────────────────────────
        dist = get_front_distance()

        # ── Speed ratio and lateral error by state ─────
        if state == State.FOLLOWING:
            speed_ratio = SPEED_FOLLOWING
            lateral_err = c_x_err
        elif state == State.COLOR_ONLY:
            speed_ratio = SPEED_COLOR_ONLY
            lateral_err = c_x_err
        elif state == State.HAND_ONLY:
            speed_ratio = SPEED_HAND_ONLY
            lateral_err = h_x_err
        else:
            speed_ratio = 0.0
            lateral_err = 0.0

        # ── Longitudinal PID (LiDAR) ───────────────────
        if dist is not None:
            distance_error = dist - TARGET_DISTANCE
            forward_speed  = forward_pid.compute(distance_error) * speed_ratio
        else:
            forward_speed = 0.0
            forward_pid.reset()

        # ── Lateral PID ────────────────────────────────
        turn_correction = lateral_pid.compute(lateral_err) * speed_ratio

        # ── Final motor speeds ─────────────────────────
        left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_correction))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_correction))

        bot.set_left_motor_speed(left_speed)
        bot.set_right_motor_speed(right_speed)

        print(f"[STATE] {state:12s} | dist={str(round(dist)) if dist else 'None':6s}mm | "
              f"lat_err={lateral_err:+.3f} | fwd={forward_speed:+6.1f} | "
              f"turn={turn_correction:+6.1f} | L={left_speed:+6.1f} R={right_speed:+6.1f} | "
              f"ratio={speed_ratio:.1f}")

        time.sleep(0.05)  # 20 Hz


# ── Channel 1: Video Stream (Pi -> Mac) ───────────────
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
            frame = cv2.rotate(frame, cv2.ROTATE_180)
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


# ── Channel 2: Command Receive (Mac -> Pi) ────────────
def command_server():
    global color_x_error, hand_x_error, color_detected, hand_detected, current_gesture

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
                        hand_x_error    = payload.get('hand_x_error',   0.0)
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
    print("Enter Pi IP on Mac when running socket_client.py")

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