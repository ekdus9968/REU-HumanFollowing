"""
socket_server_experiment.py - Run on Pi (HamBot) [EXPERIMENT + FULL LOGGING]
Path: ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing/server/socket_server_experiment.py

Identical control logic to socket_server.py plus full CSV data logging.
Drives motors. CLOSE -> STOP retained (same as the real system).

Run:
    python server/socket_server_experiment.py <trial_name>
    e.g.) python server/socket_server_experiment.py straight_1

Output files (saved to logs/ folder):
    <trial_name>_raw.csv       : full per-frame data (for graphs)
    <trial_name>_redetect.csv  : per-event REDETECT recovery tracking
    <trial_name>_summary.csv   : summary statistics on shutdown

Press Ctrl+C to stop and auto-save the summary.

REDETECT recovery definition:
    recovered = color reacquired within RECOVERY_TIMEOUT (10 s) of entering REDETECT
    failed    = no color reacquired before timeout

cd ~/Desktop/REU-HumanFollowing/Controller/REU-HumanFollowing
git pull
source ~/Desktop/REU-HumanFollowing/Hambot/hambot_venv/bin/activate

# Pi
python server/socket_server_experiment.py straight_1

# Mac (new terminal)
python3 client/socket_client.py --host <Pi_IP>
"""

import sys
sys.path.append('/home/hambot/Desktop/REU-HumanFollowing/Hambot/src')

import socket
import threading
import struct
import time
import json
import csv
import os
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
from robot_systems.robot import HamBot


# ── Trial name from CLI ────────────────────────────────
TRIAL_NAME = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("trial_%H%M%S")
LOG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

RAW_PATH      = os.path.join(LOG_DIR, f"{TRIAL_NAME}_raw.csv")
REDETECT_PATH = os.path.join(LOG_DIR, f"{TRIAL_NAME}_redetect.csv")
SUMMARY_PATH  = os.path.join(LOG_DIR, f"{TRIAL_NAME}_summary.csv")
# ──────────────────────────────────────────────────────


# ── Configuration ──────────────────────────────────────
HOST         = '0.0.0.0'
VIDEO_PORT   = 5000
CMD_PORT     = 5001

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60

TARGET_DISTANCE      = 500
MAX_SPEED            = 75
SPIN_SPEED           = 2
COLOR_LOST_THRESHOLD = 10

SPEED_FOLLOWING  = 1.0
SPEED_COLOR_ONLY = 0.7
SPEED_HAND_ONLY  = 0.2

RECOVERY_TIMEOUT = 10.0   # seconds - REDETECT recovery success window
# ──────────────────────────────────────────────────────


class State:
    IDLE       = "IDLE"
    FOLLOWING  = "FOLLOWING"
    COLOR_ONLY = "COLOR_ONLY"
    HAND_ONLY  = "HAND_ONLY"
    REDETECT   = "REDETECT"
    STOP       = "STOP"


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


# ── HamBot / Camera Initialization ────────────────────
bot = HamBot(lidar_enabled=True, camera_enabled=False)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
))
picam2.start()
# ──────────────────────────────────────────────────────

lateral_pid = PID(Kp=10.0,  Ki=0.0, Kd=2.0,   output_limit=MAX_SPEED)
forward_pid = PID(Kp=1.0, Ki=0.0, Kd=0.005, output_limit=MAX_SPEED)

# ── Global State ───────────────────────────────────────
current_state      = State.IDLE
color_x_error      = 0.0
hand_x_error       = 0.0
color_detected     = False
hand_detected      = False
current_gesture    = "NONE"
last_color_x_err   = 0.0
target_ever_found  = False
color_lost_count   = 0
stop_gesture_count = 0
lock = threading.Lock()

# ── Metric accumulators ────────────────────────────────
start_time          = None
total_frames        = 0
tracked_frames      = 0
distance_errors     = []      # FOLLOWING state only
state_time          = {s: 0 for s in [State.IDLE, State.FOLLOWING, State.COLOR_ONLY, State.HAND_ONLY, State.REDETECT, State.STOP]}
state_transitions   = 0
correct_transitions = 0
prev_gesture_m      = "NONE"
prev_color_m        = False
prev_hand_m         = False

# ── REDETECT event tracking ────────────────────────────
redetect_events     = []      # list of event dicts
in_redetect         = False
redetect_enter_time = None
redetect_enter_dir  = None
redetect_event_id   = 0

# ── CSV writers ────────────────────────────────────────
raw_file = open(RAW_PATH, 'w', newline='')
raw_writer = csv.writer(raw_file)
raw_writer.writerow([
    "timestamp", "frame", "state",
    "gesture_raw", "gesture_filtered",
    "color_detected", "hand_detected",
    "color_x_error", "hand_x_error", "last_color_x_err",
    "lateral_err_used",
    "color_lost_count", "stop_gesture_count", "target_ever_found",
    "dist_mm", "distance_error",
    "forward_speed", "turn_correction",
    "left_speed", "right_speed", "speed_ratio"
])

redetect_file = open(REDETECT_PATH, 'w', newline='')
redetect_writer = csv.writer(redetect_file)
redetect_writer.writerow([
    "event_id", "enter_time", "exit_time",
    "recovered", "recovery_time_s", "last_direction"
])
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
    """State transition logic (identical to socket_server.py)."""
    if gesture == "CLOSE":
        return State.STOP
    if not target_found:
        return State.IDLE
    if color_det and hand_det and gesture == "OPEN":
        return State.FOLLOWING
    if color_det:
        return State.COLOR_ONLY
    if not color_det and hand_det:
        return State.HAND_ONLY
    if color_lost_count >= COLOR_LOST_THRESHOLD:
        return State.REDETECT
    return State.COLOR_ONLY


def log_redetect_event(ev):
    """Append one REDETECT event row to the redetect CSV."""
    redetect_writer.writerow([
        ev["event_id"], f"{ev['enter_time']:.3f}",
        f"{ev['exit_time']:.3f}" if ev["exit_time"] is not None else "",
        ev["recovered"],
        f"{ev['recovery_time']:.3f}" if ev["recovery_time"] is not None else "",
        ev["last_direction"]
    ])
    redetect_file.flush()


def write_summary():
    """Compute and write summary statistics on shutdown."""
    elapsed = (time.time() - start_time) if start_time else 0
    track_rate = (tracked_frames / total_frames * 100) if total_frames else 0
    mean_derr  = (sum(distance_errors) / len(distance_errors)) if distance_errors else 0
    state_acc  = (correct_transitions / state_transitions * 100) if state_transitions else 0
    redetect_time_pct = (state_time[State.REDETECT] / elapsed * 100) if elapsed else 0

    n_events     = len(redetect_events)
    n_recovered  = sum(1 for e in redetect_events if e["recovered"])
    recovery_rate = (n_recovered / n_events * 100) if n_events else 0
    rec_times    = [e["recovery_time"] for e in redetect_events if e["recovered"] and e["recovery_time"] is not None]
    mean_rec_time = (sum(rec_times) / len(rec_times)) if rec_times else 0

    with open(SUMMARY_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["trial_name", TRIAL_NAME])
        w.writerow(["total_time_s", f"{elapsed:.2f}"])
        w.writerow(["total_frames", total_frames])
        w.writerow(["tracking_success_rate_pct", f"{track_rate:.2f}"])
        w.writerow(["mean_distance_error_mm", f"{mean_derr:.2f}"])
        w.writerow(["following_samples", len(distance_errors)])
        w.writerow(["time_in_redetect_pct", f"{redetect_time_pct:.2f}"])
        w.writerow(["state_transition_accuracy_pct", f"{state_acc:.2f}"])
        w.writerow(["redetect_events", n_events])
        w.writerow(["redetect_recoveries", n_recovered])
        w.writerow(["recovery_rate_pct", f"{recovery_rate:.2f}"])
        w.writerow(["mean_recovery_time_s", f"{mean_rec_time:.2f}"])
        # state time distribution
        for s in state_time:
            pct = (state_time[s] / elapsed * 100) if elapsed else 0
            w.writerow([f"time_{s}_pct", f"{pct:.2f}"])

    print(f"\n{'='*60}")
    print(f"[SUMMARY] {TRIAL_NAME}")
    print(f"  Tracking Success Rate : {track_rate:.1f}%")
    print(f"  Mean Distance Error   : {mean_derr:.1f} mm ({len(distance_errors)} samples)")
    print(f"  Time in REDETECT      : {redetect_time_pct:.1f}%")
    print(f"  REDETECT events       : {n_events} (recovered {n_recovered}, {recovery_rate:.0f}%)")
    print(f"  Mean recovery time    : {mean_rec_time:.2f} s")
    print(f"  Files saved to        : {LOG_DIR}")
    print(f"{'='*60}\n")


def motor_control_loop():
    """State machine + PID control loop with full logging at 20 Hz."""
    global current_state, last_color_x_err, target_ever_found
    global color_lost_count, stop_gesture_count
    global start_time, total_frames, tracked_frames
    global state_transitions, correct_transitions
    global prev_gesture_m, prev_color_m, prev_hand_m
    global in_redetect, redetect_enter_time, redetect_enter_dir, redetect_event_id

    print(f"[MOTOR] Experiment logging started: {TRIAL_NAME}")
    start_time = time.time()
    last_loop  = start_time

    while True:
        now = time.time()
        dt_loop = now - last_loop
        last_loop = now
        t_elapsed = now - start_time

        with lock:
            gesture = current_gesture
            c_x_err = color_x_error
            h_x_err = hand_x_error
            c_det   = color_detected
            h_det   = hand_detected
            last_x  = last_color_x_err

        # CLOSE debounce - 5 consecutive frames to trigger STOP
        if gesture == "CLOSE":
            stop_gesture_count += 1
        else:
            stop_gesture_count = 0
        filtered_gesture = "CLOSE" if stop_gesture_count >= 5 else gesture

        # target_ever_found = True when color OR hand detected
        if c_det or h_det:
            target_ever_found = True

        # Update color lost count and last known direction
        if c_det:
            last_color_x_err = c_x_err
            color_lost_count = 0
        else:
            color_lost_count += 1

        state = determine_state(filtered_gesture, c_det, h_det, target_ever_found)
        with lock:
            current_state = state

        # ── Metric accumulation ──
        total_frames += 1
        if c_det:
            tracked_frames += 1
        state_time[state] += dt_loop

        # State transition accuracy
        if gesture != prev_gesture_m or c_det != prev_color_m or h_det != prev_hand_m:
            expected = determine_state(filtered_gesture, c_det, h_det, target_ever_found)
            state_transitions += 1
            if state == expected:
                correct_transitions += 1
            prev_gesture_m = gesture
            prev_color_m   = c_det
            prev_hand_m    = h_det

        # ── REDETECT event tracking ──
        if state == State.REDETECT and not in_redetect:
            # Entering REDETECT
            in_redetect = True
            redetect_enter_time = t_elapsed
            redetect_enter_dir  = 'RIGHT' if last_x >= 0 else 'LEFT'
        elif state != State.REDETECT and in_redetect:
            # Exiting REDETECT
            in_redetect = False
            recovery_time = t_elapsed - redetect_enter_time
            recovered = c_det and (recovery_time <= RECOVERY_TIMEOUT)
            redetect_event_id += 1
            ev = {
                "event_id": redetect_event_id,
                "enter_time": redetect_enter_time,
                "exit_time": t_elapsed,
                "recovered": recovered,
                "recovery_time": recovery_time if recovered else None,
                "last_direction": redetect_enter_dir,
            }
            redetect_events.append(ev)
            log_redetect_event(ev)

        # Default log values (overwritten per state below)
        dist = None
        distance_error = ""
        forward_speed = 0.0
        turn_correction = 0.0
        left_speed = 0.0
        right_speed = 0.0
        speed_ratio = 0.0
        lateral_err_used = ""

        # ── IDLE / STOP -> stop ──
        if state in (State.IDLE, State.STOP):
            bot.stop_motors()
            lateral_pid.reset()
            forward_pid.reset()

        # ── REDETECT -> spin ──
        elif state == State.REDETECT:
            lateral_pid.reset()
            forward_pid.reset()
            if c_det:
                bot.stop_motors()
            else:
                if last_x >= 0:
                    left_speed  =  SPIN_SPEED
                    right_speed = -SPIN_SPEED
                else:
                    left_speed  = -SPIN_SPEED
                    right_speed =  SPIN_SPEED
                bot.set_left_motor_speed(left_speed)
                bot.set_right_motor_speed(right_speed)

        # ── PID states (FOLLOWING / COLOR_ONLY / HAND_ONLY) ──
        else:
            dist = get_front_distance()

            if state == State.FOLLOWING:
                speed_ratio = SPEED_FOLLOWING
                lateral_err_used = c_x_err
            elif state == State.COLOR_ONLY:
                speed_ratio = SPEED_COLOR_ONLY
                lateral_err_used = c_x_err
            elif state == State.HAND_ONLY:
                speed_ratio = SPEED_HAND_ONLY
                lateral_err_used = h_x_err

            # Longitudinal PID (LiDAR)
            if dist is not None:
                distance_error = dist - TARGET_DISTANCE
                forward_speed  = forward_pid.compute(distance_error) * speed_ratio
            else:
                forward_speed = 0.0
                forward_pid.reset()

            # Lateral PID
            turn_correction = lateral_pid.compute(lateral_err_used) * speed_ratio

            left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_correction))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_correction))

            bot.set_left_motor_speed(left_speed)
            bot.set_right_motor_speed(right_speed)

            # Mean Distance Error accumulation - FOLLOWING state only
            if state == State.FOLLOWING and dist is not None:
                distance_errors.append(abs(dist - TARGET_DISTANCE))

        # ── Write one raw log row ──
        raw_writer.writerow([
            f"{t_elapsed:.3f}", total_frames, state,
            gesture, filtered_gesture,
            c_det, h_det,
            f"{c_x_err:.3f}", f"{h_x_err:.3f}", f"{last_x:.3f}",
            f"{lateral_err_used:.3f}" if lateral_err_used != "" else "",
            color_lost_count, stop_gesture_count, target_ever_found,
            f"{dist:.0f}" if dist is not None else "",
            f"{distance_error:.1f}" if distance_error != "" else "",
            f"{forward_speed:.1f}", f"{turn_correction:.1f}",
            f"{left_speed:.1f}", f"{right_speed:.1f}", f"{speed_ratio:.1f}"
        ])

        # Console output (compact)
        print(f"[{t_elapsed:6.1f}s] {state:11s} | color={str(c_det):5s} hand={str(h_det):5s} | "
              f"dist={str(round(dist)) if dist else '--':>5s} | lat={lateral_err_used if lateral_err_used!='' else 0:+.2f} | "
              f"L={left_speed:+5.1f} R={right_speed:+5.1f}")

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
            ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
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
    print(f"=== REU-HumanFollowing | EXPERIMENT MODE: {TRIAL_NAME} ===")
    print(f"Logs will be saved to: {LOG_DIR}")
    print("Press Ctrl+C to stop and save summary.")

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
        print("\n[STOP] Saving summary...")
        # Close any REDETECT event still open at shutdown
        if in_redetect and redetect_enter_time is not None:
            redetect_event_id += 1
            ev = {
                "event_id": redetect_event_id,
                "enter_time": redetect_enter_time,
                "exit_time": (time.time() - start_time),
                "recovered": False,
                "recovery_time": None,
                "last_direction": redetect_enter_dir,
            }
            redetect_events.append(ev)
            log_redetect_event(ev)
        write_summary()
        raw_file.close()
        redetect_file.close()
        bot.stop_motors()
        picam2.stop()
