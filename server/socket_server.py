"""
socket_server.py - Run on Pi (HamBot)
Path: ~/REU-HumanFollowing/server/socket_server.py
 
Channel 1 (port 5000): Pi camera → send Mac 
Channel 2 (port 5001): Mac send gesture command → control motor
 
ref repo (no pull/push here):
    ~/HamBot/  ← only import using sys.path
"""

import sys
sys.path.append('/home/hambot/Desktop/REU-HumanFollowing/Hambot/src')

import socket
import threading
import struct
import cv2
import numpy as np
from picamera2 import Picamera2
from robot_system.robot import HamBot

# ====================
#       SETTING         
# ====================

HOST = '0.0.0.0'
VIDEO_PORT = 5000
CMD_PORT = 5001
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60       # smaller number, faster (reduce bandwidth)
MOTOR_SPEED  = 50       # motor vel (%)


# =================================
#       HAMBOT / CAMERA RESET         
# =================================

bot = HamBot(lidar_enabled=False, camera_enabled=False)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}))
picam2.start()


# ============================
#       CONTROL MOTOR       
# ============================

def apply_gesture_command(gesture: str):
    """gesture comman -> motor"""
    gesture = gesture.strip().upper()
    print(f"[CMD] Send : {gesture}")
    
    if gesture == "OPEN":
        bot.set_left_motor_speed(MOTOR_SPEED)
        bot.set_right_motor_speed(MOTOR_SPEED)
    elif gesture == "CLOSE":
        bot.set_left_motor_speed(MOTOR_SPEED)
        bot.set_right_motor_speed(MOTOR_SPEED)
    elif gesture == "POINTER":
        bot.stop_motors()
    elif gesture == "OK":
        bot.set_left_motor_speed(-MOTOR_SPEED)
        bot.set_right_motor_speed(-MOTOR_SPEED)
    else:
        print(f"[CMD] UNKNOWN COMMAND: {gesture}")


# =================================
#       CHANNEL 1: SEND VIDEO
# =================================

def video_stream_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, VIDEO_PORT))
    server.listen(1)
    print(f"[VIDEO] Waiting... port {VIDEO_PORT}")
    
    conn, addr = server.accept()
    print(f"[VIDEO] Conncected: {addr}")
    
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
        print("[VIDEO] Mac Disconnected")
    finally:
        conn.close()
        server.close()
        
# =======================================
#       CHANNEL 2: COMMAND RECEIVED
# =======================================

def command_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, CMD_PORT))
    server.listen(1)
    print(f"[CMD] Waiting... port {CMD_PORT}")
    
    conn, addr = server.accept()
    print(f"[CMD] Connected: {addr}")
    
    buffer = ""
    try: 
        while True:
            data = conn.recv(64).decode('utf-8')
            if not data:
                break
            buffer += data
            while '\n' in buffer:
                cmd, buffer = buffer.split('\n', 1)
                apply_gesture_command(cmd)
    
    except ConnectionResetError:
        print("[CMD] Mac Disconnected")
    finally:
        bot.stop_motors()
        conn.close()
        server.close()
        
if __name__ == '__main__' :
    print("=== REU-HumanFollowing | Hambot Server Start ===")
    print(f"Run Pi IP on MAC -- Enter host IP")
    
    t_video = threading.Thread(target = video_stream_server, daemon=True)
    t_cmd = threading.Thread(target=command_server, daemon=True)
    t_video.start()
    t_cmd.start()
    
    try: 
        t_video.join()
        t_cmd.join()
    except KeyboardInterrupt:
        print("\n[STOP] SERVER END")
        bot.stop_motors()
        picam2.stop()