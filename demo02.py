from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np
import time
import pickle
import socket

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

config = 'experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
snapshot = 'experiments/siamrpn_r50_l234_dwxcorr/model.pth'

# load config
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# create model
model = ModelBuilder()

# load model
model.load_state_dict(torch.load(snapshot,
                                 map_location=lambda storage, loc: storage.cpu()), strict=False)
model.eval().to(device)

# build tracker
tracker = build_tracker(model)

print("Tracker has been built")


def drawToleranceBox(frame):
    frame = cv2.rectangle(frame, (window_center_x-center_tolerance, window_center_y-center_tolerance), (window_center_x+center_tolerance, window_center_y+center_tolerance), (0, 255, 0), 2)
    return frame


def drawCenterCrossline(frame):
    frame = cv2.rectangle(frame, (0, window_center_y-1), (window_width, window_center_y+1), (255, 0, 0), -1)
    frame = cv2.rectangle(frame, (window_center_x-1, 0), (window_center_x+1, window_height), (255, 0, 0), -1)
    return frame


def drawBOXAndDot(frame, bbox):
    x, y, w, h = [int(cord) for cord in bbox]
    c_x, c_y = x + w//2, y + h//2
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    frame = cv2.circle(frame, (c_x, c_y), 7, (0, 0, 255), -1)
    return frame


def send_command(movement, speed=0):
    command = (movement, speed)
    serialized_data = pickle.dumps(command)
    client_socket_motor.send(serialized_data)


def receive_frame():
    # Send a request for a frame
    client_socket_pycam.sendall(b'next_frame')
    size_data = client_socket_pycam.recv(4)
    size = int.from_bytes(size_data, byteorder='big')
    # Receive the frame data from the server
    data = b""
    while len(data) < size:
        data += client_socket_pycam.recv(size - len(data))
    # Convert the frame data to a buffer
    buffer = np.frombuffer(data, dtype=np.uint8)
    # Decode the buffer to an image frame
    frame = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return frame

def receive_distance():
    client_socket_sonar.sendall(b'get_distance')
    dist = float(client_socket_sonar.recv(1024).decode())
    return dist


def pd_x(error_x, dt):
    global prev_error_x
    # PD control gains for the x direction
    kp = 0.1
    kd = 0.3
    # Calculate the proportional term
    proportional_x = kp * error_x
    # Calculate the derivative term (rate of change of errors)
    derivative_x = kd * (error_x - prev_error_x) / dt
    # Update the previous error
    prev_error_x = error_x
    # Calculate the control signal
    control_x = proportional_x + derivative_x
    return control_x


def move_robot(error_x):
    global prev_time
    if (abs(error_x) < center_tolerance):
        dist = receive_distance()
        if (dist >= 50):
            send_command(movement="forward", speed=60)
        else:
            print("Acquired")
            send_command(movement="brake")
    else:
        current_time = time.time()
        dt = current_time - prev_time
        error_control_x = pd_x(error_x, dt)
        speed = turn_speed_range[0] + int((turn_speed_range[1] - turn_speed_range[0]) * min(1.0, abs(
            error_control_x) / window_center_x))
        if (error_x > center_tolerance):
            print("Moving left...")
            send_command(movement="left", speed=speed)
        elif (error_x < -center_tolerance):
            print("Moving right...")
            send_command(movement="right", speed=speed)
    time.sleep(0.01)
    prev_time = time.time()


# Some variables
turn_speed_range = [30, 90]
forward_speed_range = [40, 80]

center_tolerance = 50  # 50px
window_tolerance = 20  # 20px

global prev_error_x
prev_error_x = 0

global prev_time

# Server Configuration
host = "192.168.253.66"
motor_control_port = 12345
pycam_port = 12346
sonar_port = 12347

client_socket_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_pycam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_sonar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket_motor.connect((host, motor_control_port))
client_socket_pycam.connect((host, pycam_port))
client_socket_sonar.connect((host, sonar_port))

# Receive the window width and height from the server
width_height_data = client_socket_pycam.recv(1024).decode()
window_width, window_height = map(int, width_height_data.split(','))

window_center_x = window_width/2
window_center_y = window_height/2

# Create a window to display the window
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", window_width, window_height)

# frame capturing process and tracking
frame = receive_frame()
bbox = cv2.selectROI(
    "Tracking", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)
prev_time = time.time()

while True:
    frame = receive_frame()
    outputs = tracker.track(frame)
    if 'polygon' in outputs:
        polygon = np.array(outputs['polygon']).astype(np.int32)
        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                      True, (0, 255, 0), 3)
        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
        mask = mask.astype(np.uint8)
        mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
    else:
        bbox = list(map(int, outputs['bbox']))
        bbox = [int(cord) for cord in bbox]
        x, y, w, h = bbox
        # frame = drawBOXAndDot(frame, bbox)
        # frame = drawCenterCrossline(frame)
        # frame = drawToleranceBox(frame)
        cv2.rectangle(frame, (x, y),
                              (x + w, y + h),
                              (0, 255, 0), 3)
        
        c_x, c_y = x + w/2, y + h/2
        error_x = window_center_x - c_x
        move_robot(error_x)

    cv2.imshow("Tracking", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

client_socket_pycam.sendall(b'quit')
cv2.destroyAllWindows()
client_socket_pycam.close()
client_socket_motor.close()
client_socket_sonar.close()
