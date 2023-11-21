import socket
import time
import pickle

host = "192.168.253.66"
motor_control_port = 12345


def send_command(movement, speed=0):
    command = (movement, speed)
    serialized_data = pickle.dumps(command)
    client_socket_motor.send(serialized_data)

client_socket_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_motor.connect((host, motor_control_port))

send_command(movement="forward", speed=100)
time.sleep(2)
