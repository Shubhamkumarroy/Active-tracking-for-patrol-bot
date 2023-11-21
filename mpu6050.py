import socket
import json
import time

HOST = '192.168.71.66'
PORT = 8080

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    try:
        while True:
            data_received = client_socket.recv(1024)
            data_received=data_received.decode('utf-8')
            data_received = json.loads(data_received)
            accel_data=data_received['accel']
            gyro_data=data_received['gyro']
            print("Accelerometer data:")
            print("  X:", accel_data['x'])
            print("  Y:", accel_data['y'])
            print("  Z:", accel_data['z'])
            
            print("Gyroscope data:")
            print("  X:", gyro_data['x'])
            print("  Y:", gyro_data['y'])
            print("  Z:", gyro_data['z'])
            time.sleep(2)
    except KeyboardInterrupt:
        print("Client terminated by user.")


