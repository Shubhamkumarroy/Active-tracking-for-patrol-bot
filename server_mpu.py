import socket
from mpu6050 import mpu6050
import time
import json

HOST = '0.0.0.0'
PORT = 8080

mpu_address = 0x68
mpu = mpu6050(mpu_address)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"Server listening on {HOST}:{PORT}")

    try:
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")

            while True:
                accel_data = mpu.get_accel_data()
                gyro_data = mpu.get_gyro_data()

                data_to_send = {
                    'accel': {
                        'x': accel_data['x'],
                        'y': accel_data['y'],
                        'z': accel_data['z']
                    },
                    'gyro': {
                        'x': gyro_data['x'],
                        'y': gyro_data['y'],
                        'z': gyro_data['z']
                    }
                }

                # Convert the dictionary to a JSON string
                json_data = json.dumps(data_to_send)
                conn.sendall(json_data.encode('utf-8'))

                time.sleep(1)
    except KeyboardInterrupt:
        print("Server terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
