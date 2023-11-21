import socket

def receive_distance():
    client_socket_sonar.sendall(b'get_distance')
    dist = client_socket_sonar.recv(1024).decode()
    return dist

host = "192.168.253.66"
port = 12347

client_socket_sonar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_sonar.connect((host, port))

print("Distance is: " + receive_distance())