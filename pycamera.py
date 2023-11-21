import cv2
import socket
import numpy as np

def receive_frame(host, port):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((host, port))

    # Receive the video width and height from the server
    width_height_data = client_socket.recv(1024).decode()
    video_width, video_height = map(int, width_height_data.split(','))

    # Create a window to display the video
    cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Stream", video_width, video_height)
    
    while True:
        # Send a request for a frame
        client_socket.sendall(b'next_frame')
    
        # Receive the size of the frame from the server
        size_data = client_socket.recv(4)
        size = int.from_bytes(size_data, byteorder='big')

        # Receive the frame data from the server
        data = b""
        while len(data) < size:
            data += client_socket.recv(size - len(data))

        # Convert the frame data to a buffer
        buffer = np.frombuffer(data, dtype=np.uint8)

        # Decode the buffer to an image frame
        frame = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        # Display the frame
        cv2.imshow("Live Stream", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Send termination signal to the server
    client_socket.sendall(b'quit')

    # Close the socket
    client_socket.close()

    # Close the OpenCV windows
    cv2.destroyAllWindows()

if __name__=="__main__": 
    receive_frame('192.168.71.66',12346)