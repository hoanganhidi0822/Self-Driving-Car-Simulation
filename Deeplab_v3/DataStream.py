# Save this code in a file named, for example, "car_control_module.py"

import socket
import cv2
import numpy as np

global sendBack_angle, sendBack_Speed, current_speed, current_angle

def initialize_connection():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    PORT = 54321
    s.connect(('127.0.0.1', PORT))
    print("Connected to Simulation!")
    return s

def close_connection(s):
    s.close()

def get_state(s):
    message_getState = bytes("0", "utf-8")
    s.sendall(message_getState)
    state_date = s.recv(100)

    try:
        current_speed, current_angle = map(float, state_date.decode("utf-8").split(' '))
        return current_speed, current_angle
    except Exception as er:
        print(er)
        return None, None

def send_control_command(s, sendBack_angle, sendBack_Speed):
    message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
    s.sendall(message)
    data = s.recv(100000)
    
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        return image
    except Exception as er:
        print(er)
        return None

def control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed

def main():
    s = initialize_connection()

    try:
        while True:
            current_speed, current_angle = get_state(s)

            if current_speed is not None and current_angle is not None:
                image = send_control_command(s, sendBack_angle, sendBack_Speed)
                cv2.imshow("image", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                

    finally:
        print('closing socket')
        close_connection(s)

if __name__ == "__main__":
    main()
