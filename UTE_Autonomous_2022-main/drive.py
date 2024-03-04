import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
import torch
from modelSegmentation import build_unet

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 
count = 0
angle = 10
speed = 100
device = "cuda"

pre_time = time.time()
tim_str = time.time()
#-------------value PID--------#
error_arr = np.zeros(5)
# error_arr = torch.zeros(5)
pre_t = time.time()


""" Load model Segmentation """
checkpoint_path = "./weights_unet/unet_v2.pth"
modelSeg = build_unet()
modelSeg = modelSeg.to(device)
modelSeg.load_state_dict(torch.load(checkpoint_path, map_location=device))
modelSeg.eval()


def Head_line(images):
    #------------Head------------#
    arr_head=[]
    height =  40#5
    lineRow = images[height,:]
    for x, y in enumerate(lineRow):
        if y == 255:
            arr_head.append(x)
    if not arr_head:
        arr_head = [91, 91]
    Min_Head = min(arr_head)
    Max_Head = max(arr_head)

    return Min_Head, Max_Head


def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            """ print("angle: ", current_angle)
            print("speed: ", current_speed)
            print("---------------------------------------") """
            print("speed: ", current_speed)
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            # cv2.imshow("IMG", imgage)
            # print("Img Shape: ",imgage.shape)
            
            imageSeg = imgage[125:, :]
            imageSeg = cv2.resize( imageSeg, (160, 80))
            
            start = time.time()
            """Detect lane"""
            x = torch.from_numpy(imageSeg)
            x = x.to(device)
            x = x.transpose(1, 2).transpose(0, 1)
            x = x / 255.0
            x = x.unsqueeze(0).float()
            with torch.no_grad():
                pred_y = modelSeg(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0]
                pred_y = pred_y.squeeze()
                pred_y = pred_y > 0.5
                pred_y = pred_y.cpu().numpy()
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = pred_y * 255
                
            min_point, max_point= Head_line(pred_y)
            center = (min_point + max_point)//2
            
            error = center - 80
            
            angle =  PID(error,0.2,0,0.02)
            speed = 150
            print(min_point, max_point)
            cv2.imshow("IMG", pred_y)

            
            
            #save image
            # image_name = "./img/img_{}.jpg".format(count)
            # count += 1
            # cv2.imwrite(image_name, imgage)
            key = cv2.waitKey(1)
        
            

    finally:
        print('closing socket')
        s.close()

