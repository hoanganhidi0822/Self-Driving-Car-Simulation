import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
import torch
from modelUnet import UNet
from torchvision import transforms
# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 
count = 0
angle = 1
speed = 150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rows_to_check = [40, 50, 60, 65, 75]  

pre_time = time.time()
tim_str = time.time()
#-------------value PID--------#
error_arr = np.zeros(5)
# error_arr = torch.zeros(5)
pre_t = time.time()



""" Load model Segmentation """
checkpoint_path = "D:\Documents\Researches\Self_Driving_Car\Car_Unet\weights_unet/best_model (3).pth"
modelSeg = UNet(1)
modelSeg = modelSeg.to(device)
modelSeg.load_state_dict(torch.load(checkpoint_path, map_location=device))
modelSeg.eval()


def Find_center_points(images, rows):
    result = []
    
    for height in rows:
        pointArr = []
        lineRow = images[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                pointArr.append(x)
        if not pointArr:
            pointArr = [91, 91]  # Default values
        Min_Point = min(pointArr)
        Max_Point = max(pointArr)
        
        Center_point = ((Min_Point + Max_Point)//2, height//2)
        result.append(Center_point)
    
    return result


def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



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
            start = time.time()
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(10000000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
        
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            
            imageCrop = image[125:, :]
            imageResize = cv2.resize( imageCrop, (160, 80))
            #imageBlur = cv2.blur(imageResize,(5,5))
            image = normalize(imageResize).unsqueeze(0).to(device)
            ######----Detect lane----######
            """ x = torch.from_numpy(imageResize)
            x = x.to(device)
            x = x.transpose(1, 2).transpose(0, 1)
            x = x / 255.0
            x = x.unsqueeze(0).float() """
            with torch.no_grad():
                pred_y = modelSeg(image)
                
                pred_y = torch.sigmoid(pred_y)
                print(pred_y.shape)
                pred_y = pred_y[0]
                pred_y = pred_y.squeeze()
                pred_y = pred_y > 0.5
                pred_y = pred_y.cpu().numpy()
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = pred_y * 255
                
            
                
                
            ######################################################
            center_points = Find_center_points(pred_y, rows_to_check)
            
            y_cordinate = []
            for center_point in center_points:
                y_cordinate.append(center_point[0])
                #cv2.circle(mask1, center_point, 3, (0,255,0), -1)
                
            weight = np.array([0.4, 0.38, 0.2, 0.01, 0.01])
            y_cordinate = np.array(y_cordinate)
            position = np.sum(weight*y_cordinate)
            #print(np.sum(a))
    
            error = position - 80
            #print(error)
            angle =  PID(error,0.38, 0.0001, 0.25)
            speed = 300- 3*abs(error) 
            
            cv2.imshow("IMG", pred_y)
            #print("fps: ", 1/(time.time() - start ))
            key = cv2.waitKey(1)
        
    finally:
        cv2.destroyAllWindows()
        print('closing socket')
        s.close()

