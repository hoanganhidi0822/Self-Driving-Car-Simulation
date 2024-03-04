import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model.deeplabv3 import DeepLabV3
import time
import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

count = 0
angle = 10
speed = 100

def mask_with_color(img, mask, color=(255,255,255)):
    color_mask = np.zeros(img.shape, img.dtype)
    color_mask[:,:] = color
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    return cv2.addWeighted(color_mask, 1, img, 1, 0)  


sendBack_angle = 0
sendBack_Speed = 10

img_w, img_h = 480, 240
mapping = {
     0 : (255,255,255),     # roa     # people
     1 : (106,61,154),     # car
     2 : (0, 0, 0)   ,     # no label
}

Kp = 0.1               
Kd = 0
Ki = 0


current_speed=0
current_angle=0
currentAngle = 0
prevAngle = 0
errorSum = 0
T = 0.01
error = 0


frame = np.zeros((480, 240, 3), np.uint8)
overlayed_img = np.zeros((480, 240, 3), np.uint8)

torch.backends.cudnn.benchmark = True

#####---Khởi tạo model---######
net = DeepLabV3().cuda()

#####---Load Model---######
state_dict = torch.load('D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\Trained_Model/test1_model.pth', map_location='cpu')
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

########---Image Transform---########
img_transforms = transforms.Compose([
    transforms.Resize((240, 480)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

count = 0
angle = 1
speed = 150
device = "cuda"

rows_to_check = [140, 150, 160, 185, 200]  

pre_time = time.time()
tim_str = time.time()
#-------------value PID--------#
error_arr = np.zeros(5)
# error_arr = torch.zeros(5)
pre_t = time.time()

def Find_center_points(images, rows):
    result = []
    
    for height in rows:
        arr_head = []
        lineRow = images[height, :]
        for x, y in enumerate(lineRow):
            #print(y)
            if y[0] == 255:
                #print("hi")
                arr_head.append(x)
        if not arr_head:
            arr_head = [130, 130]  # Default values
        Min_Head = min(arr_head)
        Max_Head = max(arr_head)
        
        Center_point = ((Min_Head + Max_Head)//2, height)
        result.append(Center_point)
    
    return result
def preprocess_image(image, threshold_value=10, min_contour_area=500):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    contour_mask = np.zeros_like(binary_image)
    #cv2.drawContours(contour_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(image, image, mask=contour_mask)

    return result_image

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
            #print("angle: ", current_angle)
            #print("speed: ", current_speed)
            #print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            
            """  image = image*1.2
            image = image.astype('uint8') """
            ##################################################################################
            
            #image = image*0.6
            image = image.astype('uint8')
            imageCrop = image[160:, :]
            imgs_np = cv2.resize(imageCrop, (img_w, img_h))
            imgs_copy = cv2.cvtColor(imgs_np, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(imgs_copy)

            imgs = img_transforms(img_pil)
            imgs = imgs.view(1, 3, 240, 480)
            imgs = imgs.cuda()

            #########--Output--########
            with torch.no_grad():
                out2 = net.forward(imgs)

            predict_segment = torch.argmax(out2, 1)[0]
            #print("predict_segment: ",predict_segment)

            if torch.amax(out2) >= 0.5:
                pred_image = torch.zeros(3, predict_segment.size(0), predict_segment.size(1), dtype=torch.uint8)
                for k in mapping:
                    pred_image[:, predict_segment == k] = torch.tensor(mapping[k]).byte().view(3, 1)

                final_img = pred_image.permute(1, 2, 0).numpy()
                final_img = Image.fromarray(final_img, 'RGB')
                cv2img = np.array(final_img)
                cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
                cv2img = preprocess_image(cv2img)
                overlayed_img = 0.5 * cv2.resize(imgs_np, (480, 240)) + 0.5 * cv2img
                overlayed_img = overlayed_img.astype('uint8')
            
            
            ################################---Find_Middle_Point---#########################################
            center_points = Find_center_points(cv2img, rows_to_check)
            
            y_cordinate = []
            for center_point in center_points:
                y_cordinate.append(center_point[0])
            
                
            weight = np.array([0.5, 0.38, 0.04, 0.04, 0.04])
            y_cordinate = np.array(y_cordinate)
            position = np.sum(weight*y_cordinate)
            #print(np.sum(a))
            
            error = position - 240
            print(error)
            angle =  PID(error,0.28,0,0.03)
            speed = 300 - 1.8*abs(error)
            
            
            ###################################################################################
            cv2.imshow("image",  cv2img)
            cv2.imshow("seg", overlayed_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            
           
        
    finally:
        print('closing socket')
        s.close()


