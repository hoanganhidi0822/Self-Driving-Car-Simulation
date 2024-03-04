# Another script using the car_control_module.py

import DataStream
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model.deeplabv3 import DeepLabV3
import time

def mask_with_color(img, mask, color=(255,255,255)):
    color_mask = np.zeros(img.shape, img.dtype)
    color_mask[:,:] = color
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    return cv2.addWeighted(color_mask, 1, img, 1, 0)  

s = DataStream.initialize_connection()
sendBack_angle = 0
sendBack_Speed = 150

img_w, img_h =480, 240
mapping = {
     0 : (255,255,255),     # road
     1 : (0, 0, 0)   ,     # no label
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

rows_to_check = [130, 150, 170, 190, 200]  

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

try:
    while True:
        current_speed, current_angle = DataStream.get_state(s)
        
        if current_speed is not None and current_angle is not None:
            image = DataStream.send_control_command(s, sendBack_angle, sendBack_Speed)
            # Do something with the image, current_speed, and current_angle
            
            image = image*0.6
            image = image.astype('uint8')
            
            imgs_np = cv2.resize(image, (img_w, img_h))
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
                overlayed_img = 0.5 * cv2.resize(imgs_np, (480, 240)) + 0.5 * cv2img
                overlayed_img = overlayed_img.astype('uint8')
            
                #print(np.unique(cv2img))
            ################################---Find_Middle_Point---#########################################
            center_points = Find_center_points(cv2img, rows_to_check)
            #print(center_points)
            """ for i, (min_head, max_head) in enumerate(line_ranges):
                print(f"Line {i + 1}: Min_Head = {min_head}, Max_Head = {max_head}") """
            y_cordinate = []
            for center_point in center_points:
                y_cordinate.append(center_point[0])
                #cv2.circle(mask1, center_point, 3, (0,255,0), -1)
                
            weight = np.array([0.5, 0.38, 0.04, 0.04, 0.04])
            y_cordinate = np.array(y_cordinate)
            position = np.sum(weight*y_cordinate)
            #print(np.sum(a))
            #print(position)
            error = position - 240
           
            sendBack_angle =  PID(error,0.5,0,0.1)
            print(sendBack_angle)
           
            sendBack_Speed = 100 
                
                        
        ####################################################################################################################    

        
        
            cv2.imshow("image",  cv2img)
            cv2.imshow("seg", overlayed_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        ####################--------PID-------#####################
        
        ############################################################
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
       
finally:
    print('Disconnected!!!')
    DataStream.close_connection(s)
