import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model.deeplabv3 import DeepLabV3
import time
import socket         
import time
import cv2
import numpy as np
import json
import base64
from classify.model import Network


# Create a socket object 

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 11000                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

count = 0
angle = 10
speed = 100

sendBack_angle = 0
sendBack_Speed = 10

flag_left = 0
flag_right = 0

img_w, img_h = 480, 240
mapping = {
     0 : (255,255,255),     # roa     
     1 : (106,61,154),     # car
     2 : (0, 0, 0)   ,     # no label
}


current_speed=0
current_angle=0
currentAngle = 0
prevAngle = 0
errorSum = 0
T = 0.01
error = 0


Signal_Traffic = 'straight'
pre_Signal = 'straight'
signArray = np.zeros(15)
noneArray = np.zeros(50)
fpsArray = np.zeros(50)
reset_seconds = 1.0
fps = 20
carFlag = 0

pre_time_left = 0
pre_time_right = 0
curr_time = 0

frame = np.zeros((480, 240, 3), np.uint8)
overlayed_img = np.zeros((480, 240, 3), np.uint8)

torch.backends.cudnn.benchmark = True

#####---Khởi tạo model---######
net = DeepLabV3().cuda()

#####---Load Model---######
state_dict = torch.load('model/test1_model.pth', map_location='cpu')
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


rows_to_check = [140, 150, 160, 175, 200]  

error_arr = np.zeros(5)

pre_t = time.time()

turncheck = [0, 0, 0]
T_right = 0
T_left = 0


model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='best_model_yolo.pt').cuda()

classes = ['straight', 'turn_left', 'turn_right', 'no_turn_left', 'no_turn_right', 'no_straight', 'unknown']
img_size = (64, 64)
model = Network()
model = model.to(device)
model.load_state_dict(torch.load("classify\weightClassify.pth", map_location=device))
model.eval()

def preprocess_image(image, threshold_value=100, min_contour_area=500):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    contour_mask = np.zeros_like(binary_image)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(image, image, mask=contour_mask)

    return result_image

def Predict(img_raw):
    img_rgb = cv2.resize(img_raw, img_size)
    img_rgb = img_rgb / 255
    img_rgb = img_rgb.astype('float32')
    img_rgb = img_rgb.transpose(2, 0, 1)

    img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

    with torch.no_grad():
        img_rgb = img_rgb.to(device)
        y_pred = model(img_rgb)
        _, pred = torch.max(y_pred, 1)
        pred = pred.data.cpu().numpy()
        class_pred = classes[pred[0]]
        y_pred = y_pred[0].cpu()

    return class_pred


def check_sign(signName, num_minSign):
    if signName == classes[0]:
        new_cls_id = 1
    elif signName == classes[1]:
        new_cls_id = 2
    elif signName == classes[2]:
        new_cls_id = 3
    elif signName == classes[3]:
        new_cls_id = 4
    elif signName == classes[4]:
        new_cls_id = 5
    elif signName == classes[5]:
        new_cls_id = 6

    signArray[1:] = signArray[0:-1]
    signArray[0] = new_cls_id
    num_cls_id = np.zeros(6)
    for i in range(6):
        num_cls_id[i] = np.count_nonzero(signArray == (i + 1))

    max_num = num_cls_id[0]
    pos_max = 0
    for i in range(6):
        if max_num < num_cls_id[i]:
            max_num = num_cls_id[i]
            pos_max = i

    if max_num >= num_minSign:
        signName = classes[pos_max]
    else:
        signName = "none"
    
    return signName


def Find_center_points(images, rows):
    result = []
    
    for height in rows:
        arr_head = []
        lineRow = images[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == 255:
                arr_head.append(x)
        if not arr_head:
            arr_head = [130, 130]  
        Min_Head = min(arr_head)
        Max_Head = max(arr_head)
        
        Center_point = ((Min_Head + Max_Head)//2, height)
        result.append(Center_point)
    
    return result


def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    
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

def Detect(image):
    imageOD = cv2.resize(image, (320, 320))
    results = model1(imageOD)
    
    if (len(results.pandas().xyxy[0]) != 0):
        if (float(results.pandas().xyxy[0].confidence[0])) >= 0.85:

            x_min = int(results.xyxy[0][0][0])
            y_min = int(results.xyxy[0][0][1])
            x_max = int(results.xyxy[0][0][2])
            y_max = int(results.xyxy[0][0][3])

            s_bbox = (x_max - x_min) * (y_max - y_min)
            
            img_classifier = imageOD[y_min:y_max, x_min:x_max]
            cv2.imshow("abc",img_classifier)
            sign = Predict(img_classifier)

            if s_bbox > 124 and s_bbox < 600:
                if results.pandas().xyxy[0].name[0] == 'unknown' or sign == "unknown":
                    
                    return "none"
                else:
                   
                    return "decrease"

            elif s_bbox >= 600 and s_bbox <= 1800 and y_min > 10 and float(results.pandas().xyxy[0].confidence[0]) > 0.5:  
                cv2.rectangle(imageOD, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
                
                if sign != "unknown":
                    sign_checked = check_sign(sign, 2)
                    if sign_checked != "none":
                        return sign_checked
                    else:
                        return "unknown"
                else:
                    
                    return "unknown"
        else:
            return "none"
    else:
        signArray[1:] = signArray[0:-1]
        signArray[0] = 0
        return "none"

frame = 0
count = 0
out_sign = "straight"
flag_timer = 0

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
            
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ##################################################################################
            
            image = image.astype('uint8')
            imageCrop = image[190:, :]
            imgs_np = cv2.resize(imageCrop, (img_w, img_h))
            
            img_pil = Image.fromarray(imgs_np)
            imgs = img_transforms(img_pil)
            imgs = imgs.view(1, 3, 240, 480)
            imgs = imgs.cuda()

            #########--Output--########
            with torch.no_grad():
                out2 = net.forward(imgs)

            predict_segment = torch.argmax(out2, 1)[0]
           
            if torch.amax(out2) >= 0.5:
                pred_image = torch.zeros(3, predict_segment.size(0), predict_segment.size(1), dtype=torch.uint8)
                for k in mapping:
                    pred_image[:, predict_segment == k] = torch.tensor(mapping[k]).byte().view(3, 1)

                final_img = pred_image.permute(1, 2, 0).numpy()
                final_img = Image.fromarray(final_img, 'RGB')
                cv2img = np.array(final_img)
                cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
                #cv2imgrz = cv2.resize(cv2img,(160,80))
                overlayed_img = 0.5 * cv2.resize(imgs_np, (480, 240)) + 0.5 * cv2img
                overlayed_img = overlayed_img.astype('uint8')
                cv2imgP = preprocess_image(cv2img)
                turncheck[0] = cv2imgP[54, 50][0]
                turncheck[1] = cv2imgP[54, 240][0]
                turncheck[2] = cv2imgP[54, 430][0]
                #print(turncheck)
            ###############################################################################################
            frame += 1
            if frame % 1 == 0:
                out_sign = Detect(image)
            
            ################################---Find_Middle_Point---#########################################
            center_points = Find_center_points(cv2img, rows_to_check)
            y_cordinate = []
            
            for center_point in center_points:
                y_cordinate.append(center_point[0])
                
            weight = np.array([0.5, 0.38, 0.04, 0.04, 0.04])
            y_cordinate = np.array(y_cordinate)
            position = np.sum(weight*y_cordinate)
            
            error = position - 240
            
            ################################################################################################
            if out_sign == "straight" or out_sign == "none":
                if((turncheck[0] == 0 and turncheck[1] == 0 and turncheck[2] == 0 and flag_left == 1)):
                    error =  y_cordinate[3] - 240
           
            if out_sign == "turn_right" or out_sign == "no_turn_left" :
                T_right = 1
                T_left = 0
                  
            if T_right == 1 and turncheck[0] == 0 and turncheck[1] == 0 and turncheck[2] == 0:
                pre_time_right = time.time()
                if time.time() - pre_time_right < 0.6:
                    error = 100 
                    flag_right = 1
                else:
                    pre_time_right = time.time()
                    T_right = 0
                    flag_right = 0
                
            if out_sign == "turn_left" or out_sign == "no_turn_right":
                T_right = 0
                T_left = 1  
            if T_left == 1 and  turncheck[0] == 0 and turncheck[1] == 0 and turncheck[2] == 0:
                pre_time_left = time.time()
                if time.time() - pre_time_left < 0.6:
                    error = -100    
                    flag_left = 1
                else:
                    T_left = 0
                    pre_time_left = time.time()
                    flag_left = 0
                       
            if((turncheck[0] == 0 and turncheck[1] != 0 and turncheck[2] == 0 and flag_left == 1)):
                T_left = 0
                flag_left = 0
            elif((turncheck[0] == 0 and turncheck[1] != 0 and turncheck[2] == 0 and flag_right == 1)):
                T_right = 0
                flag_right = 0
            ######################################################################################   
            print(T_right, T_left, out_sign,flag_right,flag_left) 
            
            """ speed = 300 - 3*abs(error)
            if out_sign == "decrease":
                speed = -1  """  #1m5s
            speed = 300 - 2*abs(error)
            if out_sign == "decrease" and current_speed > 55:
                speed = -30
            ###################################################################################
            
            angle =  PID(error,0.321,0, 0.043) #0.42 - 0 - 0.09
            
            ###################################################################################
            #cv2.imshow("image",  cv2imgP)
            cv2.imshow("seg", overlayed_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print('closing socket')
        s.close()


