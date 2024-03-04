# Import socket module
import socket
import cv2
import numpy as np
import torch
import time
# from model import build_unet
# import pandas as pd
from classify.model import Network
from modelSegmentation import build_unet
import warnings
from Controller_v5 import *

torch.cuda.set_device(0)

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
device = "cuda"

Signal_Traffic = 'straight'
pre_Signal = 'straight'
signArray = np.zeros(15)
noneArray = np.zeros(50)
fpsArray = np.zeros(50)
reset_seconds = 1.0
fps = 20
carFlag = 0

# ----------------------------Yolov5------------------------#
""" Load model Segmentation """
checkpoint_path = "weights_unet/unet_v2.pth"
modelSeg = build_unet()
modelSeg = modelSeg.to(device)
modelSeg.load_state_dict(torch.load(checkpoint_path, map_location=device))
modelSeg.eval()

""" Load model Object detection """
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='weights_yolov5/best_m.pt').cuda()

# ----------------------------Classifier--------------------#
classes = ['straight', 'turn_left', 'turn_right', 'no_turn_left', 'no_turn_right', 'no_straight', 'unknown']
img_size = (64, 64)
model = Network()
model = model.to(device)
model.load_state_dict(torch.load("./classify/weight6_sum.pth", map_location=device))
model.eval()


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


# def remove_small_contours(image):
#     image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
#     contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
#     mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
#     image_remove = cv2.bitwise_and(image, image, mask=mask)
#     return image_remove

def remove_small_contours(image, max_area = 300):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    filteredContours=[] 
    for i in contours: 
        area=cv2.contourArea(i) 
        if area>max_area: 
            filteredContours.append(i) 

    mask = cv2.drawContours(image_binary, filteredContours, -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove

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

    # new_cls_id = box[6] + 1
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
    # print(signArray)
    # print(num_cls_id)
    return signName

# ------------------------------Simulator--------------------#
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))
# s.connect(('127.0.0.1', PORT))

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


def Detect(image):
    imageOD = cv2.resize(image, (320, 320))
    # cv2.imshow("imageOD", imageOD)
    results = model1(imageOD)
    # print(len(results.pandas().xyxy[0].confidence))
    if (len(results.pandas().xyxy[0]) != 0):
        if (float(results.pandas().xyxy[0].confidence[0])) >= 0.85:

            x_min = int(results.xyxy[0][0][0])
            y_min = int(results.xyxy[0][0][1])
            x_max = int(results.xyxy[0][0][2])
            y_max = int(results.xyxy[0][0][3])

            x_c = int(x_min + (x_max - x_min)/2)
            y_c = int(y_min + (y_max - y_min)/2)

            s_bbox = (x_max - x_min) * (y_max - y_min)
            # print("y min:  ", y_min, "y max:   ", y_max)
            # print("Size bbox:   ", s_bbox)
            # if results.pandas().xyxy[0].name[0] == 'car' and s_bbox >= 800:
            #     if x_max >= 160 and y_max > 150:
            #         return "car_right"
            #     else:
            #         return "car_left"

            img_classifier = imageOD[y_min:y_max, x_min:x_max]
            cv2.imshow("classify", img_classifier)
            sign = Predict(img_classifier)

            if s_bbox > 30 and s_bbox < 250:
                if results.pandas().xyxy[0].name[0] == 'unknown' or sign == "unknown":
                    # print("Normal", s_bbox)
                    return "none"
                else:
                    # print("Slow down", s_bbox)
                    return "decrease"


            elif s_bbox >= 250 and s_bbox <= 1200 and y_min > 10 and float(results.pandas().xyxy[0].confidence[0]) > 0.88:  # and y_min > 10 and x_max < 270:
                cv2.rectangle(imageOD, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
                # cv2.imshow("cls", img_classifier)
                if sign != "unknown":
                    sign_checked = check_sign(sign, 2)
                    if sign_checked != "none":
                        return sign_checked
                    else:
                        return "unknown"
                else:
                    # print("Unknown ------ Ignore")
                    return "unknown"

            elif s_bbox >= 1000: #1200
                if results.pandas().xyxy[0].name[0] == "car":
                    s_max = 0
                    x_max = 0
                    y_max = 0
                    for i in range(len(results.xyxy[0])):
                        x1 = int(results.xyxy[0][0][0])
                        y1 = int(results.xyxy[0][0][1])
                        x2 = int(results.xyxy[0][0][2])
                        y2 = int(results.xyxy[0][0][3])
                        s = (x2 - x1) * (y2 - y1)
                        if s > s_max:
                            s_max = s
                            x_max = x2
                            y_max = y2
                    if x_max >= 160 and y_max > 150:
                        return "car_right"
                    else:
                        return "car_left"

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
        while True:

            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(10000000)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )

                # -------------------------------------------Workspace---------------------------------- #
                if not flag_timer:
                    start = time.time()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    imageSeg = image[125:, :]
                    imageSeg = cv2.resize(imageSeg, (160, 80))

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

                    try:
                        pred_y = remove_small_contours(pred_y, 250)
                    except Exception as e:
                        print(e)

                    frame += 1
                    if frame % 1 == 0:
                        out_sign = Detect(image)
                        # print("Objects:   ", out_sign)
                        # if out_sign == 'unknown' and Signal_Traffic == 'decrease' and pre_Signal =="decrease":
                        #     Signal_Traffic = 'straight'

                    # ------------------- Check none array --------------- #
                    if carFlag == 0:
                        if frame >= 50 and frame < 100:
                            fpsArray[frame - 50] = fps
                        elif frame >= 100 and frame < 120:
                            noneArray = np.zeros(int(np.mean(fpsArray) * reset_seconds))
                            carArray = noneArray[1:int(len(noneArray) / 2)]
                        elif frame > 150:
                            if out_sign == "none" or out_sign == None:
                                noneArray[1:] = noneArray[0:-1]
                                noneArray[0] = 0

                            else:
                                noneArray[1:] = noneArray[0:-1]
                                noneArray[0] = 1

                            if np.sum(noneArray) == 0:
                                out_sign = "straight"

                    elif carFlag == 1:
                        if out_sign == "none" or out_sign == None or out_sign == "unknown":
                            carArray[1:] = carArray[0:-1]
                            carArray[0] = 0

                        else:
                            carArray[1:] = carArray[0:-1]
                            carArray[0] = 1

                        if np.sum(carArray) == 0:
                            out_sign = "straight"
                    # ---------------------------------------------------- #
                    # print("***********Segment + OD***********")
                # ---------------Controller--------------#
                pre_Signal = Signal_Traffic

                if out_sign != "unknown" and out_sign != None and out_sign != "none":
                    if out_sign == "car_left" or out_sign == "car_right":
                        carFlag = 1
                    else:
                        carFlag = 0
                    Signal_Traffic = out_sign


                # print("out_sign: ", out_sign)
                print("Signal Traffic: ", Signal_Traffic)
                # print("---------------")
                Signal_Traffic, sendBack_Speed, error, flag_timer = Control_Car(pred_y, out_sign, Signal_Traffic,
                                                                                current_speed)

                # -------------Angle------------#
                sendBack_angle = -PID(error, 0.40, 0, 0.05)

                Control(sendBack_angle, sendBack_Speed)

                # print("flag_timer: " + str(flag_timer))
                # print("Trafficsign: ", out_sign)
                # print("Car flag:  ", carFlag)
               

                end = time.time()
                fps = 1 / (end - start)
                # print(fps)
                cv2.imshow("image", image)
                cv2.imshow("image segmentation", pred_y)
                cv2.waitKey(1)

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()
