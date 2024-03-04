#------------Include Library----------#
import numpy as np
import time
import cv2
from remove_nouse.remove_nouse_mask import remove_nouse_mask

#-------------time-------------#
pre_time = time.time()
tim_str = time.time()
#-------------value PID--------#
error_arr = np.zeros(5)
# error_arr = torch.zeros(5)
pre_t = time.time()

check = 0
corner = 0
count = 0

#-------Center------#
Center_Left = 0 
Center_Right = 159 - Center_Left

error = 25
width_road = 70
width = np.zeros(10)

#-----------Init One Lane-----------#
UN_MIN_1 = 30
OV_MIN_1 = 60
UN_MAX_1 = 90 # 100
OV_MAX_1 = 130 # 120

def Head_line(images):
    #------------Head------------#
    arr_head=[]
    height = 7 #5
    lineRow = images[height,:]
    for x, y in enumerate(lineRow):
        if y == 255:
            arr_head.append(x)
    if not arr_head:
        arr_head = [91, 91]
    Min_Head = min(arr_head)
    Max_Head = max(arr_head)

    return Min_Head, Max_Head

def Check_line(images):
    #---------------Normal---------------#
    arr_normal=[]
    height = 20 #18  Try 20
    lineRow = images[height,:]
    for x, y in enumerate(lineRow):
        if y == 255:
            arr_normal.append(x)
    if not arr_normal:
        arr_normal = [40, 120]

    Min_Normal = min(arr_normal)
    Max_Normal = max(arr_normal)

    return Min_Normal, Max_Normal

def PWM_Func(error):
    return -3*abs(error) + 300

def Max_SPD_Func(error):
    return -0.125*abs(error) + 50

def Control_Car(mask, pre_Signal, Signal_Traffic, current_speed):
    
    global check
    global corner
    global count
    global error
    global center

    PWM = PWM_Func(error)
    MAX_SPEED = Max_SPD_Func(error)
    
    
    #****************Classified*****************#
    try:
        mask = remove_nouse_mask(mask, sign=Signal_Traffic)
    except Exception as e:
        print(e)

    Min_Normal, Max_Normal = Check_line(mask)
    
    #-----------Straight---------------#
    if Signal_Traffic == 'decrease':
        print("decrease")
        #Straight(mask, PWM, Under_Sendback, MAX_SPEED, current_speed, Min_Normal, Max_Normal)
        sendBack_Speed, center = Straight(mask, PWM, -10, 45, current_speed, Min_Normal, Max_Normal)
        #---------Reset Variable---------#
        corner = 0
        check = 0
        count = 0
        
    elif Signal_Traffic == 'straight':
        print("straight")
        sendBack_Speed, center = Straight(mask, PWM, 10, MAX_SPEED, current_speed, Min_Normal, Max_Normal)

        #---------Reset Variable---------#
        corner = 0
        check = 0
        count = 0

    #-----------no_Straight------------#
    elif Signal_Traffic == 'no_straight':
        print("no_straight")
        sendBack_Speed, center = Straight(mask, PWM, 0, 42, current_speed, Min_Normal, Max_Normal)
    
        if Min_Normal <= 10 and check == 0:
            check = 1
        if Max_Normal >= 150 and check == 0: #Max_Normal == 159
            check = 2

        if check == 1: #Ngã 3 dọc
            Signal_Traffic, center= Turn_Left(Signal_Traffic)

        elif check == 2: # Queọ phải
            Signal_Traffic, center = Turn_Right(Signal_Traffic)
    
    #-----------turnRight------------# #-------------no_turnLeft------------#   
    elif Signal_Traffic == 'turn_right' or Signal_Traffic == 'no_turn_left':
        
        sendBack_Speed, center = Straight(mask, PWM, 0, 42, current_speed, Min_Normal, Max_Normal)
        
        if Max_Normal >= 134 and not check:
            check = 1
        if check:
            Signal_Traffic, center = Turn_Right(Signal_Traffic)
            print("turn_right")
    
    #-------------turnLeft------------# #-----------no_turnRight------------#        
    elif Signal_Traffic == 'turn_left' or Signal_Traffic == 'no_turn_right':
        sendBack_Speed, center = Straight(mask, PWM, 0, 42, current_speed, Min_Normal, Max_Normal)

        if Min_Normal <= 25 and not check:
            check = 1
        if check:
            Signal_Traffic, center = Turn_Left(Signal_Traffic)
            print("turn_left")

    elif Signal_Traffic == 'car_right':
        sendBack_Speed, center = Straight(mask, PWM, 10, MAX_SPEED, current_speed, Min_Normal, Max_Normal)
        center -= 5
                      
    elif Signal_Traffic == 'car_left':
        sendBack_Speed, center = Straight(mask, PWM, 10, MAX_SPEED, current_speed, Min_Normal, Max_Normal)
        center += 5 
    
    # print("Min_Head %d" %(Min_Head))
    # print('Max_Head %d' %(Max_Head))
    # print("Min_Normal %d" %(Min_Normal))
    # print("Max_Normal %d" %(Max_Normal))
    # print("Check %d" %(check))
    # print("Corner %d" %(corner))
    # print("Count %d" %(count))
    # print(Signal_Traffic)

    error = int(mask.shape[1]/2)-center

    return Signal_Traffic, sendBack_Speed, error, corner

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

#----------------Straight-------------#
def Straight(mask, PWM, Under_Sendback, MAX_SPEED, current_speed, Min_Normal, Max_Normal):
    global width_road
    global error
    global count
    global corner

    #----------Center---------#
    Min = Min_Normal
    Max = Max_Normal
    Min_H, Max_H = Head_line(mask)
    
    
    ## 
    if 100 <= Max <= 150 and 2 <= Min <= 70 and not error and not corner:
        width[1:] = width[0:-1]
        if Max - Min > 60:
            width[0] = Max - Min
        width_road = np.average(width)
    # print("width_road: " + str(width_road))

    center = int((Min+Max)/2)
    #----------Safe Setpoint--------#
    if Max >= OV_MAX_1 and UN_MIN_1 <= Min <= OV_MIN_1 and not Min_H == Max_H == 91:
        center = Min + int(width_road/2)
        # print("Change Min")
    elif Min < UN_MIN_1 and UN_MAX_1 <= Max <= OV_MAX_1 and not Min_H == Max_H == 91:
        center = Max - int(width_road/2)
        # print("Change Max")

    #----------Speed----------#
    sendBack_Speed = PWM
    if float(current_speed) < 20.0:
        sendBack_Speed = 300
    elif float(current_speed) > MAX_SPEED: #Adjust Speed
        sendBack_Speed = Under_Sendback

    # Position Turn
    if Max_Normal == 120 and Min_Normal == 40 or Max_Normal == 159 or Min_Normal == 0:
        count += 1

    return sendBack_Speed, center

def Turn_Right(Signal_Traffic):
    global center
    global pre_time
    global check
    global corner
    global count

    if not corner and count:
        pre_time = time.time()       
        corner = 1

    if corner:
        if time.time() - pre_time < 1.0:
            center = Center_Right
        else:
            Signal_Traffic = 'straight'
            corner = 0
            check = 0
            count = 0
        
    return Signal_Traffic, center
    
def Turn_Left(Signal_Traffic):
    global center
    global pre_time
    global check
    global corner
    global count

    if not corner and count:
        pre_time = time.time()       
        corner = 1

    if corner:
        if time.time() - pre_time < 1.0:
            center = Center_Left
            print("quẹo trái")
        else:
            Signal_Traffic = 'straight'
            corner = 0
            check = 0
            count = 0

    return Signal_Traffic, center
 
