import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.serialization import load
from tqdm import tqdm
import time
from model.deeplabv3 import DeepLabV3
def mask_with_color(img, mask, color=(255,255,255)):
    color_mask = np.zeros(img.shape, img.dtype)
    color_mask[:,:] = color
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    return cv2.addWeighted(color_mask, 1, img, 1, 0)  
mapping = {
(31,120,180):0, # road
(227,26,28) :1, #people
(106,61,154):2, #car
(0, 0, 0)   :3, #no label
}#100 duong,86 nguoi, 84 xe
frame = np.zeros((640, 480, 3), np.uint8)
overlayed_img = np.zeros((640, 480, 3), np.uint8)
none_img = np.zeros((640, 480, 3), np.uint8)
torch.backends.cudnn.benchmark = True
net = DeepLabV3().cuda()
state_dict = torch.load('D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\Trained_Model/best_model.pth', map_location='cpu')
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v


net.load_state_dict(compatible_state_dict, strict=False)
net.eval()
#cap = cv2.VideoCapture('a.avi')
imgs_np=cv2.imread('D:\Documents\Researches\Self_Driving_Car\Images\Data1580.jpg')

ret = True
img_transforms = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
rev_mapping = {mapping[k]: k for k in mapping}
img_w, img_h = 640, 480

#overlayed_img = np.zeros((480, 640, 3), np.uint8)
# rev_mapping = {mapping[k]: k for k in mapping}
#plt.cla()
pre_time = time.time()
#
#ret , imgs_np = cap.read()

imgs_np=cv2.resize(imgs_np,(img_w,img_h))
#print(imgs_np.shape)

imgs_copy = cv2.cvtColor(imgs_np, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(imgs_copy)

imgs = img_transforms(img_pil)

imgs = imgs.view(1, 3, 480, 640)
imgs = imgs.cuda()

# with torch.no_grad():
out2 = net.forward(imgs)
fps = 1/(time.time() - pre_time)
    #print(out2)

    # print(type(out2))
    # print(out2.shape)
######################segment###########################
predict_segment= torch.argmax(out2,1)[0]
#print(predict_segment.size())
out_seg = out2.reshape(4, 480, 640)

if torch.amax(out2) >= 0.5:
    pred_image = torch.zeros(3, predict_segment.size(0), predict_segment.size(1), dtype=torch.uint8)
    for k in rev_mapping:
        pred_image[:, predict_segment==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
    print(pred_image.size())
    final_img = pred_image.permute(1, 2, 0).numpy()
    #print(final_img.shape())
    final_img = Image.fromarray(final_img,'RGB')
    cv2img = np.array(final_img)
    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
    overlayed_img =0.6*cv2.resize(imgs_np,(640,480)) + 0.4*cv2img
    overlayed_img=overlayed_img.astype('uint8')
    
######################---Find_Middle_Point---##############################
    cv2img = cv2.GaussianBlur(cv2img, (7,7),1)
    contour_image = np.zeros_like(cv2img)

    # Draw multiple white horizontal lines on the blank canvas
    """ for y_coord in range(100, 420, 20):
        cv2.line(contour_image, (0, y_coord), (640, y_coord), (255, 255, 255), 1) """

    lower_bound = np.array([180, 120, 31], dtype=np.uint8)  # Adjust these values based on your needs
    upper_bound = np.array([200, 140, 51], dtype=np.uint8)  # Adjust these values based on your needs
    # Create a binary mask for the road segment
    mask = cv2.inRange(cv2img, lower_bound, upper_bound)
    edges = cv2.Canny(mask, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)  # Green color, thickness = 1

    # Find intersection points for each line
    middle_points = [] 
    for y_coord in range(100, 420, 20):
        intersection_points = []
        
        
        for contour in contours:
            for point in contour[:, 0]:
                if point[1] == y_coord:  # Check if the y-coordinate matches the line
                    intersection_points.append(point)
                    #print((intersection_points))

        # Extract x-coordinates
        x_coordinates = [point[0] for point in intersection_points]

        # If there are more than two points, consider only the highest and lowest x-coordinates
        if len(x_coordinates) > 2:
            max_x = max(x_coordinates)
            min_x = min(x_coordinates)
            intersection_points = [(min_x, y_coord), (max_x, y_coord)]

        #print(f"Intersection points for line at y={y_coord}: {intersection_points}")

        # Mark intersection points on the original image
        #print(intersection_points)
        for point in intersection_points:
            intersection_point = list(intersection_points)
            if len(intersection_point) == 2:
                #print(intersection_point)
                middle_point = ((intersection_point[0][0] + intersection_point[1][0])//2, intersection_point[0][1] )
                middle_points.append(middle_point)
                cv2.circle(overlayed_img, middle_point, 3, (0, 0, 0), -1)  # Green color, filled circle
            else:
                pass
            #print(middle_point)
            cv2.circle(overlayed_img, point, 3, (0, 255, 0), -1)  # Green color, filled circle
    print(middle_points)
    if middle_points:
        mean_x = np.mean([point[0] for point in middle_points])
        #print("Mean x-coordinate for middle points:", int(mean_x))
        cv2.circle(overlayed_img, (int(mean_x), 260), 3, (0, 0, 255), -1)  # Green color, filled circle
    else:
        print("No middle points found.")
    
        
####################################################################################################################
    
#cv2.putText(overlayed_img, f'{fps}',(50,50), font, 1,(255,255,255),2) 

cv2.imshow('Road Contour', contour_image) 
cv2.imwrite('RoadOutliner.jpg', mask)
cv2.imshow("final_img", cv2img)
cv2.imshow("seg", overlayed_img)
  
#cv2.imwrite("result.png", overlayed_img)
cv2.waitKey()
cv2.destroyAllWindows()

