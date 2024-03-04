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


img_w, img_h = 160, 80
mapping = {
    0 : (31,120,180),  
    1 : (0, 255, 0)   ,   # road
    2 : (0, 0, 0)   ,     # no label
}

frame = np.zeros((img_w, img_h, 3), np.uint8)
overlayed_img = np.zeros((img_w, img_h, 3), np.uint8)

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

######----Input Image----######
video_path = 'D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\video.mp4'
cap = cv2.VideoCapture(video_path)

########---Image Transform---########
img_transforms = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



while cap.isOpened():
    ret, imgs_np = cap.read()
    pre_time = time.time()
    if not ret:
        break
    #print(imgs_np.shape)
    img_rz = imgs_np[115:, :]
    imgs_np = cv2.resize(img_rz, (img_w, img_h))
    imgs_copy = cv2.cvtColor(imgs_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(imgs_copy)
    



    imgs = img_transforms(img_pil)
    imgs = imgs.view(1, 3, img_h, img_w)
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
        #print(cv2img.shape)
        overlayed_img = 0.5 * cv2.resize(imgs_np, (img_w, img_h)) + 0.5 * cv2img
        overlayed_img = overlayed_img.astype('uint8')
        
    
    ###############################################################################################
    
    ###############################################################################################    
        
    time.sleep(0.01)
    #cv2.imshow("contours",  mask)
    cv2.imshow("image",  cv2img)
    cv2.imshow("seg", overlayed_img)
    cv2.waitKey(1)
    fps = 1 / (time.time() - pre_time)
    #print("Fps: ",fps)
cap.release()
cv2.destroyAllWindows()
