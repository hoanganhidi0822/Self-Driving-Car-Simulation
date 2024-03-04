
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
image_path = "D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\dataset test\data4k\images\data1069.jpg"
image = cv2.imread(image_path)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(np.real(image))
plt.show()