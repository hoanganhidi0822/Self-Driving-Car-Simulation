import numpy as np
import cv2
import matplotlib.pyplot as plt

def Find_center_points(images, rows):
    result = []
    
    for height in rows:
        arr_head = []
        lineRow = images[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_head.append(x)
        if not arr_head:
            arr_head = [91, 91]  # Default values
        Min_Head = min(arr_head)
        Max_Head = max(arr_head)
        
        Center_point = ((Min_Head + Max_Head)//2, height)
        
        result.append(Center_point)
    
    return result



# Create larger masks with white lines
mask1 = cv2.imread("D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\RoadOutliner.jpg", 0)
mask1 = cv2.resize(mask1, (640,480))



""" lower_bound = np.array([180, 120, 31], dtype=np.uint8)  # Adjust these values based on your needs
upper_bound = np.array([200, 140, 51], dtype=np.uint8)  # Adjust these values based on your needs
# Create a binary mask for the road segment
mask = cv2.inRange(mask1, lower_bound, upper_bound) """

rows_to_check = [200, 240, 280, 300, 340]  # Example rows

center_points = Find_center_points(mask1, rows_to_check)
print(center_points)
""" for i, (min_head, max_head) in enumerate(line_ranges):
    print(f"Line {i + 1}: Min_Head = {min_head}, Max_Head = {max_head}") """
y_cordinate = []
for center_point in center_points:
    y_cordinate.append(center_point[0])
    cv2.circle(mask1, center_point, 3, (0,255,0), -1)
    
weight = np.array([0.5, 0.38, 0.04, 0.04, 0.04])
y_cordinate = np.array(y_cordinate)
a = weight*y_cordinate
print(np.sum(a))
# Display the masks and the intersection
plt.figure(figsize=(18, 8))
plt.subplot(1, 3, 1)
plt.imshow(mask1, cmap='gray')
plt.title('Mask 1')
plt.subplot(1, 3, 2)
plt.show()


