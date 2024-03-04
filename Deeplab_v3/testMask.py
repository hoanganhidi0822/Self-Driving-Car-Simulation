import numpy as np
import cv2
def binary_mask_to_rgb(rgb_image, true_color=[31, 120, 180], false_color=[0, 0, 0]):
    
    rgb_image = np.array(rgb_image)

   
    binary_mask = (rgb_image == 1).all(axis=-1)

    h, w, _ = rgb_image.shape
    final_rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    
    final_rgb_image[binary_mask] = true_color
    final_rgb_image[~binary_mask] = false_color

    return final_rgb_image
index = 10000
# Example usage:
# Assuming your input image is (80, 160, 3) with values of 0 and 1
mask = cv2.imread(f"/media/hoanganh/New Volume/Documents/Researches/Self_Driving_Car/datasetRoad/Train/labels/data{index}.png")
image = cv2.imread(f"/media/hoanganh/New Volume/Documents/Researches/Self_Driving_Car/datasetRoad/Train/images/data{index}.png")
# Define true color for class 1
true_color_for_class_1 = [180, 120, 31]

# Convert RGB image to RGB image with specified colors
final_rgb_image = binary_mask_to_rgb(mask, true_color=true_color_for_class_1, false_color=[0, 0, 0])
final_rgb_image = cv2.resize(final_rgb_image,(640,480))
print(np.unique(final_rgb_image))
cv2.imshow("RGB Mask", final_rgb_image )
cv2.imshow("Image", cv2.resize(image,(640,480)) )
cv2.waitKey()
cv2.destroyAllWindows()