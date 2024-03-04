import cv2
import numpy as np

cv2img = cv2.imread('D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\segImg.jpg')  # Replace 'segmentation_output.jpg' with your image file

# Create a blank canvas with the same size as the input image
contour_image = np.zeros_like(cv2img)

# Draw multiple white horizontal lines on the blank canvas
for y_coord in range(100, 440, 20):
    cv2.line(contour_image, (0, y_coord), (640, y_coord), (255, 255, 255), 1)

        
#cv2img = cv2.GaussianBlur(cv2img, (3,3),1)
#contour_image = np.zeros_like(cv2img)

# Draw multiple white horizontal lines on the blank canvas
""" for y_coord in range(100, 420, 20):
    cv2.line(contour_image, (0, y_coord), (640, y_coord), (255, 255, 255), 1) """

lower_bound = np.array([180, 120, 31], dtype=np.uint8)  # Adjust these values based on your needs
upper_bound = np.array([200, 140, 51], dtype=np.uint8)  # Adjust these values based on your needs
# Create a binary mask for the road segment
mask = cv2.inRange(cv2img, lower_bound, upper_bound)
edges = cv2.Canny(mask, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(cv2img, contours, -1, (255, 255, 255), 1)  
# Display the original image with marked intersection points

cv2.imshow('contourimg', cv2img)
#cv2.imwrite("lineMask.jpg", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()