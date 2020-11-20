import numpy as np
import cv2
from matplotlib import pyplot as plt

img_dir = 'undistorted.jpg'

img = cv2.imread(img_dir)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_hue = np.array([0,0,0])
upper_hue = np.array([50,50,100])
mask = cv2.inRange(hsv,lower_hue,upper_hue)


kernel = np.ones((5,5),np.float32)/20

blr = cv2.filter2D(img,-1,kernel)

img2 = img.copy()
img2[mask!=0] = blr[mask!=0]


plt.imshow(cv2.cvtColor(blr, cv2.COLOR_BGR2RGB)), plt.show()
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.show()
cv2.imwrite("unblack_line.jpg", img2)


