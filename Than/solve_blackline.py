import numpy as np
import cv2
from matplotlib import pyplot as plt

img_dir = 'undistorted3.jpg'

img = cv2.imread(img_dir)
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_hue = np.array([0,0,0])
upper_hue = np.array([180,255,30])
mask = cv2.inRange(hsv,lower_hue,upper_hue)
# cv2.imshow('mask',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = np.ones((5,5),np.float32)/20
blr = img.copy()
blr[:,:,0] = cv2.filter2D(img[:,:,0],-1,kernel)
blr[:,:,1] = cv2.filter2D(img[:,:,1],-1,kernel)
blr[:,:,2] = cv2.filter2D(img[:,:,2],-1,kernel)

img2 = img.copy()
print(img2.shape)
print(blr.shape)
img2[mask!=0,0] = blr[mask!=0,0]
img2[mask!=0,1] = blr[mask!=0,1]
img2[mask!=0,2] = blr[mask!=0,2]


# plt.imshow(cv2.cvtColor(blr, cv2.COLOR_BGR2RGB)), plt.show()
# plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.show()
print('Done!')
cv2.imwrite("unblack_line3.jpg", img2)


