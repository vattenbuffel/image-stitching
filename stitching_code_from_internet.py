import cv2
import numpy as np
import matplotlib.pyplot as plt
right_img1 = cv2.imread("right.JPG")
right_img2 = cv2.cvtColor(right_img1,cv2.COLOR_BGR2GRAY)
left_img1 = cv2.imread("left.JPG")
left_img2 = cv2.cvtColor(left_img1,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(right_img2,None)
kp2, des2 = sift.detectAndCompute(left_img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
        matches = np.asarray(good)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    #print H
else:
    raise AssertionError("Can't find enough keypoints.")

dst = cv2.warpPerspective(right_img1,H,(left_img1.shape[1] + right_img1.shape[1], left_img1.shape[0]))
cv2.imwrite("warped.jpg",dst)
dst[0:left_img1.shape[0], 0:left_img1.shape[1]] = left_img1
cv2.imwrite("output.jpg",dst)