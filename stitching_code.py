import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

print("Start!")

right_img1 = cv2.imread("right.JPG")
right_img2 = cv2.cvtColor(right_img1,cv2.COLOR_BGR2GRAY)
left_img1 = cv2.imread("left.JPG")
left_img2 = cv2.cvtColor(left_img1,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(right_img2,None)
kp2, des2 = sift.detectAndCompute(left_img2,None)

# Only get the matching keypoints
# not sure really how this works or what it does
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Generate list of good matches
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
        matches = np.asarray(good)

index_in_kp1 = [m.queryIdx for m in matches[:,0]]
index_in_kp2 = [m.trainIdx for m in matches[:,0]]
kp1_matching = np.array([np.array(list(kp1[index].pt)) for index in index_in_kp1])
kp2_matching = np.array([np.array(list(kp2[index].pt)) for index in index_in_kp2])

kp1_matching_KP = np.array([kp1[index] for index in index_in_kp1])

# Generate list of homographies
# Images need to be of same size
homographies = []
(max_x,max_y,z) = right_img1.shape
tau_H = 100
allowed_distance = 25
for t in range(tau_H):
    # Calculate keypoints which are inside the allowed radius of random particle p
    p = [np.random.randint(0, max_x), np.random.randint(0, max_y)] * len( matches[:,0])
    p = np.array(p).reshape(-1,2)

    diff = np.subtract(np.array(kp1_matching), p)
    distance = np.linalg.norm(diff, axis=1)
    extraction_rule = distance < allowed_distance

    # If there are enough matching keypoint to estimate homography, do it 
    if np.sum(extraction_rule) >= 4:
        kp1_in_distance = kp1_matching[extraction_rule]
        kp2_in_distance = kp2_matching[extraction_rule]
        
        outImage = cv2.drawKeypoints(right_img1, list(kp1_matching_KP[extraction_rule]), right_img1)
        cv2.imwrite("keypoints.jpg", outImage) 

        H, masked = cv2.findHomography(kp1_in_distance, kp2_in_distance, cv2.RANSAC, 5.0)
        homographies.append(H)




print("Done!")