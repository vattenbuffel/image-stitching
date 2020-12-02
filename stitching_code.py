import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

print("Start!")

right_img1 = cv2.imread("right.JPG")
right_img2 = cv2.cvtColor(right_img1,cv2.COLOR_BGR2GRAY)
left_img1 = cv2.imread("left.JPG")
left_img2 = cv2.cvtColor(left_img1,cv2.COLOR_BGR2GRAY)

# Finding homographies step

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
        
        #outImage = cv2.drawKeypoints(right_img1, list(kp1_matching_KP[extraction_rule]), right_img1)
        #cv2.imwrite("keypoints.jpg", outImage) 

        H, masked = cv2.findHomography(kp1_in_distance, kp2_in_distance, cv2.RANSAC, 5.0)
        homographies.append(H)

homographies = np.array(homographies)


# Filtering step
# Screening
# Remove homographies which are unlikely to give realistic images
threshold = 0.01 # 0.01 is a standard value 
(x_max,y_max,z) = right_img1.shape
corners = np.array([[0, 0, x_max, x_max], [0, y_max, y_max, 0], [1,1,1,1]])

# Convert homogenous coordinates to euclidian
def homogenous_to_euclidian(x):
    height, _  = x.shape
    return np.true_divide(x[0:height-1, :], x[-1, :])

# Helper function to convert 1D array to similarity matrix
def convert_1D_array_to_similarity_matrix(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[4]
    H_s = [[a, -b, c], [b, a, d], [0,0,1]]
    return np.array(H_s)

# Find the optimal similarity matrix. This is done by minimizing this function
def difference_between_similarity_and_H(x, H):
    H_s = convert_1D_array_to_similarity_matrix(x)

    # Calculate the location of the corners
    corners_H_s = np.matmul(H_s, corners)
    corners_H = np.matmul(H, corners)
    
    # Convert from homogenous to euclidian
    corners_H_s = homogenous_to_euclidian(corners_H_s)
    corners_H = homogenous_to_euclidian(corners_H)

    # Calculate the difference between the corners
    diff = np.subtract(corners_H_s, corners_H)
    norm = np.linalg.norm(diff, axis = 0)
    return np.sum(norm)

homographies_to_keep = []
for i in range(homographies.shape[0]):
    H = homographies[i]
    x0 = [1, 1, 1, 1, 1]
    H_s = optimize.fmin(difference_between_similarity_and_H, x0, disp = False, args = (H,))
    
    difference_normalized = difference_between_similarity_and_H(H_s, H) / (max_x*max_y) # It should be normalized by image size. It's a bit unclear what image size is though...
    homographies_to_keep.append( difference_normalized< threshold)
        
homographies = homographies[homographies_to_keep,:,:]


# Remove homographies which the magnitude of the scaling parameters is bigger than a threshold
# I'm not sure if this is correct or not
magnitude_threshold = 1.2 # I have no idea what this should be
scales = np.matmul(homographies, corners[:,2])
homographies_to_keep = np.absolute(scales[:, -1]) < magnitude_threshold # Shouldn't there be a minimum threshold as well?
homographies = homographies[homographies_to_keep]
    

 
# Remove homographies which are too close to identity



# Remove duplicates

print("Done!")