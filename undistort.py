# Undistort images

import cv2
import numpy as np
import scipy.linalg as scipy_linalg
from scipy import optimize
import numpy.matlib as npm

# Convert 2D homogenous coordinates to euclidian
def homogenous_to_euclidian_2D(x):
    height, _  = x.shape
    return np.true_divide(x[0:height-1, :], x[-1, :])

# Convert 2D euclidian coordinates to homogenous
def euclidian_to_homogenous_2D(x):
    _, width  = x.shape
    return np.vstack([x, np.ones((1, width))])


right_img1 = cv2.imread("right.JPG")
right_img2 = cv2.cvtColor(right_img1,cv2.COLOR_BGR2GRAY)
left_img1 = cv2.imread("left.JPG")
left_img2 = cv2.cvtColor(left_img1,cv2.COLOR_BGR2GRAY)

# This should be removed
scale_percent = 100 # percent of original size
width = int(right_img2.shape[1] * scale_percent / 100)
height = int(right_img2.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
right_img2 = cv2.resize(right_img2, dim, interpolation = cv2.INTER_AREA)
left_img2 = cv2.resize(left_img2, dim, interpolation = cv2.INTER_AREA)
right_img1 = cv2.resize(right_img1, dim, interpolation = cv2.INTER_AREA)
left_img1 = cv2.resize(left_img1, dim, interpolation = cv2.INTER_AREA)

img1 = right_img2
img2 = left_img2
##########################################################

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Find the inliers using ransac
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

good_matches = np.array(good)[np.argwhere(np.array(matchesMask)).reshape(-1)]
matching_keypoints_img = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None, **draw_params)
cv2.imwrite("matching_keypoints.jpg", matching_keypoints_img) 

# Extract the matching keypoints, ie inliners
kp1_matching = np.array([np.array(list(kp1[match.queryIdx].pt)) for match in good_matches]).transpose()
kp2_matching = np.array([np.array(list(kp2[match.trainIdx].pt)) for match in good_matches]).transpose()

# Move the image origin to the center of the image
y_max, x_max = img1.shape
kp1_matching = (kp1_matching.transpose() - np.array([x_max/2, y_max/2])).transpose()
kp2_matching = (kp2_matching.transpose() - np.array([x_max/2, y_max/2])).transpose()

# Rescale the image by the image diameter, divide by W+H
kp1_matching /= (y_max+x_max)
kp2_matching /= (y_max+x_max)

# Create D1, D2 and D3
x_1 = kp1_matching[0,:]        # x in paper
y_1 = kp1_matching[1,:]        # y in paper
r_1 = (x_1**2 + y_1**2)**0.5   # r in paper
x_2 = kp2_matching[0,:]        # x' in paper
y_2 = kp2_matching[1,:]        # y' in paper
r_2 = (x_2**2 + y_2**2)**0.5   # r' in paper

n_rows = x_1.shape[0]

D1 = np.array([np.multiply(x_1, x_2), np.multiply(x_2, y_1), x_2, np.multiply(y_2, x_1), np.multiply(y_2, y_1), y_2, x_1, y_1, np.ones((n_rows,))]).transpose()
D2 = np.array([np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(x_2, r_1**2), np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(y_2, r_1**2), np.multiply(x_1, r_2**2), np.multiply(y_1, r_2**2), r_1**2+r_2**2]).transpose()
D3 = np.array([np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(r_2**2, r_1**2)]).transpose()



# To solve the polynomial eigen value problem black magic is necessary, https://octave.1599824.n4.nabble.com/Quadratic-Eigen-value-problems-td1626007.html
A = np.matmul(D1.transpose(), D1)
B = np.matmul(D1.transpose(), D2)
C = np.matmul(D1.transpose(), D3)

top = np.concatenate((B, A), axis=1)
bottom = np.concatenate((np.eye(9), np.zeros((9, 9))), axis=1)
side_with_lambda = np.concatenate((top, bottom))

top = np.concatenate((-C, np.zeros((9, 9))), axis=1)
bottom = np.concatenate((np.zeros((9, 9)), np.eye(9)), axis=1)
side_without_lambda = np.concatenate((top, bottom))

res = scipy_linalg.eig(side_without_lambda, b=side_with_lambda)
eig_vals = 1/res[0]
eig_vals_inv = res[0]
eig_vectors = res[1]
non_complex_eig_vals = np.logical_not(np.iscomplex(eig_vals_inv))
non_nan_eig_vals = np.logical_not(np.isnan(eig_vals_inv))
non_inf_eig_vals = np.logical_not(np.isnan(eig_vals_inv))
non_too_big_eig_vals = np.abs(eig_vals) < 10

good_eig_vals = np.logical_and(non_complex_eig_vals, non_nan_eig_vals)
good_eig_vals = np.logical_and(good_eig_vals, non_inf_eig_vals)
good_eig_vals = np.logical_and(good_eig_vals, non_too_big_eig_vals)

good_eig_vectors = eig_vectors[good_eig_vals]
good_eig_vals = eig_vals[good_eig_vals]
good_eig_vals = np.ndarray.astype(good_eig_vals, dtype='float')


# Find the best lambda/undistort value by finding the one that minimizes a modified version of equation 5 in the paper
error_values = []
for lambda_ in good_eig_vals:
    # Undistort the matching keypoints
    z_kp1 = (1 + lambda_*(kp1_matching[0]**2+kp1_matching[1]**2))
    z_kp2 = (1 + lambda_*(kp2_matching[0]**2+kp2_matching[1]**2))

    p_kp1 = np.true_divide(kp1_matching[0:2, :], z_kp1)
    p_kp2 = np.true_divide(kp2_matching[0:2, :], z_kp2)

    H, masked = cv2.findHomography(p_kp1.transpose(), p_kp2.transpose(), cv2.RANSAC, 5.0)
    p_kp1_in_img2 = cv2.perspectiveTransform(p_kp1.transpose().reshape(-1,1,2),H).reshape(-1,2).transpose()

    block_error = p_kp1_in_img2 - p_kp2
    euclidian_error = np.linalg.norm(block_error)
    error_values.append(euclidian_error)

# The best lambda is the one where the error is the smallest
smallest_err_index = np.argmin(np.abs(error_values))
lambda_ = good_eig_vals[smallest_err_index]

# Undistort the image
n_rows = img1.shape[0]
n_cols = img1.shape[1]

# Calculate locations of p in img1 and img2
p_img1 = np.zeros((3, np.prod(img1.shape)))

# create the different x_coordinates and y_coordinates values and normalize them
x_coordinates = (np.arange(n_cols) - x_max/2)/(n_rows+n_cols)
x_coordinates = npm.repmat(x_coordinates, 1, n_rows)
y_coordinates = (np.arange(n_rows) - y_max/2)/(n_rows+n_cols)
y_coordinates = npm.repmat(y_coordinates.reshape(-1,1), 1, n_cols).reshape(-1)

p_img1[0] = x_coordinates
p_img1[1] = y_coordinates

# Undistort the coordinates
p_img1[2] = (1 + lambda_*(p_img1[0]**2+p_img1[1]**2))
p_img1 = np.true_divide(p_img1[0:2, :], p_img1[2])

# Scale back the coordinates
p_img1[0] *= (n_rows+n_cols)
p_img1[1] *= (n_rows+n_cols)

p_img1 = np.round(p_img1)
p_img1 = np.ndarray.astype(p_img1, dtype='int')

# Create a cv img where the undistorted image gets the correct colors, from the distorted image
n_cols_distorted = right_img2.shape[1]

x_max = p_img1[0].max()
x_min = p_img1[0].min()
y_max = p_img1[1].max()
y_min = p_img1[1].min()

n_rows_undistorted = int(y_max-y_min+1)
n_cols_undistorted = int(x_max-x_min+1)
new_img = (p_img1.transpose() - np.array([x_min, y_min], dtype='int')).transpose()


undistorted_image = np.zeros((n_rows_undistorted, n_cols_undistorted, 3))
undistorted_image[new_img[1], new_img[0]] = right_img1.reshape(-1,3)
cv2.imwrite("undistorted.jpg", undistorted_image) 


print('Done!')