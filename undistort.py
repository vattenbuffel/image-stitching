# Undistort images

import cv2
import numpy as np
import scipy.linalg as scipy_linalg

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
kp1_matching = np.array([np.array(list(kp1[index].pt)) for index in index_in_kp1]).transpose()
kp2_matching = np.array([np.array(list(kp2[index].pt)) for index in index_in_kp2]).transpose()

# Test
# kp1_matching = np.array([[1, 2, 3],[4, 5, 6]])
# kp2_matching = np.array([[1, 2, 3],[4,5,6]])
############
# Create D1, D2 and D3
x_1 = kp1_matching[0,:] # x' in paper
y_1 = kp1_matching[1,:] # y' in paper
r_1 = x_1**2 + y_1**2   # r' in paper
x_2 = kp2_matching[0,:] # x in paper
y_2 = kp2_matching[1,:] # y in paper
r_2 = x_2**2 + y_2**2   # r in paper

n_rows = x_1.shape[0]

D1 = np.array([np.multiply(x_1, x_2), np.multiply(x_1, y_2), x_1, np.multiply(y_1, x_2), np.multiply(y_1, y_2), y_1, x_1, y_1, np.ones((n_rows,))]).transpose()
D2 = np.array([np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(x_1, r_2**2), np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(y_1, r_2**2), np.multiply(x_2, r_1**2), np.multiply(y_2, r_1**2), r_2**2+r_1**2]).transpose()
D3 = np.array([np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.zeros((n_rows, )), np.multiply(r_1**2, r_2**2)]).transpose()



# To solve the polynomial eigen value problem black magic is necessary, https://octave.1599824.n4.nabble.com/Quadratic-Eigen-value-problems-td1626007.html
A = np.matmul(D1.transpose(), D1)
B = np.matmul(D1.transpose(), D2)
C = np.matmul(D1.transpose(), D3)

top = np.concatenate((B, A), axis=1)
bottom = np.concatenate((np.eye(9), np.zeros((9, 9))), axis=1)
right_side = np.concatenate((top, bottom))

top = np.concatenate((-C, np.zeros((9, 9))), axis=1)
bottom = np.concatenate((np.zeros((9, 9)), np.eye(9)), axis=1)
left_side = np.concatenate((top, bottom))


res = scipy_linalg.eig(left_side, b=right_side)
eig_vals = res[0]
eig_vals_inv = 1/res[0]
eig_vectors = res[1]
non_complex_eig_vals = np.logical_not(np.iscomplex(eig_vals_inv))
non_nan_eig_vals = np.logical_not(np.isnan(eig_vals_inv))
non_inf_eig_vals = np.logical_not(np.isnan(eig_vals_inv))
non_too_big_eig_vals = np.abs(eig_vals) < 10

good_eig_vals = np.logical_and(non_complex_eig_vals, non_nan_eig_vals)
good_eig_vals = np.logical_and(good_eig_vals, non_inf_eig_vals)
good_eig_vals = np.logical_and(good_eig_vals, non_too_big_eig_vals)

#good_eig_vectors = eig_vectors[good_eig_vals]
good_eig_vals = eig_vals[good_eig_vals]
good_eig_vals = np.ndarray.astype(good_eig_vals, dtype='float')

# Undistort the image
# I use the first eig val here
lambda_ = good_eig_vals[0]

# Create an array containing the locations of the points in the first image
n_rows = right_img2.shape[0]
n_cols = right_img2.shape[1]
#biggest_dim = np.maximum(n_rows, n_cols)

# Save the biggest and smallest coordinates
x_min = 10**10
x_max = -x_min
y_min = x_min
y_max = x_max

new_img = np.zeros((2, np.prod(right_img2.shape)))
for i in range(n_rows*n_cols):
    x = i%n_cols
    y = i//n_cols

    # Undistort
    z = 1/(1+lambda_*(x**2+y**2))
    x*=z 
    y*=z

    x_min = x if x < x_min else x_min
    x_max = x if x > x_max else x_max
    y_min = y if y < y_min else y_min
    y_max = x if y > y_max else y_max

    new_img[0,i] = x
    new_img[1,i] = y

new_img = np.ndarray.astype(new_img, dtype='int')


# Create a cv img where the undistorted image gets the correct colors, from the distorted image
n_cols_distorted = n_cols
n_rows_undistorted = int(y_max-y_min+1)
n_cols_undistorted = int(x_max-x_min+1)
new_img = (new_img.transpose() - np.array([x_min, y_min], dtype='int')).transpose()


undistorted_image = np.zeros((n_rows_undistorted, n_cols_undistorted))
for i,p in enumerate(new_img.transpose()):
    x_distorted_image = i%n_cols_distorted
    y_distorted_image = i//n_cols_distorted

    undistorted_image[p[1], p[0]] = right_img2[y_distorted_image,x_distorted_image]

cv2.imwrite("undistorted.jpg", undistorted_image) 
print('Done!')