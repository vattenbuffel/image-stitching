# Undistort images

import cv2
import numpy as np
import scipy.linalg as scipy_linalg
from scipy import optimize

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
scale_percent = 200 # percent of original size
width = int(right_img2.shape[1] * scale_percent / 100)
height = int(right_img2.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
right_img2 = cv2.resize(right_img2, dim, interpolation = cv2.INTER_AREA)
left_img2 = cv2.resize(left_img2, dim, interpolation = cv2.INTER_AREA)

img1 = right_img2
img2 = left_img2
##########################################################

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Only get the matching keypoints
bf = cv2.BFMatcher()
matchesbf = bf.knnMatch(des1,des2, k=2)
# Sort the keypoints so that those with smallest distance between them come at top
matches = sorted(matchesbf, key = lambda x:(x[0].distance - x[1].distance)**2, reverse=True)
matches = np.array(matches)
good_matches = matches[0:8]

index_in_kp1 = [m.queryIdx for m in good_matches[:,0]]
index_in_kp2 = [m.trainIdx for m in good_matches[:,0]]
kp1_matching = np.array([np.array(list(kp1[index].pt)) for index in index_in_kp1]).transpose()
kp2_matching = np.array([np.array(list(kp2[index].pt)) for index in index_in_kp2]).transpose()


# Plot the matching keypoints
matching_keypoints_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, img1, flags=2) # This plots way too many line for some reason
cv2.imwrite("matching_keypoints.jpg", matching_keypoints_img) 


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

# Undistort the image
# I use the biggest eig val here
biggest_eig_val_index = np.argmax(np.abs(good_eig_vals))
lambda_ = good_eig_vals[biggest_eig_val_index]
f = good_eig_vectors[biggest_eig_val_index, 0:9]


# Create an array containing the locations of the points in the first image
n_rows = img1.shape[0]
n_cols = img1.shape[1]
#biggest_dim = np.maximum(n_rows, n_cols)


p1 = np.zeros((2, np.prod(img1.shape)))
p2 = np.zeros((2, np.prod(img1.shape)))
for i in range(n_rows*n_cols): #This for loop is an abomination and should be removed
    x = i%n_cols - x_max/2
    y = i//n_cols - y_max/2

    x/= (n_rows+n_cols)
    y/= (n_rows+n_cols)


    # Undistort
    z = 1/(1+lambda_*(x**2+y**2))
    x*=z*(n_rows+n_cols)
    y*=z*(n_rows+n_cols)

    p1[0,i] = x
    p1[1,i] = y

p2 = p1

p_img1 = p1
p_img2 = p2

p_img1 = euclidian_to_homogenous_2D(p_img1)
p_img2 = euclidian_to_homogenous_2D(p_img2)


# Solve the minimization problem, equation 5 in the paper
F = f.reshape(3,3)
def epipolar_constraint(x):
    x_in_img1 = x[0:3]
    x_in_img2 = x[3:]

    res = np.matmul(F, x_in_img1)
    res = np.matmul(x_in_img2.transpose(), res)
    return res

cons = {'type':'eq', 'fun':epipolar_constraint}

def fun_to_minimize(x, args):
    x_in_img1 = x[0:3]
    x_in_img2 = x[3:]
    p_in_img1 = args[0]
    p_in_img2 = args[1]

    return np.dot(p_in_img2-x_in_img2, p_in_img2-x_in_img2) + np.dot(p_in_img1-x_in_img1, p_in_img1-x_in_img1)

# p_in_img1_min = []
# p_in_img2_min = []
# for i in range(p_img1.shape[1]): # This takes an ungodly amount of time. It is not feasible to do it like this. If I try to bunch it all up into 1 step. Then it would require several TB of memory.
#     x0 = [p_img1[:,i], p_img2[:,i]]
#     sol = optimize.minimize(fun_to_minimize, x0, args=x0, constraints=cons)
#     p_in_img1_min.append(sol.x[0:3])
#     p_in_img2_min.append(sol.x[3:])
p_in_img1_min = p_img1.transpose()
p_in_img2_min = p_img2.transpose()

# Convert the newly found minimized coordinates to euclidian
p_in_img1_min = homogenous_to_euclidian_2D(np.array(p_in_img1_min).transpose())
p_in_img1_min = np.round(p_in_img1_min)
p_in_img1_min = np.ndarray.astype(p_in_img1_min, dtype='int')


# Create a cv img where the undistorted image gets the correct colors, from the distorted image
n_cols_distorted = right_img2.shape[1]

x_max = p_in_img1_min[0].max()
x_min = p_in_img1_min[0].min()
y_max = p_in_img1_min[1].max()
y_min = p_in_img1_min[1].min()

n_rows_undistorted = int(y_max-y_min+1)
n_cols_undistorted = int(x_max-x_min+1)
new_img = (p_in_img1_min.transpose() - np.array([x_min, y_min], dtype='int')).transpose()


undistorted_image = np.zeros((n_rows_undistorted, n_cols_undistorted))
for i,p in enumerate(new_img.transpose()): # should be able to just do undistorted_image[new_img[1], new_img[0]] = right_img2?, maybe it should be left_image?
    x_distorted_image = i%n_cols_distorted
    y_distorted_image = i//n_cols_distorted

    undistorted_image[p[1], p[0]] = right_img2[y_distorted_image,x_distorted_image]

# Average the pixels in the image to get rid of black lines
#undistorted_image = cv2.medianBlur(undistorted_image,5) # The image needs to rgb for this to work

cv2.imwrite("undistorted.jpg", undistorted_image) 
print(undistorted_image.shape)
print('Done!')