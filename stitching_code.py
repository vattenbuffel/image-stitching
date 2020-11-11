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
(max_y,max_x,z) = right_img1.shape
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
(y_max,x_max,z) = right_img1.shape
corners = np.array([[0, 0, x_max, x_max], [0, y_max, y_max, 0], [1,1,1,1]])

# Convert homogenous coordinates to euclidian
def homogenous_to_euclidian(x):
    height, _  = x.shape
    return np.true_divide(x[0:height-1, :], x[-1, :])

# Convert euclidian coordinates to homogenous
def euclidian_to_homogenous(x):
    _, width  = x.shape
    return np.vstack([x, np.ones((1, width))])

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

homographies_to_keep_similarity = []
for i in range(homographies.shape[0]):
    H = homographies[i]
    x0 = [1, 1, 1, 1, 1]
    H_s = optimize.fmin(difference_between_similarity_and_H, x0, disp = False, args = (H,))
    
    difference_normalized = difference_between_similarity_and_H(H_s, H) / (max_x*max_y) # It should be normalized by image size. It's a bit unclear what image size is though...
    homographies_to_keep_similarity.append( difference_normalized< threshold)
        
homographies = homographies[homographies_to_keep_similarity,:,:]


# Remove homographies which the magnitude of the scaling parameters is bigger than a threshold
# I'm not sure if this is correct or not
magnitude_threshold = 1.2 # I have no idea what this should be
scales = np.matmul(homographies, corners[:,2])
homographies_to_keep_scale = np.absolute(scales[:, -1]) < magnitude_threshold # Shouldn't there be a minimum threshold as well?
homographies = homographies[homographies_to_keep_scale]
    

 
# Remove homographies which are too close to identity
# Do this by checking the overlap between I*H and I
# Do that like this https://stackoverflow.com/questions/42402303/opencv-determine-area-of-intersect-overlap
overlap_threshold = 0.95
homographies_to_keep_identity = []

(y_max,x_max,z) = right_img1.shape
corners = np.array([[0, 0, x_max, x_max], [0, y_max, y_max, 0], [1,1,1,1]])
corner_locations = []

for H in homographies:
    corners_H = np.matmul(H, corners)
    corners_H = homogenous_to_euclidian(corners_H)

    # Find the corners of the warped image
    x_min_H = np.min(corners_H[0, :])
    x_max_H = np.max(corners_H[0,:])
    y_min_H = np.min(corners_H[1, :])
    y_max_H = np.max(corners_H[1,:])

    # Save the corners for future calculations
    corner_locations.append([[x_min_H, x_min_H, x_max_H, x_max_H], [y_min_H, y_max_H, y_max_H, y_min_H]])

    # Find the dimensions of the new image
    x_min_res = np.minimum(0, x_min_H)
    x_max_res = np.maximum(x_max, x_max_H)
    y_min_res = np.minimum(0, y_min_H)
    y_max_res = np.maximum(y_max, y_max_H)

    res_img_width = x_max_res-x_min_res
    res_img_height = y_max_res-y_min_res

    # Handle the case where the homography maps to negative coordinates
    # https://stackoverflow.com/questions/6087241/opencv-warpperspective
    fix_matrix = np.identity(3)
    x_origin = 0
    y_origin = 0
    if x_min_H < 0:
        fix_matrix[0,2] = -x_min_H
        x_origin = int(-x_min_H)
    if y_min_H < 0:
        fix_matrix[1,2] = -y_min_H
        y_origin = int(-y_min_H)
    H = np.matmul(H, fix_matrix)
        
    # Augment the image width and height to also incorporate the offset introduced above
    res_img_width = np.ceil(res_img_width+np.abs(x_min_H * int(x_min_H<0))).astype('uint')
    res_img_height = np.ceil(res_img_height+np.abs(y_min_H* int(y_min_H<0))).astype('uint')

    # Warp the image
    warped_image = cv2.warpPerspective(right_img1, H, (res_img_width, res_img_height))
    warped_image[warped_image != 0] = 1
    warped_image[y_origin:y_origin+y_max, x_origin:x_origin+x_max] += 1
    #cv2.imwrite("overlap.jpg", warped_image*127) # Not needed. Just shows the intersection result
    intersected_area = np.sum(warped_image == 2)
    total_area = np.sum(warped_image == 1) + intersected_area

    homographies_to_keep_identity.append(intersected_area/total_area < overlap_threshold)

homographies = homographies[homographies_to_keep_identity]

# Remove homographies where either diagonal is shorter than half the length of the diagonal of the original image
original_diagonal = np.linalg.norm(right_img2.shape)
corner_locations = np.array(corner_locations)[homographies_to_keep_identity]# Only continue with the corners which have a connected homography

upper_left_to_lower_right = np.subtract(corner_locations[:, :, 2], corner_locations[:, :, 0])
lower_left_to_upper_right = np.subtract(corner_locations[:, :, 3], corner_locations[:, :, 1])
diagonal_distances = np.array([np.linalg.norm(upper_left_to_lower_right, axis=1), np.linalg.norm(lower_left_to_upper_right, axis=1)])
homographies_to_keep_diagonal = diagonal_distances > (original_diagonal/2)
homographies_to_keep_diagonal = np.logical_or(homographies_to_keep_diagonal[0,:], homographies_to_keep_diagonal[1,:])
homographies = homographies[homographies_to_keep_diagonal]

# Remove duplicates
# I'm not entierly sure what the reprojection error is...
# Calculate D_0
reprojection_error_threshold = 10 # No idea what this should be
inclusion_radius = 25 # No idea what this should be
cosine_similarity_threshold = 0.6 # No idea what this should be
max_number_of_homograhpies = 10000 # No idea what this should be

homo_kp1 = euclidian_to_homogenous(np.transpose(kp1_matching))
kp1_in_img2 = np.matmul(homographies, homo_kp1)
kp1_in_img2 = np.array([np.true_divide(kp1_in_img2[i,0:2,:], kp1_in_img2[i,-1,:]) for i in range(kp1_in_img2.shape[0])])
normal_format_kp2 = np.transpose(kp2_matching)
reprojection_error_block_distance = np.subtract(kp1_in_img2, normal_format_kp2)
reprojection_error = np.linalg.norm(reprojection_error_block_distance, axis=1)
matching_key_point_to_D_0 = reprojection_error < reprojection_error_threshold

D_t_index = matching_key_point_to_D_0
done = False

while not done:
    starting_number_of_elements = np.sum(D_t_index)

    for i in range(D_t_index.shape[0]):
        points_in_D_t = kp1_matching[np.argwhere(D_t_index[i])].reshape(-1,2)
        points_not_in_D_t = kp1_matching[np.argwhere(np.logical_not(D_t_index[i]))].reshape(-1,2)

        # If all points already in D_t[i] then move on
        if points_not_in_D_t.size == 0:
            continue

        # The following code is really, really hard to understand
        matrix_with_points_on_row = np.repeat(kp1_matching[np.argwhere(D_t_index[i])], points_not_in_D_t.shape[0], axis=0).reshape(-1, points_not_in_D_t.shape[0]*2)
        matrix_with_block_distances = np.subtract(matrix_with_points_on_row, points_not_in_D_t.reshape(1,-1))
        vectors_with_block_distance = np.array(np.hsplit(matrix_with_block_distances, points_not_in_D_t.shape[0]))#.reshape(points_not_in_D_t.shape[0],points_in_D_t.shape[0],-1)
        distance = np.linalg.norm(vectors_with_block_distance, axis = 2)
        matching_points_to_add_to_D_t = distance < inclusion_radius
        points_inside_r = np.sum(matching_points_to_add_to_D_t, axis=1)>0
        
        tested_points = np.argwhere(np.logical_not(D_t_index[i])).reshape(-1)
        D_t_index[i][tested_points] =  np.logical_or(D_t_index[i][tested_points], points_inside_r)

    done = starting_number_of_elements == np.sum(D_t_index)

# Create indicator vectors
# The indicator vectors are just D_t_index

# Save homographies based on |D_t| and cosine distance
D_t_euclidian_distance_block = [np.subtract(kp1_matching[x], kp2_matching[x]) for x in D_t_index]
D_t_euclidian_distance = [np.linalg.norm(D_t_euclidian_distance_block[i], axis=1) for i in range(D_t_index.shape[0])]
D_t_euclidian_distance_sum = np.array([np.sum(distances) for distances in D_t_euclidian_distance])
D_t_euclidian_distance_ordered = np.flip(np.sort(D_t_euclidian_distance_sum))

def cosine_distance(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

homographies_to_keep_distance = [np.where(D_t_euclidian_distance_sum == D_t_euclidian_distance_ordered[0])[0][0]] # Always save the homography with the biggest |D_t|
for distance in D_t_euclidian_distance_ordered:
    contender_index = np.where(D_t_euclidian_distance_sum == distance)[0][0]
    
    # Check the cosine distance
    contender_indicator_vector = D_t_index[contender_index]
    add = True
    for already_saved in homographies_to_keep_distance:
        already_saved_indicator_vector = D_t_index[already_saved]
        if cosine_distance(contender_indicator_vector, already_saved_indicator_vector) > cosine_similarity_threshold:
            add = False
            break
    
    if add: homographies_to_keep_distance.append(contender_index)

homographies = homographies[homographies_to_keep_distance]


print("Done!")