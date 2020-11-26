import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt


img_path = '/Users/hadleybai/Downloads/design_project/v_img_result'
img1 = cv2.imread(img_path + '/calibresult_1.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(img_path + '/calibresult_3.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# img_path = '/Users/hadleybai/Downloads/design_project'
# img1 = cv2.imread(img_path + '/stitch_result_21.png')
# img2 = cv2.imread(img_path + '/stitch_result_43.png')

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
raw_matches = bf.knnMatch(des1, des2, k=2)

ratio = 0.75

matches = []
for m in raw_matches:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))

min_match = 4
# reprojThresh = 4.0
reprojThresh = 4.0

kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

if len(matches) > min_match:
    pts1 = np.float32([kp1[i] for (_, i) in matches])
    pts2 = np.float32([kp2[i] for (i, _) in matches])
    (H, status) = cv2.findHomography(pts2, pts1, cv2.RANSAC, reprojThresh)
    # transform, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=reprojThresh)

# result = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
# result = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
# result[0:img1.shape[0], 0:img1.shape[1]] = img1

result = cv2.warpPerspective(img2, H, (img1.shape[1],  img2.shape[0] + img1.shape[0]))
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# img_overlay = cv2.warpAffine(img2, transform, (img1.shape[1], img1.shape[0]))
# img_overlay = cv2.addWeighted(img1, 0.5, img_overlay, 0.5, gamma=0.0)

# plt.imshow(result), plt.show()
cv2.imwrite('stitch_result_13.png', result)
# cv2.imwrite('stitch_result.png', img_overlay)
print('finish')
