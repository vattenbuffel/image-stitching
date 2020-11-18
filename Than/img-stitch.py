import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

## Functions ##
def ransac_H(X1,X2,good,iteration):
    # By using RANSAC method estimating homography matrix
    # Takes the match locations as X1 and X2 from image one and two respectively
    # X1 and X2 are in the form of [x_coord,y_coord,1] and matched point long
    # Code looks for how many inliers are inside our threshold whicH I set as 36
    # good means good_matches, and iteration here sets how many times will ransac iterate to find best H matrix
    H=np.ones((3,3,iteration))
    inlier=np.zeros(iteration)
    if len(good)>min_match:
        for t in range(iteration):
            subset = random.sample(range(len(good)), 4)
            A = np.zeros((12,9))
            for j,(i) in enumerate(subset):
                A[3*j:3*(j+1),:] = np.kron(X1[:,i], skewsym(X2[:,i]))
            [U,S,V] =np.linalg.svd(A,full_matrices=True)
            H[:,:,t] =np.reshape(V[8,:],(3,3), order='F')
            # calculate inliers with each H matrix
            X2_ = np.matmul(H[:,:,t] , X1)
            du = X2_[0,:]/X2_[2,:] - X2[0,:]/X2[2,:]
            dv = X2_[1,:]/X2_[2,:] - X2[1,:]/X2[2,:]
            projected=np.square(du) + np.square(dv)
            inlier[t]=sum(i < 36 for i in projected)
        Hbest=H[:,:,np.argmax(inlier)]
        Hbest/=Hbest[2,2]
        return Hbest
    else:
        print('Error: Matching point is not enough')


def image_warp(im1, im2, H):
    # Takes 2 images and their homography matrix and stitches them
    # First canvas is calculated
    w1, h1, dum = im2.shape
    w2, h2, dum = im1.shape

    box1 = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    box2_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
    box2 = cv2.perspectiveTransform(box2_temp, H)

    canvas_dims = np.concatenate((box1, box2), axis=0)

    [x_min, y_min] = np.int32(canvas_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(canvas_dims.max(axis=0).ravel() + 0.5)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    stitched = cv2.warpPerspective(im1, T.dot(H), (x_max - x_min, y_max - y_min))

    A = (im2).astype(float) + (stitched[-y_min:w1 - y_min, -x_min:h1 - x_min]).astype(float)
    B = gridd(stitched[-y_min:w1 - y_min, -x_min:h1 - x_min], im2)
    stitched[-y_min:w1 - y_min, -x_min:h1 - x_min] = np.divide(A, B)

    return stitched


def skewsym(v):
    # Takes a vector and converst it to skew-symmetric matrix
    A=np.zeros((3,3))
    A[0,1]=-v[2]
    A[0,2]=v[1]
    A[1,2]=-v[0]
    A-=np.transpose(A)
    return A

def gridd(X,Y):
    gridd1=(X>0).astype(int)
    gridd2=(Y>0).astype(int)
    gridd3=gridd1+gridd2
    asd=(gridd3==0).astype(int)
    res=gridd3+asd
    return res

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# img_dir = ['./img/sticky/1.jpg','./img/sticky/2.jpg','./img/sticky/3.jpg','./img/sticky/4.jpg']
# img_dir = ['./img/ze/1.jpg','./img/ze/2.jpg','./img/ze/3.jpg','./img/ze/4.jpg']
img_dir = ['./img/resolved/1r.jpg','./img/resolved/2r.jpg','./img/resolved/3r.jpg','./img/resolved/4r.jpg']
img_list = []
min_match = 4

for dir in img_dir:
    img = cv2.imread(dir)
    img_list.append(img)

im2 = img_list[0]
for img_n in range(len(img_list)-1):
    im1 = img_list[img_n+1]


    img1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    (kps1, descs1) = sift.detectAndCompute(img1,None)
    (kps2, descs2) = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1,descs2, k=2)

    good = []
    if len(matches) > 40:
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append([m])

    if len(good) < 30:
        print("It couldn't find enough match points for next image")
        print("%d images are stitched together" % (img_n + 1))
        break

    X1 = np.float32([kps1[m[0].queryIdx].pt for m in good]).reshape(-1, 2)
    X2 = np.float32([kps2[m[0].trainIdx].pt for m in good]).reshape(-1, 2)
    X1 = np.concatenate((X1, np.ones((len(X1), 1))), axis=1)
    X2 = np.concatenate((X2, np.ones((len(X2), 1))), axis=1)

    X1 = np.transpose(X1)
    X2 = np.transpose(X2)

    H = ransac_H(X1, X2, good, 100)
    stitched = image_warp(im1, im2, H)

    # plt.imshow(stitched),plt.show()

    if stitched.shape[1] > 2400:
        stitched = cv2.resize(stitched, (0, 0), fx=0.9, fy=0.9)

    im2=stitched
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)), plt.show()