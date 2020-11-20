import numpy as np
import cv2

def scoring(dst):
    lower = np.array([15, 120, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower, upper)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, opening_kernel)
    kernel = np.ones((4,4),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)


    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    detect_horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    hori_area = 0.0
    for c in cnts:
        area_temp = cv2.contourArea(c)
        hori_area = hori_area + area_temp
    # print(hori_area)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,60))
    detect_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    verti_area = 0.0
    for c in cnts:
        area_temp = cv2.contourArea(c)
        verti_area = verti_area + area_temp
    # print(verti_area)
    area = hori_area + verti_area
    return area


def resolve_dist(src, k1):
    width = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4, 1))

    # k1 = -1.0e-5  # negative to remove barrel distortion
    # k1 = k1 * 4
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.  # define focal length x
    cam[1, 1] = 10.  # define focal length y

    dst = cv2.undistort(src, cam, distCoeff)
    return dst

src_dir = './img/ze/1.jpg'

src = cv2.imread(src_dir)

init_k1 = -1.0e-5
range_of_multipliers = np.linspace(0,8,20)
range_of_multipliers = range_of_multipliers.tolist()
th = -600
old_score = 0
ind = 0
for i in range_of_multipliers:
    k1 = init_k1*i
    dst = resolve_dist(src,k1)
    score = scoring(dst)
    if score - old_score > th:
        print(i)
        print(score)
        old_score = score
        ind = i
    else:
        print('exit')
        print(i)
        print(score)
        break
k1 = init_k1 * ind
dst = resolve_dist(src,init_k1*4)
# score = scoring(dst)
# print(score)


dst_t = cv2.resize(dst,(int(dst.shape[1]*0.5),int(dst.shape[0]*0.5)))
src_t = cv2.resize(src,(int(src.shape[1]*0.5),int(src.shape[0]*0.5)))
cv2.imwrite('./img/resolved/1r.jpg',dst)
cv2.imshow('src',src_t)
cv2.imshow('dst',dst_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
