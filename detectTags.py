import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt


def detectTags(image):
    #image = cv2.imread("v_img/4.jpg")
    w = image.shape[0]
    h = image.shape[1]

    aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_1000)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters)

    mask = np.zeros([w,h, 3], dtype="uint8")
    for marker in rejectedImgPoints:
        pt1 = marker[0][0]
        pt2 = marker[0][1]
        pt3 = marker[0][2]
        pt4 = marker[0][3]

        yValues = [pt1[0], pt2[0], pt3[0], pt4[0]]
        xValues = [pt1[1], pt2[1], pt3[1], pt4[1]]
        mask[int(min(xValues)):int(max(xValues)), int(min(yValues)):int(max(yValues)), :] = -1

    maskedImage = cv2.bitwise_and(image, mask)
    """fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.imshow(image)
    ax2.imshow(maskedImage)
    plt.show()"""
    return maskedImage
