import cv2
from getFeaturePoints import *
from helperFunctions import *
import matplotlib.pyplot as plt
import time
# Connect to the videos
vid1 = cv2.VideoCapture('videos/vid1.mkv')
vid2 = cv2.VideoCapture('videos/vid2.mkv')
vid3 = cv2.VideoCapture('videos/vid3.mkv')
vid4 = cv2.VideoCapture('videos/vid4.mkv')

# Read the first frame
ret1, frame1 = vid1.read()
ret2, frame2 = vid2.read()
ret3, frame3 = vid3.read()
ret4, frame4 = vid4.read()

frames = []
frames.append(frame1)
frames.append(frame2)
frames.append(frame3)
frames.append(frame4)
# Get featurepoints
kp, des = getFeaturePoints(frames)

# Undistort frames
for frame in frames:
    #plt.imshow(undistort(kp[0], kp[1], frame)), plt.show()
    pass


# Perfrom mapping
good = matchFeatures(des[0], des[1])


# Estimate the homographies

ret = [ret1, ret2, ret3, ret4]
while ret:
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()
    ret3, frame3 = vid3.read()
    ret4, frame4 = vid4.read()
    ret = [ret1, ret2, ret3, ret4]

    cv2.imshow('Stream',frame1)

    # stitch and display results

