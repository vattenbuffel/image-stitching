import numpy as np
import cv2
from matplotlib import pyplot as plt
from undistort import Undistorter

# Start by creating the undistorter and initializing it
right = cv2.imread("right.JPG")
left = cv2.imread("left.JPG")
height, width, _ = right.shape
size = (width, height)

undistorter = Undistorter(right, left)

# This reads the video
vid1 = cv2.VideoCapture('./videos/vid1.mkv')

# This will create the video
out = cv2.VideoWriter('undistorted_vid1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

counter = 0
# Loop over the video and undistort it
got_frame = True
while True:
    got_frame, frame = vid1.read()

    if got_frame:
       undistorted_image = undistorter.undistort(frame)
       out.write(undistorted_image)

    counter = counter + 1
    if counter > 2000:
        break

# Write the actual video
out.release()
