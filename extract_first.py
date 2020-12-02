import cv2

vid1dir = 'D:/Image-stitching-project/image-stitching/Than/raw/raw_video/my_video-1.mkv'
vid2dir = 'D:/Image-stitching-project/image-stitching/Than/raw/raw_video/my_video-2.mkv'
vid3dir = 'D:/Image-stitching-project/image-stitching/Than/raw/raw_video/my_video-3.mkv'
vid4dir = 'D:/Image-stitching-project/image-stitching/Than/raw/raw_video/my_video-4.mkv'

out1dir = '1.jpg'
out2dir = '2.jpg'
out3dir = '3.jpg'
out4dir = '4.jpg'


cap1 = cv2.VideoCapture(vid1dir)
cap2 = cv2.VideoCapture(vid2dir)
cap3 = cv2.VideoCapture(vid3dir)
cap4 = cv2.VideoCapture(vid4dir)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
ret3, frame3 = cap3.read()
ret4, frame4 = cap4.read()

cv2.imwrite(out1dir,frame1)
cv2.imwrite(out2dir,frame2)
cv2.imwrite(out3dir,frame3)
cv2.imwrite(out4dir,frame4)

cap1.release()
cap2.release()
cap3.release()
cap4.release()

cv2.destroyAllWindows()
