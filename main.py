## Import packages
import cv2
import os
import time
import numpy as np
import cv2.aruco as aruco
from undistort import Undistorter
from stitchAndBlend import Image_Stitching

##############################################################
## Here is to place functions
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def empthy_folder(folder_path):
    temp_names = os.listdir(folder_path)
    for name in temp_names:
        os.remove(folder_path + '/' + name)
    return None

def extract_frames(caps):
    rets = []
    frames = []
    for cap in caps:
        ret_temp, frame_temp = cap.read()
        rets.append(ret_temp)
        frames.append(frame_temp)
    if not all(rets):
        print("Can't receive frames...")
        if not not_first_frame:
            print("Something wrong...")
            exit()
    return rets, frames

def get_FPS(caps):
    fpss = []
    for cap in caps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fpss.append(fps)
    fps = fpss[0]
    fpss = np.array(fpss)
    chck = fpss == fps
    if all(chck):
        return fps
    else:
        print('Videos are not at the same FPS')
        exit()

def create_vid(temp_dir, FPS):
    frame_names = os.listdir(temp_dir)
    frame_temp = cv2.imread(temp_dir + '/' + frame_names[0])
    dimensions = frame_temp.shape
    h = dimensions[0]
    w = dimensions[1]
    frame_size = (w,h)
    out = cv2.VideoWriter('stitched.avi',cv2.VideoWriter_fourcc(*'DIVX'),FPS,frame_size)

    for name in frame_names:
        frame_temp = cv2.imread(temp_dir + '/' + name)
        out.write(frame_temp)
    out.release()
    return None

def create_undistorters(frames):
    undistorters = []
    n_frames = len(frames)
    for i in range(n_frames):
        if i == 0:
            left = frames[i]
            right = frames[-1]
        else:
            left = frames[i]
            right = frames[0]
        undistorter_temp = Undistorter(right,left)
        undistorters.append(undistorter_temp)
    return undistorters

def undistort_frames(frames,undistorters):
    n_frames = len(frames)
    undis_frames = []
    for i in range(n_frames):
        frame = frames[i]
        undistorter = undistorters[i]
        undistorted_temp = undistorter.undistort(frame)
        undis_frames.append(undistorted_temp)
    return undis_frames


def stitch_and_blend(frames):
    stitcher = Image_Stitching()
    n_frames = len(frames)
    frame_temp1 = frames[0]
    frame_temp2 = frames[1]
    final_frame = stitcher.blending(frame_temp1,frame_temp2)
    for i in range(n_frames-2):
        frame_temp = frames[i+2]
        final_frame = stitcher.blending(final_frame,frame_temp)

    return final_frame

##############################################################
tic = time.perf_counter()
print('Start...')
## Select raw video folder
# Each video must be sync (Start at the same time) and have the same length.
videos_dir = './raw_video'
temp_dir = './temp'
## Get file list from folder
vid_names = os.listdir(videos_dir)

## Get rid off readme.txt
for name in vid_names:
    if name.endswith(".txt"):
        vid_names.remove(name)

## Define video capture for all videos
caps = []
for name in vid_names:
    caps.append(cv2.VideoCapture(videos_dir + '/' + name))

fps = get_FPS(caps)
print('FPS :' + str(fps))

## Extract first frame of each video into frames
not_first_frame = False
rets,frames = extract_frames(caps)

## Extract lambda from images in frames: input = frames, output = undistorters
undistorters = create_undistorters(frames)

## Create folder to store stitched frames
os.mkdir(temp_dir)


## Start while loop (Start processing)
print('Processing frames...')
while all(rets):

    ## Extract frames from video
    ## Noted: For the first frame, will use the one extracted for extract lambda
    if not_first_frame:
        frame_id = frame_id + 1
        rets, frames = extract_frames(caps)
        if not all(rets):
            print("Last frame...")
            break
    else:
        # Use the same frames as in extract lambda
        frame_id = 0
        not_first_frame = True

    ## Undistored frames: input = frames, output = undistored_frames
    undistorted_frames = undistort_frames(frames,undistorters)

    ## Stitch + Blend: input = undistored_frames, output = final_frames
    # Still got some error about H somewhere.
    final_frame = stitch_and_blend(undistorted_frames)

    ## Just a dummy function (concat 4 frames tgt)
    # frames_resize = [cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #                   for frame in undistorted_frames]
    # final_frame = concat_tile([[frames_resize[0],frames_resize[1]],[frames_resize[2],frames_resize[3]]])

    ## Write final_frame into temp folder
    # The reason is this could save RAM from being full.
    # Pad 0 in front of the number. Work only until 10 million frames.

    ## If the result is too big>> Add resize
    frame_id_pad = "{0:08}".format(frame_id)
    cv2.imwrite(temp_dir+'/temp'+ frame_id_pad + '.jpeg', final_frame)



## Create video from frames in temp folder: input = temp dir,FPS, output = None (but write video into stitched.avi)
print('Writing video...')
create_vid(temp_dir, fps)

## Remove temp folder
empthy_folder(temp_dir)
os.removedirs(temp_dir)

## Release all caps
for cap in caps:
    cap.release()
toc = time.perf_counter()
print(f"Total processing time: {toc - tic:0.3f} seconds")
print("Done...")







