## Import packages
import cv2
import os
import time
import numpy as np

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

## Extract lambda from images in frames: input = frames, output = lambda
####### to-be-done

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
    ####### to-be-done

    ## Solve black line: input = undistorted_frames, output = solved_frames
    ####### to-be-done

    ## Stitch + Blend: input = solved_frames, output = final_frames
    ####### to-be-done


    ## Just a dummy function (concat 4 frames tgt)
    frames_resize = [cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                      for frame in frames]
    final_frame = concat_tile([[frames_resize[0],frames_resize[1]],[frames_resize[2],frames_resize[3]]])

    ## Write final_frame into temp folder
    # The reason is this could save RAM from being full.
    # Pad 0 in front of the number. Work only until 10 million frames.
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






