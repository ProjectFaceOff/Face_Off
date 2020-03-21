####################################################################################################################
# Title: Generate Dataset
# Purpose: Detects and crops faces from .mp4 videos (currently set up for Kaggle dataset)
# Author: Shangqiu Li (original)
# Source: https://www.kaggle.com/unkownhihi/starter-kernel-with-cnn-model-ll-lb-0-69235
# Version: 1.0
####################################################################################################################
# Date           Author      Comments
# XX/XX/XX       XXXXXXX     XXXXXXXXXXXXXXXXXXX
# 03/21/20       msikes      Initial file creation for starting, added comments to file for readability
#
####################################################################################################################
# NOTE List:
#               - Run this script with the following command 'python dataset_generator.py XX' where XX is the
#                   name of the folder you want to convert into cropped images
#               - Make sure this script is located in the same folder as your dataset. Otherwise you need to modify.
####################################################################################################################
# TODO List: 
#               - Implement BlazeFace PyTorch for comparison (BlazeFace is supposed to run really fast & returns
#                   PyTorch tensors which may help with features)
#               - Modify number of frames captured
####################################################################################################################

from mtcnn import MTCNN
import tqdm
import datetime
import smtplib
import os
import cv2
import numpy as np
import sys
import shutil

# Run python script with argument detailing the desired folder
# i.e. python dataset_generator.py 0
# runs the script on the folder dfdc_train_part_0
d_num=sys.argv[1]

# Concatenates 0 if necessary due to naming structure of Kaggle
if len(d_num)==1:
    a_num = d_num
    d_num='0'+d_num
else:
    a_num=d_num

# Creates MTCNN object
detector = MTCNN()

# Function that detects a face from given image.
def detect_face(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    final = []
    # Detects faces and returns a list of JSON objects
    detected_faces_raw = detector.detect_faces(img)
    # If empty array returned, no faces were found
    if detected_faces_raw==[]:
        print('No faces found!')
        return []
    # Add detected bounding box to list
    confidences=[]
    for n in detected_faces_raw:
        x,y,w,h=n['box']
        final.append([x,y,w,h])
        confidences.append(n['confidence'])
    # If the highest confidence of detected faces is below 0.7, return no face
    if max(confidences)<0.7:
        return []
    max_conf_coord=final[confidences.index(max(confidences))]
    # Return highest confidence face detected
    return max_conf_coord

# Crops image based on x,y,w,h values from face detection
def crop(img,x,y,w,h):
    # Modifies bounding box for larger margin
    x-=40
    y-=40
    w+=80
    h+=80
    if x<0:
        x=0
    if y<=0:
        y=0
    # Returns cropped and downsized image of 256x256
    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],(256,256)),cv2.COLOR_BGR2RGB)

# Detects faces in each frame of video
def detect_video(video):
    # Turn video into cv2 VideoCapture object
    v_cap = cv2.VideoCapture(video)
    # Sets location in video based on current frame number
    # 1 == Current position of file in milliseconds
    v_cap.set(1, NUM_FRAME)
    success, vframe = v_cap.read()
    #vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) # This code seems redundant b/c cvtColor is in detect_face
    # Send frame into detect_face function
    bounding_box=detect_face(vframe)
    # If no bounding box was found, check the next frames
    if bounding_box==[]:
        count=0
        current=NUM_FRAME
        while bounding_box==[] and count<MAX_SKIP:
            current+=1
            v_cap.set(1,current)
            success, vframe = v_cap.read()
            #vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) # This code seems redundant b/c cvtColor is in detect_face
            bounding_box=detect_face(vframe)
            count+=1
        if bounding_box==[]:
            print('No faces found in video.')
            return None
    x,y,w,h=bounding_box
    v_cap.release()
    return crop(vframe,x,y,w,h)

# Create list of video files & create directory to store cropped images
test_dir = './dfdc_train_part_' + a_num + '/'
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
os.makedirs('./DeepFake' + d_num,exist_ok=True)

# Set max frames to skip and number of frames
# Initially set to
# MAX_SKIP=10
# NUM_FRAME=150
# According to author, frame number starts at 150 because it is middle of the video where results are more reliable
MAX_SKIP=10
NUM_FRAME=150
count=0
for video in tqdm.tqdm(test_video_files):
    try:
        # Copy metadata into current directory with folder number in its name
        if video=='./dfdc_train_part_'+a_num+'/metadata.json':
            shutil.copyfile(video,'./metadata'+str(a_num)+'.json')
        # Run video through script
        img_file=detect_video(video)
        # Remove video after completion
        os.remove(video)
        # If no faces were found, go to next video
        if img_file is None:
            count+=1
            continue
        # Write cropped image to new folder
        cv2.imwrite('./DeepFake'+d_num+'/'+video.replace('.mp4','').replace(test_dir,'')+'.jpg',img_file)
    except Exception as err:
      print(err)