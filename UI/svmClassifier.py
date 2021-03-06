#FaceOff Deepfake Detector SVM Classifier
#Lead Developer: Margo Sikes
#Contributions by: Robert Harp

#Partially based on code provided by user GreatGameDota @ Kaggle.com
#Adapted to Python for usage in GUI with SVM
#Original notebook: https://www.kaggle.com/greatgamedota/xception-binary-classifier-inference

import os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import warnings
import re
from sklearn.model_selection import train_test_split
import sklearn.metrics
warnings.filterwarnings("ignore")
from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage import data
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data
from skimage.transform import resize
import pickle
from sklearn.decomposition import PCA
from datetime import datetime

# Store working directory and insert two helper modules into PATH variable.
current_path = os.getcwd()
sys.path.insert(0, current_path+"\\lib\\blazeface-pytorch")
sys.path.insert(0, current_path+"\\lib\\deepfakes-inference-demo")

# Import blazeface from above
import blazeface

# Define classifier for CNN with filenames as input
def classifier(files):
    # Open log file and begin to append
    log = open("deepfakeLog.log", "a+")
    # Sort videos
    test_videos = sorted([x for x in files if x[-4:] == ".mp4"])

    # Check for whether GPU is available.
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up face detection
    facedet = blazeface.BlazeFace().to(gpu)
    facedet.load_weights(current_path+"\\lib\\blazeface-pytorch\\blazeface.pth")
    facedet.load_anchors(current_path+"\\lib\\blazeface-pytorch\\anchors.npy")
    _ = facedet.train(False)

    from helpers.read_video_1 import VideoReader
    from helpers.face_extract_1 import FaceExtractor
    # Determine how many frames are analyzed per video
    frames_per_video = 20
    # Set up video reader and face extractor helpers
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)

    # Size of images
    input_size = 150

    # Resize image
    def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
        h, w = img.shape[:2]
        if w > h:
            h = h * size // w
            w = size
        else:
            w = w * size // h
            h = size

        resized = cv2.resize(img, (w, h), interpolation=resample)
        return resized

    # Turn image into correct size
    def make_square_image(img):
        h, w = img.shape[:2]
        size = max(h, w)
        t = 0
        b = size - h
        l = 0
        r = size - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    # Prediction
    def predict_on_video(video_path, batch_size):
        try:
            facesList = []
            # Find the faces for N frames in the video.
            faces = face_extractor.process_video(video_path)

            # Only look at one face per frame.
            face_extractor.keep_only_best_face(faces)

            # facesList = faces
            
            if len(faces) > 0:
                # NOTE: When running on the CPU, the batch size must be fixed
                # or else memory usage will blow up. (Bug in PyTorch?)
                x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

                # If we found any faces, prepare them for the model.
                n = 0
                frameNum = []
                for frame_data in faces:
                    frameNum.append(frame_data["frame_idx"])
                    for face in frame_data["faces"]:
                        # Resize to the model's required input size.
                        # We keep the aspect ratio intact and add zero
                        # padding if necessary.
                        facesList.append(face)                    
                        resized_face = isotropically_resize_image(face, input_size)
                        resized_face = make_square_image(resized_face)

                        if n < batch_size:
                            x[n] = resized_face
                            n += 1
                        else:
                            print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                        
                        # Test time augmentation: horizontal flips.
                        # TODO: not sure yet if this helps or not
                        # x[n] = cv2.flip(resized_face, 1)
                        # n += 1
                
                # Contributed by Robert Harp (137-158)
                # Remove alpha dimension from array
                x = x[:, :, :, 0]
                magnitude_spectrum = []
                ## Create magnitude spectrum
                for img in x:
                    img = resize(img,(72,72))
                    f = (np.fft.rfft2(img))
                    fshift = np.fft.fftshift(f)
                    magnitude_spectrum.append(20*np.log(np.abs(fshift)))
                ## Convert magnitude spectrum to np array
                magnitude_spectrum2 = np.asarray(magnitude_spectrum)
                ## Reshape array
                magnitude_spectrum2 = magnitude_spectrum2.reshape(magnitude_spectrum2.shape[0],(magnitude_spectrum2.shape[1]*magnitude_spectrum2.shape[2]))
                ## Square array
                magnitude_spectrum2 = np.square(magnitude_spectrum2)
                ## Pass into PCA
                x2 = PCA(n_components=2).fit_transform(magnitude_spectrum2.tolist())
                ## Load pickled model
                clf = pickle.load(open(current_path+"\\lib\\svmmodel.sav", 'rb'))
                ## Predict on x2 values and output predctions
                y_pred = clf.predict(x2)

                # Log prediction with name
                video_name = os.path.basename(video_path)
                time = datetime.now()
                log.write(time.strftime("%d/%m/%Y %H:%M:%S") + " -- Entered {0} into SVM algorithm\n".format(video_name))
                log.write("\tPrediction: " + str(y_pred[:n].mean().item()) + "\n\n")

                # Print and return prediction
                print(y_pred[:n].mean().item())
                return y_pred[:n].mean().item()

        # Error handling
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            lineno = tb.tb_lineno
            time = datetime.now()
            log.write("\n%s -- Prediction error on video %s: %s\n\n" % (time.strftime("%d/%m/%Y %H:%M:%S"), video_path, str(e)))
            print("Prediction error on video %s: %s %s" % (video_path, lineno, str(e)))
        # Return 0.5 if error (0.5 == Not Sure)
        return 0.5

    from concurrent.futures import ThreadPoolExecutor

    # Predict on multiple videos
    def predict_on_video_set(videos, num_workers):
        def process_file(i):
            y_pred = predict_on_video(filename, batch_size=frames_per_video)
            return y_pred

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            predictions = ex.map(process_file, range(len(videos)))

        return list(predictions)

    # If True, will return a prediction of classification length
    # Ultimately was not able to use this in GUI due to threading difficulty
    speed_test = False

    if speed_test:
        start_time = time.time()
        speedtest_videos = test_videos[:5]
        predictions = predict_on_video_set(speedtest_videos, num_workers=4)
        elapsed = time.time() - start_time
        print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))

    predictions = predict_on_video_set(test_videos, num_workers=4)
    log.close()
    return predictions