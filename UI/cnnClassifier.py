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
from datetime import datetime
warnings.filterwarnings("ignore")

# Store working directory and insert two helper modules into PATH variable.
current_path = os.getcwd()
sys.path.insert(0, current_path+"\\lib\\blazeface-pytorch")
sys.path.insert(0, current_path+"\\lib\\deepfakes-inference-demo")

import blazeface

def classifier(files):
    f = open("deepfakeLog.log", "a+")
    test_videos = sorted([x for x in files if x[-4:] == ".mp4"])

    # Check for whether GPU is available.
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    facedet = blazeface.BlazeFace().to(gpu)
    facedet.load_weights(current_path+"\\lib\\blazeface-pytorch\\blazeface.pth")
    facedet.load_anchors(current_path+"\\lib\\blazeface-pytorch\\anchors.npy")
    _ = facedet.train(False)

    from helpers.read_video_1 import VideoReader
    from helpers.face_extract_1 import FaceExtractor

    frames_per_video = 20

    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)

    input_size = 150

    from torchvision.transforms import Normalize

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean, std)

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


    def make_square_image(img):
        h, w = img.shape[:2]
        size = max(h, w)
        t = 0
        b = size - h
        l = 0
        r = size - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    from pytorchcv.model_provider import get_model
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

    class Head(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super(Head, self).__init__()
            
            self.f = nn.Flatten()
            self.l = nn.Linear(in_f, 512)
            self.d = nn.Dropout(0.5)
            self.o = nn.Linear(512, out_f)
            self.b1 = nn.BatchNorm1d(in_f)
            self.b2 = nn.BatchNorm1d(512)
            self.r = nn.ReLU()

        def forward(self, x):
            x = self.f(x)
            x = self.b1(x)
            x = self.d(x)

            x = self.l(x)
            x = self.r(x)
            x = self.b2(x)
            x = self.d(x)

            out = self.o(x)
            return out

    class FCN(torch.nn.Module):
        def __init__(self, base, in_f):
            super(FCN, self).__init__()
            self.base = base
            self.h1 = Head(in_f, 1)
        
        def forward(self, x):
            x = self.base(x)
            return self.h1(x)

    net = []
    model = FCN(model, 2048)
    if (gpu == "gpu"):
        model = model.cuda()
    model.load_state_dict(torch.load(current_path+'\\lib\\model.pth', map_location=torch.device(gpu)))
    net.append(model)

    def predict_on_video(video_path, batch_size):
        try:
            facesList = []
            # Find the faces for N frames in the video.
            faces = face_extractor.process_video(video_path)

            # Only look at one face per frame.
            face_extractor.keep_only_best_face(faces)

            #facesList = faces
            
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
                        #x[n] = cv2.flip(resized_face, 1)
                        #n += 1

                if n > 0:
                    x = torch.tensor(x, device=gpu).float()

                    # Preprocess the images.
                    x = x.permute((0, 3, 1, 2))

                    for i in range(len(x)):
                        x[i] = normalize_transform(x[i] / 255.)
                        # x[i] = x[i] / 255.

                    # Make a prediction, then take the average.
                    with torch.no_grad():
                        y_pred = model(x)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        highestThree = sorted(range(len(y_pred)), key=lambda i: y_pred[i])[-3:]
                        #lowestThree = sorted(range(len(y_pred)), key=lambda i: y_pred[i])[:3]

                        highestFrameNums = []
                        for index in highestThree:
                            highestFrameNums.append(frameNum[index])

                        time = datetime.now()
                        f.write(time.strftime("%d/%m/%Y %H:%M:%S") + " -- Entered {0} into CNN algorithm\n".format(video_path))
                        f.write("\tPrediction: " + str(y_pred[:n].mean().item()) + "\n\n")
                        save_sus_frames(highestFrameNums, video_path)
                        return y_pred[:n].mean().item()

        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            lineno = tb.tb_lineno
            time = datetime.now()
            log.write("\n%s -- Prediction error on video %s: %s\n" % (time.strftime("%d/%m/%Y %H:%M:%S"), video_path, str(e)))
            print("Prediction error on video %s: %s %s" % (video_path, lineno, str(e)))

        return 0.5

    # Grabs the three most suspicious frames based on the y predictions and saves them
    def save_sus_frames(highestFrameNums, video_path):
        vid = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]
        highestFrameNums = sorted(highestFrameNums)
        total_frames = vid.get(7)
        cd = os.getcwd()
        if not os.path.isdir(video_name):
            os.mkdir('{0}'.format(video_name))
        for value in highestFrameNums:
            vid.set(1, value)
            ret, frame = vid.read()
            print('{0}/{1}/{1}_frame_{2}.jpg'.format(cd,video_name,value))
            cv2.imwrite('{0}/{1}/{1}_frame_{2}.jpg'.format(cd,video_name,value), frame)

    from concurrent.futures import ThreadPoolExecutor

    def predict_on_video_set(videos, num_workers):
        def process_file(i):
            filename = videos[i]
            y_pred = predict_on_video(filename, batch_size=frames_per_video)
            return y_pred

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            predictions = ex.map(process_file, range(len(videos)))

        return list(predictions)

    speed_test = False

    if speed_test:
        start_time = time.time()
        speedtest_videos = test_videos[:5]
        predictions = predict_on_video_set(speedtest_videos, num_workers=4)
        elapsed = time.time() - start_time
        print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))

    model.eval()
    predictions = predict_on_video_set(test_videos, num_workers=4)
    f.close()
    return predictions