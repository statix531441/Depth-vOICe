import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from scipy.io import wavfile

import torch

from util import get_dmap, get_object_bboxes

import argparse

parser = argparse.ArgumentParser(description="Train a machine learning model")

parser.add_argument("--video", default='videos/car2cam.mp4', type=str)


args = parser.parse_args()

CONF_THRES = 0.6
CROP = [None, None, None, 1250]
SHAPE = (300,150)

file = args.video
out_file = file.split('/')[-1].split('.')[0]
os.makedirs(f'outputs/{out_file}', exist_ok=True)

vid = cv2.VideoCapture(file)
out = cv2.VideoWriter(f'outputs/{out_file}/video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, SHAPE, 0)

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


fps = 15
final = None

sr = 48000
wav_range = [-1,1]
#creating all the waves
time_x=np.arange(0, 1, 1.0/sr)
freqs = np.exp(np.linspace(np.log(20_000), np.log(30), 150))
tones = np.sin(2.0 * np.pi * freqs.reshape(-1, 1) * time_x)



# Read video and get signals
frame_num = 0
while vid.isOpened():
    ret, frame = vid.read()
    if ret and frame_num<=120:
        if frame_num%fps==0:
            print(frame_num)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img[CROP[0]:CROP[1], CROP[2]:CROP[3]]
        
            dmap = get_dmap(img)
            df = get_object_bboxes(img)

            obj_filtered = np.zeros_like(dmap)
            for objs in df[df.confidence>CONF_THRES].iterrows():
                xmin, ymin, xmax, ymax = objs[1][:4]
                obj_filtered[ymin:ymax, xmin:xmax] = dmap[ymin:ymax, xmin:xmax]

            obj_filtered = cv2.resize(obj_filtered, SHAPE)

            signal = np.repeat(obj_filtered, tones.shape[1]/obj_filtered.shape[1], axis=1) * tones
            signal = signal.sum(axis=0)[1000:]

            if final is None:
                final = signal
                print(signal.shape)
                duration = 1
                left = np.linspace(1, 0, signal.size)
                right = np.linspace(0, 1, signal.size)
            else:
                final = np.concatenate((final, signal))
                duration += 1

        out.write((255*obj_filtered).astype('uint8'))
        cv2.imshow('Final video', cv2.resize(obj_filtered, img.shape[1::-1]))
        frame_num += 1

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
    
vid.release() 
cv2.destroyAllWindows() 

### Normalize the signal and convert to wav file.
final -= final.min()
final /= final.max() / (wav_range[1] - wav_range[0])
final += wav_range[0]

left = np.tile(left, duration)
right = np.tile(right, duration)

tone_y_stereo=np.vstack((left*final, right*final))
# tone_y_stereo=np.vstack((final, final))
tone_y_stereo=tone_y_stereo.transpose()
wavfile.write(f'outputs/{out_file}/audio.wav', sr, tone_y_stereo)