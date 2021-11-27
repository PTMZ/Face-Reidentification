'''
Facenet + Simple Temporal Consistency

Original pipeline:
Multi-task CNN (MTCNN) is responsible for face detection (Drawing bounding boxes around faces).
Facenet (InceptionResnetv1) takes in an image of a face and computes a feature vector.
Image -> MTCNN -> Crop Image and Normalise -> Facenet -> Feature Vector

Database:
The database is precalculated from folders of images.
This is then stored into a N x d matrix where N is the number of database images 
and d is the dimension of the feature vector.
 
Once a feature vector is obtained, cosine similarity is computed with feature vectors in the database.
If the max cosine similarity found is above a threshold, it indicates as detected
and matches cropped bounding box of the face to the name in the database.

Simple Temporal Consistency
To improve the prediction accuracy, we can make use of the fact that there is little difference
between consecutive frames.

'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import time
from collections import deque

import cv2
from PIL import Image
import hashlib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Variables to initalise
# skip boolean is for skipping frames when reading from a video file
webcam = True
skip = False
input_filename = 'test_vid2.mp4'
output_filename = 'out3.avi'

# State class to represent information for each frame
# Given a list of names to track, each name is assigned:
#   1) Probability of it in the frame
#   2) Bounding box of the face, represented as (x1,y1,x2,y2)
# update_state method adds the info of probs and boxes from facenet
# combine_state method takes in the previous state and combines info from both to have a better prediction
class State:
    def __init__(self, name_list):
        self.probs = {n:0 for n in name_list}
        self.box = {n:(0,0,0,0) for n in name_list}
        self.name_list = name_list
        self.decay = 0.7
        self.threshold = 0.3
    
    def update_state(self, name, prob, x1, y1, x2, y2):
        self.probs[name] = prob
        self.box[name] = (x1,y1,x2,y2)

    def combine_state(self, prev):
        # self is new state
        for n in self.name_list:
            if self.probs[n] == 0 and prev.probs[n] > 0:
                self.probs[n] = prev.probs[n] * self.decay
            elif self.probs[n] > 0 and prev.probs[n] > 0:
                self.probs[n] = 1 - (1-self.probs[n])*(1-prev.probs[n])
            
            if self.box[n] == (0,0,0,0) and self.probs[n] > self.threshold:
                self.box[n] = prev.box[n]

# Helper function to clamp values within a range.
# This is used to keep the bounding box within the frame
def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

# Helper function to normalise values in the image.
# This is required preprocessing for input to Facenet
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def compare_database(face_embed, all_embed, all_names, dist_thresh):
    scores = torch.matmul(all_embed, face_embed.T)
    score, idx = torch.max(scores, dim=0)
    score = score.item()
    idx = idx.item()
    if score > dist_thresh:
        return all_names[idx], score
    return None, None


dist_thresh = 0.6

mtcnn = MTCNN(
    image_size=160, margin=0,
    thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(input_filename)
    cap_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    print(cap_w, cap_h, cap_fps)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), cap_fps, (cap_w,cap_h))


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# For loop database of references
face_dir_list = os.listdir('face_database')
face_embed_dict = {}

print('Loading database...')
all_embed = []
all_names = []
for name in face_dir_list:
    face_embed_dict[name] = 0
    for image_file in os.listdir(f'face_database/{name}'):
        img = Image.open(f"face_database/{name}/{image_file}").convert('RGB')
        face_tensor = mtcnn(img).unsqueeze(0).to(device)
        face_embed = resnet(face_tensor).detach().cpu()
        face_embed = face_embed / torch.linalg.norm(face_embed)
        all_embed.append(face_embed)
        all_names.append(name)
        face_embed_dict[name] += 1

all_embed = torch.cat(all_embed)
print('Database loaded')


for k,v in face_embed_dict.items():
    print(f"{k}: {v} reference frames found.")

skip_frames = 0
cur_state = State(face_dir_list)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if frame is None:
        break
    
    if skip and (not webcam) and skip_frames > 0:
        skip_frames -= 1
        continue
    
    # Detect and draw bounding boxes
    boxes, prob = mtcnn.detect(frame)
    new_state = State(face_dir_list)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = clamp(x1, 0, frame.shape[1]-1)
            x2 = clamp(x2, 0, frame.shape[1]-1)
            y1 = clamp(y1, 0, frame.shape[0]-1)
            y2 = clamp(y2, 0, frame.shape[0]-1)
            
            face_tensor = torch.Tensor(normalize(cv2.resize(frame[y1:y2, x1:x2], (160, 160))))
            face_tensor = face_tensor.permute(2,0,1).unsqueeze(0).to(device)
            face_embed = resnet(face_tensor).detach().cpu()
            face_embed = face_embed / torch.linalg.norm(face_embed)

            detected_name, distance = compare_database(face_embed, all_embed, all_names, dist_thresh)
            if detected_name is not None:
                new_state.update_state(detected_name, distance, x1, y1, x2, y2)
                #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                #cv2.putText(frame, detected_name + f'__{distance:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                pass
                #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
    
    # Combine info from prev frame
    new_state.combine_state(cur_state)

    cur_state = new_state
    # Draw bounding rectangles and label face
    for n in cur_state.name_list:
        if cur_state.probs[n] > cur_state.threshold:
            x1, y1, x2, y2 = cur_state.box[n]
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, n + f'__{cur_state.probs[n]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    if not webcam:
        out.write(frame)
    cv2.imshow('Input', frame)

    end_time = time.time()
    skip_frames = int((end_time - start_time) / 0.1)

    c = cv2.waitKey(1)
    # Press Esc to exit
    if c == 27:
        break

cap.release()
if not webcam:
    out.release()

cv2.destroyAllWindows()



