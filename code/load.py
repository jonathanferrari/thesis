import os
import pandas as pd
import numpy as np
import cv2
# Define the paths
videos_dir = '../data/videos'
labels_file = '../data/labels.csv'

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)



# Load the labels from the CSV file
labels_df = pd.read_csv(labels_file)

videos_fns = os.listdir(videos_dir)
videos = []
labels = []

for video_fn in videos_fns:
    video_id = int(video_fn.split('.')[0])
    label = labels_df[labels_df['video_id'] == video_id]['label'].values[0]
    full_fn = os.path.join(videos_dir, video_fn)
    video = cv2.VideoCapture(full_fn)
    labels.append(label)