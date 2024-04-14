import os
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence

# Data directory and labels setup
dir = '../data/videos/original'
labels_df = pd.read_csv("../data/labels.csv")
labels = dict(zip(labels_df['id'], labels_df['label']))
files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.mp4')]
files = [f for f in files if int(os.path.basename(f).split('.')[0]) in labels]
num_to_label = dict(enumerate(np.sort(np.unique(list(labels.values())))))
label_to_num = {v : k for k, v in num_to_label.items()}

# Helper functions
def get_video_label(file):
    file_id = int(os.path.basename(file).split('.')[0])
    return label_to_num[labels[file_id]]

def load_video(file_path, target_frames=20, target_size=(64, 64)):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()

    # Convert list of frames to numpy array
    video = np.array(frames)
    num_frames = video.shape[0]
    if num_frames < target_frames:
        pad = np.zeros((target_frames - num_frames, *target_size))
        video = np.concatenate([pad, video], axis=0)
    elif num_frames > target_frames:
        start = np.random.randint(0, num_frames - target_frames)
        video = video[start:start + target_frames]

    # Expand dimensions to have the channel last
    video = np.expand_dims(video, axis=-1)
    return video

class VideoDataGenerator(Sequence):
    def __init__(self, file_paths, batch_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.classes = [get_video_label(file) for file in file_paths]
        self.num_classes = len(np.unique(self.classes))

    def __len__(self):
        return np.ceil(len(self.file_paths) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        videos = [load_video(file) for file in batch_files]
        labels = [get_video_label(file) for file in batch_files]
        videos = np.array(videos)
        return videos, np.array(labels)