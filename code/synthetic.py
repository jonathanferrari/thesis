import cv2
import numpy as np

def rotate_video(input_path, output_path, angle=90):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if angle == 90 or angle == 270:
        size = (height, width)
    else:
        size = (width, height)
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if angle == 90 else cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        writer.write(frame)
    
    cap.release()
    writer.release()

def flip_video(input_path, output_path, flip_code=1):
    # flip_code: 0 for vertical, 1 for horizontal, -1 for both axes
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, flip_code)
        writer.write(frame)
    
    cap.release()
    writer.release()

def crop_video(input_path, output_path, x, y, w, h):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (w, h)
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[y:y+h, x:x+w]
        writer.write(frame)
    
    cap.release()
    writer.release()

def jitter_and_noise_video(input_path, output_path, brightness=0.2, noise_amount=0.04):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply brightness jitter
        frame = np.clip(frame * (1 + brightness * (np.random.rand() - 0.5)), 0, 255).astype(np.uint8)
        # Apply noise
        noise = np.random.randn(*frame.shape) * 255 * noise_amount
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    
    cap.release()
    writer.release()