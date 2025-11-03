import os
import time
import cv2
import pickle
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from dwpose import DWposeDetector

poseDet = DWposeDetector()

def findAllFile(base, ext=None):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if not ext is None and not fullname.endswith(ext):
                continue
            file_path.append(fullname)
    return file_path

def real2ske(video_path, output_dir):
    # 读取视频并获取基本信息
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    height, width, _ = frame.shape
    video.release()
    
    # 重新打开视频进行处理
    real_video = cv2.VideoCapture(video_path)
    
    # 设置输出路径
    output_path = video_path.replace('videos', 'output')
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = video_path.replace('videos', 'output').replace('mp4', 'pickle')
    
    # 初始化变量
    frames = []
    pickle_data = []
    start_time = time.time()
    frame_num = 0
    
    # 逐帧处理视频
    while real_video.isOpened():
        ret, frame = real_video.read()
        if ret:
            try:
                detect_map, pose = poseDet(frame)
            except Exception as e:
                with open('error.txt', 'a') as f:
                    f.writelines(f"{pickle_path}: {str(e)}\n")
                detect_map, pose = frame, {}
            # 转换颜色空间并保存帧
            detect_map_rgb = cv2.cvtColor(detect_map, cv2.COLOR_BGR2RGB)
            frames.append(detect_map_rgb)
            pickle_data.append(pose)
        else:
            break
        frame_num += 1
        
    # 保存姿态数据
    with open(pickle_path, 'wb') as file:
        pickle.dump(pickle_data, file)
    
    # 释放视频资源
    real_video.release()
    
    # 使用moviepy保存处理后的视频
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(output_path, codec='libx264', bitrate='500k')
    
    return pickle_path, output_path

if __name__ == '__main__':
    video_root = 'data/videos'
    video_paths = findAllFile(video_root, 'mp4')
    print(video_paths)
    for video_path in video_paths:
        print(video_path)
        output_dir = os.path.dirname(video_path).replace('videos', 'output')
        real2ske(video_path, output_dir)
