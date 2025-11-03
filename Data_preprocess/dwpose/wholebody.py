import cv2
import numpy as np
# import torch

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose, preprocess

class Wholebody:
    def __init__(self):
        device = 'cuda:0'  # if torch.cuda.is_available() else 'cpu'
        providers = ['CUDAExecutionProvider']
        onnx_det = 'data/ckpts/yolox_l.onnx'
        onnx_pose = 'data/ckpts/dw-ll_ucoco_384.onnx'

        try:
            self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
            self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
        except:
            print('CUDAExecutionProvider is not support by now, running on CPUExecutionProvider')
            providers = ['CPUExecutionProvider']
            self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
            self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
            
    
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores
    
    def det(self, oriImg):
        out_bbox = inference_detector(self.session_det, oriImg)
        img = [oriImg[int(x[1]):int(x[3]), int(x[0]): int(x[2])] for x in out_bbox]
        resized_img, center, scale = preprocess(oriImg, out_bbox, (288, 384))
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = [x * std + mean for x in resized_img]
        return resized_img, out_bbox
