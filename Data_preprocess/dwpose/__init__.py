# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W, img=None):
    bodies = pose['bodies']
    faces = pose['faces']
    foot = pose['foot']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    if img is None:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    else:
        canvas = img

    canvas = util.draw_bodypose(canvas, candidate, subset)

    # canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_footpose(canvas, foot)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()
        
    def det(self, oriImg):
        oriImg = oriImg.copy()
        return self.pose_estimation.det(oriImg)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        # with torch.no_grad():
        candidate, subset = self.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        confi = score.reshape(nums*18, 1).copy()
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset<0.3
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])
        
        bodies = dict(candidate=body, subset=score, confi=confi)
        pose = dict(bodies=bodies, hands=hands, faces=faces, H=H, W=W, foot=foot)

        return draw_pose(pose, H, W, oriImg), pose
