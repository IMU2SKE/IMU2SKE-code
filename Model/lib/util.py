import os
import torch
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def split_get(peaks, valleys):
    pv = np.sort(np.concatenate((peaks, valleys)))
    pv_flag = [1 if x in peaks else 0 for x in pv]
    splits = [(pv[i], pv[i + 1]) for i in range(0, len(pv) - 1) if pv_flag[i] != pv_flag[i + 1]]
    return splits


def only_spilts(trajectory_data):
    x = [point[0] for point in trajectory_data]
    unalign_indexes = []
    
    coordinates = np.array(x)
    
    window_size = 50
    poly_order = 3

    smoothed_coordinates = savgol_filter(coordinates, window_size, poly_order)

    smoothed_peaks, _ = find_peaks(smoothed_coordinates, distance=40, height=0.5)
    smoothed_valleys, _ = find_peaks(-smoothed_coordinates, distance=40, height=-0.5)
    
    splits = split_get(smoothed_peaks, smoothed_valleys)
    return splits


def gen_graph(datas):
    attr = [data['bodies']['candidate'] for data in datas if not data == {}]
    attr = np.concatenate(attr, axis=0).reshape((-1, 18, 2))
    attr = attr[:, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), :]
    
    f, n, c = attr.shape
    frame_nums = 64
    ratios = max(1, f // frame_nums)
    attr = attr[::ratios, :, :]
    attr = attr[:frame_nums, :, :]
    
    f, n, c = attr.shape
    # print(f)
    
    num_nodes = f * n
    neighbor_link = [(0, 1), (1, 5), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
    
    base_src = [x[0] for x in neighbor_link]
    base_dst = [x[1] for x in neighbor_link]

    body_src = base_src + base_dst 
    body_dst = base_dst + base_src
    
    spartial_src = [x + n * i for i in range(f) for x in body_src]
    spartial_dst = [x + n * i for i in range(f) for x in body_dst]
    
    temporal_src = [x + n * i for i in range(f) for x in range(n-1)]
    temporal_dst = [x + n * i for i in range(f) for x in range(1, n)]
    
    src = spartial_src + temporal_src
    dst = spartial_dst + temporal_dst
    
    g = dgl.graph((spartial_src, spartial_dst), num_nodes=num_nodes)
    g.ndata['attr'] = torch.Tensor(attr.reshape(f * n, c))
    return g