# from lib.data import PoseTorchDataset, convert_from_bvh
# from skeleton_visual import save_skeleton_animation
# import torch
# from lib.model import DiT_models
import numpy as np



if __name__ == "__main__":
    '''
    lr:5e-3, batch_size:128, epoch:50 
    '''
    IMUModels = [0.965, 0.96444, 0.955, 0.94393,
                 0.93861, 0.93667, 0.93417, 0.91863, 0.91437,
                 0.90963, 0.90074, 0.89719, 0.89333, 0.89086]
    
    LSTM = [0.96222, 0.95778, 0.93949, 0.93856,
            0.93639, 0.91111, 0.90815, 0.90785, 0.90272,
            0.89778, 0.89659, 0.89304, 0.88915, 0.88178, 0.86278]
    
    TCN = [0.92028, 0.91608, 0.90272, 0.89889, 0.88821,
           0.88652, 0.875, 0.87074, 0.86785, 0.85541,
           0.85274, 0.82944, 0.80917, 0.72034]
    
    for group in [IMUModels, LSTM, TCN]:
        print(np.mean(group), np.std(group))
    
    # data = PoseTorchDataset('train', None, 'data/PDFEinfo.csv', 'data/output', 'data/IMU', dual=False)
    # for x in data:
    #     print(x['attr'].shape)
    #     print(x['imu'].shape)
    #     print(x['label'].shape)
    #     print(x['flag'].shape)
    # m = DiT_models['DiT-B/2'](in_channels=4)
    # a = torch.rand(1, 3584, 8, 2)
    # print(m(a).shape)