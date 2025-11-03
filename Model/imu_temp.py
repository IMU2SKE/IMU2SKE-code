import torch
import sys
from lib.mm import DSTMW2, EnhancedDSTMW2, SISD, SISDv2
from lib.data import IMUDataset, SkeletonIMUDataset

if __name__ == "__main__":
    # 设置测试参数
    # batch_size = 8
    # skeleton_dim = 2  # 骨架数据维度
    # imu_dim = 6        # IMU数据维度
    # seq_len = 100      # 序列长度

    # # 创建测试输入
    # skeleton_input = torch.randn(batch_size, 2, 243, 6, 1)
    # imu_input = torch.randn(batch_size, imu_dim, 1000)

    # # # 初始化模型
    # # 初始化模型并加载预训练权重
    # model = SISD(freeze_encoder='True', 
    #                skeleton_pretrain_path='/home/tzh/Project/MotionBERT/checkpoint/pretrain/best_epoch.bin',
    #                imu_pretrain_path='/home/tzh/Project/GaitFreezing/data/imu2clip/i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt')
    
    # # 测试是否加载了预训练模型
    # # print("\n测试skeleton预训练模型加载:")
    # # # 打印模型参数数量
    # # total_params = sum(p.numel() for p in model.dst_extractor.parameters())
    # # print(f"Skeleton模型总参数数量: {total_params}")
    
    # # # 打印前几个参数值
    # # print("\nSkeleton模型前5个参数值:")
    # # for name, param in list(model.dst_extractor.named_parameters())[:5]:
    # #     print(f"{name}: {param.data[:2]}")
        
    # # # 手动加载skeleton预训练模型
    # # skeleton_pretrain_path = '/home/tzh/Project/MotionBERT/checkpoint/pretrain/best_epoch.bin'
    # # print(f"\n手动加载Skeleton预训练模型: {skeleton_pretrain_path}")
    
    # # # 加载预训练权重
    # # state_dict = torch.load(skeleton_pretrain_path)
    # # print(state_dict.keys())
    
    # # # 打印state_dict
    # # print("\nSkeleton模型state_dict:")
    # # for key in list(state_dict['model'].keys())[:5]:
    # #     print(f"{key}: {state_dict['model'][key][:2]}")
    
    
    
    # # # 打印模型结构
    # # print("\nSkeleton模型结构:")
    # # print(model.dst_extractor)

    # # 前向传播
    # outputs = model(skeleton_input, skeleton_input, imu_input, imu_input)

    # # 打印输出结果
    # print("\n测试结果：")
    # print(f"输出logits的形状: {outputs['freeze_logits'].shape}")
    # print(f"输出logits的值: \n{outputs['freeze_logits']}")
    data = SkeletonIMUDataset(mode='train', imu_dir='data/IMU', skeleton_dir='data/output')
    for d in data:
        # print(d['imu'].shape)
        imu_d = d['imu']
        break

    from matplotlib import pyplot as plt
    plt.figure(figsize=(20, 5))
    for i in range(6):
        plt.plot(imu_d[i, :])
    plt.savefig('imu.png')
    plt.savefig('imu.eps')
    #     first = d['skeleton']
    #     break
    
    # data = SkeletonIMUDataset(mode='train', imu_dir='data/IMU', skeleton_dir='data/output')
    # for d in data:
    #     print(d['imu'].shape)
    #     second = d['skeleton']
    #     break
    
    # assert torch.all(first == second), f"The two tensors are not equal. Max absolute difference: {(first - second).abs().max()}"



    # Generate random IMU-like motions as examples
    # imu_motions: array <n_samples x 6 x 1000>
    # imu_motions = torch.rand(2, 6, 1000)
    # print(imu_motions.dtype)
    # print("Generated random IMU-like motions:", imu_motions)

    # # Load the IMU encoder
    # """
    #     The following example .pt model is configured as
    #     - i2c: IMU2CLIP
    #     - s_i: source modality = IMU
    #     - t_v: target modality for alignment = Video
    #     - t_t: target modality for alignment = Text
    #     - mw2: MW2StackRNNPooling as the encoder
    #     - w_5.0: window size of 2.5 x 2 seconds
    # """
    # #path_imu_encoder = "./i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt"
    # # path_imu_encoder = "./i2c_s_i_t_t_ie_mw2_w_2.5_master_imu_encoder.pt"
    # # path_imu_encoder = "/home/pranayr_umass_edu/imu2clip/shane_models/i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt"
    # path_imu_encoder = 'data/imu2clip/i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt'

    # loaded_imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    # loaded_imu_encoder.load_state_dict(torch.load(path_imu_encoder))
    # print("Loaded IMU Encoder:", loaded_imu_encoder)
    # loaded_imu_encoder.eval()
    # print("Done loading the IMU Encoder")

    # # Inference time
    # outputs_and_states = loaded_imu_encoder(imu_motions)

    # print('Raw IMU Signals (random)', imu_motions.shape)
    # print('Encoded IMU2CLIP embeddings', imu2clip_embeddings.shape)


    # # check if raw and encoded are the same
    # print('Are the raw and encoded the same?', torch.allclose(imu_motions, imu2clip_embeddings))
