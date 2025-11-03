import os
import csv
import copy
import wandb
import torch
import random
import argparse

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report

from lib import WarmupCosineLR, findAllFile, args
from lib.data import IMUDataset, SkeletonIMUDataset
from imu_lib.imu_models import FoGClassifier, SDFoG, LSTMClassifier, TCNClassifier
from imu_lib.loss import CLIPLoss, InfoNCE
from lib.mm import DSTMW2, EnhancedDSTMW2, SISD, SISDv2, SKE2IMU, IMULIP, IMU2SKE, IMU2SKEA
from skeleton_visual import save_skeleton_animation, save_comparison_skeleton_animation
# import swanlab

# swanlab.sync_wandb()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.distribute:
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = args.device

epochs = args.epochs
warmup_epochs = 5
learnning_rate = args.lr
weight_decay = 0
batch_size = 128

model_classes = {
    'IMUModelv1': FoGClassifier,
    'SDIMU': SDFoG,
    'MM': DSTMW2,
    'MM2_6': EnhancedDSTMW2,
    'SISD': SISD,
    'SISDv2': SISDv2,
    'LSTM': LSTMClassifier,
    'TCN': TCNClassifier,
    'SKE2IMU_c2': SKE2IMU,
    'IMULIPv2': IMULIP,
    'IMU2SKE_v6_mask_5fold': IMU2SKE,
    'IMU2SKEA': IMU2SKEA
}
def split_fold10(dataset, fold_idx=0):
    # 获取所有唯一人名
    unique_names = list(set([name.split('_')[0] for name in dataset.names]))
    unique_names.sort()
    
    # 统计每个人名下所有样本的0/1比例
    name_to_ratio = {}
    for name, flag in zip(dataset.names, dataset.labels):
        base_name = name.split('_')[0]
        if base_name not in name_to_ratio:
            name_to_ratio[base_name] = {'total': 0, 'positive': 0}
        
        # 统计每个样本的flag值
        name_to_ratio[base_name]['total'] += 1
        name_to_ratio[base_name]['positive'] += int(flag.item())
    
    # 计算每个人名的正样本比例
    ratios = []
    for name in unique_names:
        if name_to_ratio[name]['total'] > 0:
            ratio = name_to_ratio[name]['positive'] / name_to_ratio[name]['total']
        else:
            ratio = 0
        ratios.append(ratio)
    
    # 将比例离散化为5个区间用于分层
    bins = np.linspace(0, 1, 6)
    digitized = np.digitize(ratios, bins)
    
    # 使用StratifiedKFold按比例分层划分
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    idx_list = []
    for idx in skf.split(np.zeros(len(unique_names)), digitized):
        idx_list.append(idx)
    
    train_names_idx, valid_names_idx = idx_list[fold_idx]
    
    # 获取训练集和验证集的人名
    train_names = [unique_names[i] for i in train_names_idx]
    valid_names = [unique_names[i] for i in valid_names_idx]
    
    # 根据人名划分样本索引
    train_idx = [i for i, name in enumerate(dataset.names) if name.split('_')[0] in train_names]
    valid_idx = [i for i, name in enumerate(dataset.names) if name.split('_')[0] in valid_names]
    
    return train_idx, valid_idx

def evaluate(dataloader, device, model, file='temp.txt'):
    """评估模型性能"""
    model.eval()
    total_samples = 0
    total_correct = 0
    preds = torch.Tensor([]).to(device)
    labels = torch.Tensor([]).to(device)
    names = []
    
    # 初始化结果文件
    with open(file, 'w') as f:
        f.write("Name\tFile\tPrediction\tLabel\n")
       
    with torch.no_grad():
        for data in dataloader:
            # 获取数据
            labels_batch = data['flag'].to(device)
            imu = data['imu'].to(device)
            names_batch = data['name']
            files_batch = data['file']
            if isinstance(model, SDFoG):
                limu = data['limu'].to(device)
            
                # 模型推理
                outputs = model(imu, limu)
            elif isinstance(model, DSTMW2) or isinstance(model, EnhancedDSTMW2) \
                or isinstance(model, SKE2IMU) or isinstance(model, IMULIP) \
                    or isinstance(model, IMU2SKE) or isinstance(model, IMU2SKEA):
                skeleton = data['skeleton'].to(device)
                outputs = model(skeleton, imu)
            elif isinstance(model, SISD) or isinstance(model, SISDv2):
                limu = data['limu'].to(device)
                skeleton = data['skeleton'].to(device)
                lskel = data['lskel'].to(device)
                outputs = model(lskel, skeleton, limu, imu)
            else:
                outputs = model(imu)
            probs_batch = outputs['freeze_logits']
            preds_batch = torch.argmax(probs_batch, dim=-1)
            
            # 统计结果
            batch_size = labels_batch.shape[0]
            total_samples += batch_size
            labels_batch = labels_batch.squeeze(-1)
            total_correct += (preds_batch == labels_batch).sum().item()
            
            # 保存预测结果
            preds = torch.cat([preds, preds_batch], dim=0)
            labels = torch.cat([labels, labels_batch], dim=0)
            names.extend(names_batch)
                 
            # 写入文件
            # with open(file, 'a') as f:
            #     for i in range(batch_size):
            #         name = names_batch[i]
            #         file = files_batch[i]
            #         pred = str(probs_batch[i].cpu().numpy().tolist())
            #         label = labels_batch[i].item()
            #         f.write(f"{name}\t{file}\t{pred}\t{label}\n")
    
    # 计算准确率
    acc = 1.0 * total_correct / total_samples
    
    return acc, preds, labels, names

def train(train_loader, val_loader, args, model, fold_idx):
    """训练模型"""
    # 初始化损失函数和优化器
    if isinstance(model, SKE2IMU) or isinstance(model, IMULIP) or isinstance(model, IMU2SKE) or isinstance(model, IMU2SKEA):
        clip_loss_fcn = CLIPLoss()
    freeze_loss_fcn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device))  # 类别权重处理样本不平衡
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=learnning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                learnning_rate,
                epochs,
                steps_per_epoch=len(train_loader)
            )
    best_acc = 0  # 初始化最佳准确率
    if isinstance(model, SKE2IMU):
        model.enable_skeleton(False)
    
    for epoch in range(epochs):
        if epoch >= 0.5 * epochs and isinstance(model, SKE2IMU):
            model.enable_skeleton(True)
            # model.set_fusion_weight(0.1 + 0.4 * (epoch - 0.2 * epochs) / (0.8 * epochs))
        model.train()
        total_loss = 0
        total_freeze_loss = 0
        total_clip_loss = 0
        total_frames = 0
        total_train_correct = 0
        
        for batch, data in enumerate(train_loader):
            # 获取数据
            imu = data['imu'].to(device)
            labels = data['flag'].to(device).long()
            if isinstance(model, SDFoG):
                limu = data['limu'].to(device)
            
                # 模型推理
                outputs = model(imu, limu)
            elif isinstance(model, DSTMW2) or isinstance(model, EnhancedDSTMW2) \
                or isinstance(model, SKE2IMU) or isinstance(model, IMULIP)\
                    or isinstance(model, IMU2SKE) or isinstance(model, IMU2SKEA):
                skeleton = data['skeleton'].to(device)
                outputs = model(skeleton, imu)
            elif isinstance(model, SISD) or isinstance(model, SISDv2):
                limu = data['limu'].to(device)
                skeleton = data['skeleton'].to(device)
                lskel = data['lskel'].to(device)
                outputs = model(lskel, skeleton, limu, imu)
            else:
                outputs = model(imu)
            
            # 计算训练准确率
            preds = torch.argmax(outputs['freeze_logits'], dim=-1)
            labels = labels.squeeze(1)
            batch_correct = (preds == labels).sum().item()
            batch_frames = labels.numel()
            total_train_correct += batch_correct
            total_frames += batch_frames

            # 计算损失
            freeze_loss = freeze_loss_fcn(
                outputs['freeze_logits'].reshape(-1, outputs['freeze_logits'].size(-1)),
                labels.reshape(-1)
            )
            if (not isinstance(model, SKE2IMU) or args.clip == 'False') and not isinstance(model, IMULIP) and (not isinstance(model, IMU2SKE) or args.clip == 'False') \
                and (not isinstance(model, IMU2SKEA) or args.clip == 'False'):
                loss = freeze_loss
            else:
                dual_clip_loss = clip_loss_fcn(outputs['skeleton_features'], outputs['imu_features']) if args.dual_clip == 'True' else 0
                imu_clip_loss = clip_loss_fcn(outputs['imu_features'], outputs['imu_features']) if args.imu_clip == 'True' else 0
                skeleton_clip_loss = clip_loss_fcn(outputs['skeleton_features'], outputs['skeleton_features']) if args.skeleton_clip == 'True' else 0
                clip_loss = dual_clip_loss + imu_clip_loss + skeleton_clip_loss           
                loss = freeze_loss + clip_loss
                clip_loss = torch.Tensor(0).to(device) if not isinstance(clip_loss, torch.Tensor) else clip_loss
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            total_freeze_loss += freeze_loss.item()
            if (isinstance(model, SKE2IMU) and args.clip == 'True') or isinstance(model, IMULIP) or (isinstance(model, IMU2SKE) and args.clip == 'True') \
                or (isinstance(model, IMU2SKEA) and args.clip == 'True'):
                    if clip_loss.numel() > 0:
                        total_clip_loss += clip_loss.item()
            
            # 打印batch信息
            batch_acc = 1.0 * batch_correct / batch_frames
            print(
                "Epoch {:05d} | Batch {:03d} | batch_loss {:.4f} | batch_acc {:.4f}".format(
                    epoch, batch, loss.item(), batch_acc
                )
            )
            
        # 计算训练准确率
        train_acc = 1.0 * total_train_correct / total_frames
            
        # 更新学习率
        scheduler.step()
        
        # 验证模型
        valid_acc, _, _, _ = evaluate(val_loader, device, model)

        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'{args.path}/IMUModel_{fold_idx}.pth')
            best_model = copy.deepcopy(model)
            
        # 记录日志
        lr = optimizer.param_groups[0]['lr']
        loss = total_loss / (batch + 1)
        wandb.log({
            'lr': lr,
            'train_acc': train_acc,
            'valid_acc': valid_acc, 
            'best_acc': best_acc,
            'freeze_loss': total_freeze_loss / (batch + 1),
            'clip_loss': total_clip_loss / (batch + 1),
        })
        
        # 打印训练信息
        print(
            "Epoch {:05d} | lr {:.4f} | freeze_loss {:.4f} | clip_loss {:.4f} | Train Acc {:.4f} | Validation Acc {:.4f} | Best Acc {:.4f}".format(
                epoch, lr, 
                total_freeze_loss / (batch + 1),
                total_clip_loss / (batch + 1),
                train_acc,
                valid_acc, best_acc
            )
        )
        
    # 保存最终模型
    torch.save(model.state_dict(), f'{args.path}/IMUModel_last_{fold_idx}.pth')

def process(args):
    mode = args.mode
    mask = args.mask
    
    dataset = SkeletonIMUDataset(mode='train', imu_dir='data/IMU', skeleton_dir='data/output')
    dataset_val = SkeletonIMUDataset(mode='train', imu_dir='data/IMU', skeleton_dir='data/output')

    eval_name = []
    eval_freeze_preds = []
    eval_freeze_labels = []
    os.popen(f'rm {args.path}/{mode}.txt')
    for fold_idx in range(5):
        model = model_classes[args.model](
            num_classes=2,
            freeze_encoder=args.freeze_encoder,
            skeleton_pretrain_path=args.skele_path,
            imu_pretrain_path=args.pretrain_path
            ).to(device)
        
        train_idx, valid_idx = split_fold10(dataset, fold_idx=fold_idx)
        print(f"训练集长度: {len(train_idx)}, 验证集长度: {len(valid_idx)}")
        
        train_names = set([dataset.names[i].split('_')[0] for i in train_idx])
        valid_names = set([dataset.names[i].split('_')[0] for i in valid_idx])
        assert len(train_names.intersection(valid_names)) == 0, "训练集和验证集存在重叠样本"
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=8
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=8,
            shuffle=False
        )
        
        if mode == 'train':
            print(f"Training IMU model")
            config_dict = vars(args).copy() if hasattr(args, "__dict__") else args.copy()

            config_dict.pop("device", None)
            
            wandb.init(
                project="MICCAI2025",
                config=config_dict
            )
            print("Training...")
            
            train(train_loader, val_loader, args, model, fold_idx)
            wandb.finish()
        elif mode == 'val' or mode == 'test':
            try:
                print(f'Loading model from {args.path}/IMUModel_{fold_idx}_{epochs}.pth')
                state_dict = torch.load(f'{args.path}/IMUModel_{fold_idx}_{epochs}.pth')
            except FileNotFoundError:
                print(f'Model not found at {args.path}/IMUModel_{fold_idx}_{epochs}.pth, trying {args.path}/IMUModel_{fold_idx}.pth')
                state_dict = torch.load(f'{args.path}/IMUModel_{fold_idx}.pth')
                
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.eval()

            if mode == 'test':
                print(f'Loading test dataset from {args.data_path}')
                dataset = IMUDataset('test', args.data_path)
                val_loader = DataLoader(
                    dataset, batch_size=32, drop_last=False, num_workers=4
                )
                
            file_path = args.path + f'/{mode}_{fold_idx}.txt'
            print(f'Evaluating model and saving results to {file_path}')
            
            freeze_acc, freeze_preds, freeze_labels, names = evaluate(val_loader, device, model, file_path)
            
            freeze_preds = freeze_preds.cpu().numpy()
            freeze_labels = freeze_labels.cpu().numpy()
                        
            eval_name.append(names)
            eval_freeze_preds.append(freeze_preds)
            eval_freeze_labels.append(freeze_labels)
        if args.cv == 'False':
            break
            
    if mode == 'val' or mode == 'test':
        val_name_np = np.concatenate(eval_name, axis=0)
        freeze_preds = np.concatenate(eval_freeze_preds, axis=0)
        freeze_labels = np.concatenate(eval_freeze_labels, axis=0)
        cm = confusion_matrix(freeze_labels, freeze_preds)
        print(cm)
        print(classification_report(freeze_labels, freeze_preds))
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        print(f'Sensitivity: {sensitivity:.4f}')
        print(f'Specificity: {specificity:.4f}')

        AUC = (sensitivity + specificity) / 2
        print(f'AUC: {AUC:.4f}')
        
        np.save(f'{args.path}/freeze_preds.npy', freeze_preds)
        np.save(f'{args.path}/freeze_labels.npy', freeze_labels)
        
        freeze_acc = np.mean(freeze_preds == freeze_labels)
        print(f'Freeze Prediction Accuracy: {freeze_acc:.4f}')

if __name__ == "__main__":
    process(args)