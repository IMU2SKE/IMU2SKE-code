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

from lib import WarmupCosineLR, Graph, findAllFile, args
from lib.data import PoseTorchDataset, convert_from_bvh, bone_pairs
from lib.model import STGCN_o, ContrastGCN, HDGCN, DiT_models, LDT
from skeleton_visual import save_skeleton_animation, save_comparison_skeleton_animation

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
batch_size = 2


model_classes = {
    'STGCN': STGCN_o,
    'HDGCN': HDGCN,
    'ContrastSTGCN': ContrastGCN,
    'LDT': LDT
}
model_classes.update(DiT_models)

def split_fold10(dataset, fold_idx=0):
    # 获取所有唯一人名
    unique_names = list(set([name.split('_')[0] for name in dataset.names]))
    unique_names.sort()
    
    # 计算每个人名下所有样本的flag中1的总数
    name_to_flag_count = {}
    for name, flag in zip(dataset.names, dataset.flags):
        base_name = name.split('_')[0]
        if base_name not in name_to_flag_count:
            name_to_flag_count[base_name] = 0
        name_to_flag_count[base_name] += torch.sum(flag == 1).item()
    
    # 将人名按flag中1的总数分组
    flag_counts = [name_to_flag_count[name] for name in unique_names]
    bins = np.linspace(min(flag_counts), max(flag_counts), 6)
    digitized = np.digitize(flag_counts, bins)
    
    # 使用StratifiedKFold按flag中1的数量分组进行分层划分
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    
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
    total_frames = 0
    total_freeze_correct = 0
    freeze_preds = torch.Tensor([]).to(device)
    freeze_labels = torch.Tensor([]).to(device)
    names = []
    
    # 初始化结果文件
    with open(file, 'w') as f:
        f.write("Name\tFreeze Prediction\tFreeze Label\n")
       
    with torch.no_grad():
        for data in dataloader:
            # 获取数据
            freeze_labels_batch = data['flag'].to(device)
            skeleton = data['attr'].to(device)
            imu = data['imu'].to(device)
            names_batch = data['name']
            
            # 模型推理
            mask = torch.triu(torch.ones(3584, 3584), diagonal=1).bool().to(device)
            outputs = model(skeleton, imu, mask)
            freeze_probs_batch = outputs['freeze_logits']
            freeze_preds_batch = torch.argmax(freeze_probs_batch, dim=-1)
            
            # 统计结果
            batch_size, seq_len = freeze_labels_batch.shape
            total_frames += batch_size * seq_len
            total_freeze_correct += (freeze_preds_batch == freeze_labels_batch).sum().item()
            
            # 保存预测结果
            freeze_preds = torch.cat([freeze_preds, freeze_preds_batch], dim=0)
            freeze_labels = torch.cat([freeze_labels, freeze_labels_batch], dim=0)
            names.extend(names_batch)
                 
            # 写入文件
            with open(file, 'a') as f:
                for i in range(len(freeze_labels_batch)):
                    name = names_batch[i]
                    for j in range(freeze_preds_batch.size(1)):
                        freeze_pred = freeze_preds_batch[i, j].item()
                        freeze_label = freeze_labels_batch[i, j].item()
                        f.write(f"{name}\t{freeze_pred}\t{freeze_label}\n")
    
    # 计算准确率（考虑所有标签）
    freeze_acc = 1.0 * total_freeze_correct / total_frames
    
    return freeze_acc, freeze_preds, freeze_labels, names

def train(train_loader, val_loader, args, model, fold_idx):
    """训练模型"""
    # 初始化损失函数和优化器
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
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_freeze_loss = 0
        total_skeleton_clip_loss = 0
        total_imu_clip_loss = 0
        total_multi_modal_clip_loss = 0
        total_imu_skeleton_clip_loss = 0
        total_frames = 0
        total_train_correct = 0
        
        for batch, data in enumerate(train_loader):
            # 获取数据
            skeleton = data['attr'].to(device)
            imu = data['imu'].to(device)
            labels = data['flag'].to(device).long()
            
            # 前向传播
            mask = torch.triu(torch.ones(3584, 3584), diagonal=1).bool().to(device)
            outputs = model(skeleton, imu, mask)
            
            # 计算训练准确率
            preds = torch.argmax(outputs['freeze_logits'], dim=-1)
            total_train_correct += (preds == labels).sum().item()
            total_frames += labels.numel()

            # 计算损失（使用加权损失函数）
            freeze_loss = freeze_loss_fcn(
                outputs['freeze_logits'].reshape(-1, outputs['freeze_logits'].size(-1)),
                labels.reshape(-1)
            )
            
            # 根据args.use参数动态计算各损失权重
            weight_skeleton = 0.1 if 'all' in args.use or 'use_skeleton' in args.use else 0
            skeleton_clip_loss = weight_skeleton * outputs['skeleton_clip_loss']
            
            weight_imu = 0.1 if 'all' in args.use or 'use_imu' in args.use else 0
            imu_clip_loss = weight_imu * outputs['imu_clip_loss']
            
            weight_multi = 0.2 if 'all' in args.use or 'use_multi' in args.use else 0
            multi_modal_clip_loss = weight_multi * outputs['multi_modal_clip_loss']
            
            weight_imu_skeleton = 0.2 if 'all' in args.use or 'use_imu_skeleton' in args.use else 0
            imu_skeleton_clip_loss = weight_imu_skeleton * outputs['imu_skeleton_clip_loss']
            loss = (
                freeze_loss +
                skeleton_clip_loss +
                imu_clip_loss +
                multi_modal_clip_loss + 
                imu_skeleton_clip_loss
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            total_freeze_loss += freeze_loss.item()
            total_skeleton_clip_loss += skeleton_clip_loss.item()
            total_imu_clip_loss += imu_clip_loss.item()
            total_multi_modal_clip_loss += multi_modal_clip_loss.item()
            total_imu_skeleton_clip_loss += imu_skeleton_clip_loss.item()
            
        # 计算训练准确率
        train_acc = 1.0 * total_train_correct / total_frames
            
        # 更新学习率
        scheduler.step()
        
        # 验证模型
        valid_acc, _, _, _ = evaluate(val_loader, device, model)

        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'{args.path}/DiT_{fold_idx}.pth')
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
            'skeleton_clip_loss': total_skeleton_clip_loss / (batch + 1),
            'imu_clip_loss': total_imu_clip_loss / (batch + 1),
            'multi_modal_clip_loss': total_multi_modal_clip_loss / (batch + 1),
            'imu_skeleton_clip_loss': total_imu_skeleton_clip_loss / (batch + 1)
        })
        
        # 打印训练信息
        print(
            "Epoch {:05d} | lr {:.4f} | freeze_loss {:.4f} | skeleton_clip {:.4f} | imu_clip {:.4f} | multi_modal {:.4f} | imu_skeleton {:.4f} | Train Acc {:.4f} | Validation Acc {:.4f} | Best Acc {:.4f}".format(
                epoch, lr, 
                total_freeze_loss / (batch + 1),
                total_skeleton_clip_loss / (batch + 1),
                total_imu_clip_loss / (batch + 1),
                total_multi_modal_clip_loss / (batch + 1),
                total_imu_skeleton_clip_loss / (batch + 1),
                train_acc,
                valid_acc, best_acc
            )
        )
        
    # 保存最终模型
    torch.save(model.state_dict(), f'{args.path}/DiT_last_{fold_idx}.pth')

def process(args):
    mode = args.mode
    mask = args.mask
    
    dual = True if 'Contrast' in args.model or "DiT" in args.model else False
    dual = False if 'single' in args.path else dual
    direction_token = True if 'MotionBert' in args.model else False
    dataset = PoseTorchDataset('train', None, 'data/PDFEinfo.csv', 'data/output', 'data/IMU', dual=False)
    dataset_val = PoseTorchDataset('val', None, 'data/PDFEinfo.csv', 'data/output', 'data/IMU', dual=False)

    in_size = 2
    out_size = 3
    eval_name = []
    eval_fog_scores = []
    eval_fog_labels = []
    eval_freeze_preds = []
    eval_freeze_labels = []
    os.popen(f'rm {args.path}/{mode}.txt')
    for fold_idx in range(5):
        if 'DiT' in args.model:
            inp = 56
            model = model_classes[args.model](num_classes=3, in_channels=inp, learn_sigma=False).to(device)
        elif args.model == 'LDT':
            model = model_classes[args.model](num_classes=2, in_channels=16, attention_ratio=args.attention_ratio).to(device)
        train_idx, valid_idx = split_fold10(dataset, fold_idx=fold_idx)
        
        train_names = set([dataset.names[i].split('_')[0] for i in train_idx])
        valid_names = set([dataset.names[i].split('_')[0] for i in valid_idx])
        assert len(train_names.intersection(valid_names)) == 0, "训练集和验证集存在重叠样本"
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=4,
            shuffle=False
        )
        
        if mode == 'train':
            print(f"Training with DST module")
            
            wandb.init(
                project="GaitFreezing",
                config={
                "learning_rate": learnning_rate,
                "architecture": args.model,
                'lr':args.lr,
                'mask':args.mask,
                "dataset": dataset.format,
                "epochs": epochs,
                'warmup_epochs': warmup_epochs,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'save_path': args.path,
                }
            )
            print("Training...")
            
            train(train_loader, val_loader, args, model, fold_idx)
            wandb.finish()
        elif mode == 'val' or mode == 'test':
            try:
                print(f'Loading model from {args.path}/DiT_{fold_idx}_{epochs}.pth')
                state_dict = torch.load(f'{args.path}/DiT_{fold_idx}_{epochs}.pth')
            except FileNotFoundError:
                print(f'Model not found at {args.path}/DiT_{fold_idx}_{epochs}.pth, trying {args.path}/DiT_{fold_idx}.pth')
                state_dict = torch.load(f'{args.path}/DiT_{fold_idx}.pth')
                
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
                dataset = PoseTorchDataset('test', mask, data_path=args.data_path, data_format='dwpose')
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
            
        if not args.cv:
            break
            
    if mode == 'val' or mode == 'test':
        val_name_np = np.concatenate(eval_name, axis=0)
        freeze_preds = np.concatenate(eval_freeze_preds, axis=0)
        freeze_labels = np.concatenate(eval_freeze_labels, axis=0)
        
        np.save(f'{args.path}/freeze_preds.npy', freeze_preds)
        np.save(f'{args.path}/freeze_labels.npy', freeze_labels)
        
        freeze_acc = np.mean(freeze_preds == freeze_labels)
        print(f'Freeze Prediction Accuracy: {freeze_acc:.4f}')

if __name__ == "__main__":
    process(args)