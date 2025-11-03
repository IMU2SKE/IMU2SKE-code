import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from skele_lib.DSTformer import DSTformer
from imu_lib.imu_models import MW2StackRNNPooling
from lib.stgcn import ST_GCN_18

from torch import einsum
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 dim=512,
                 head=8,
                 head_dim=64,
                 drop_ratio=0):
        super().__init__()
        self.head = head
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, head * head_dim)
        self.k_proj = nn.Linear(dim, head * head_dim)
        self.v_proj = nn.Linear(dim, head * head_dim)
        
        self.out = nn.Sequential(
            nn.Linear(head_dim * head, dim),
            nn.Dropout(drop_ratio)
        )
        
    def forward(self, image, text):
        b, n, _ = image.shape
        h = self.head
        
        q = self.q_proj(image)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        k = self.k_proj(text)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)

        v = self.v_proj(text)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        attn_map = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn_map = attn_map.softmax(-1)
        
        output = einsum('b h i j, b h j d -> b h i d', attn_map, v)
        
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        return self.out(output)


class CrossAttentionBlock(nn.Module):
    def __init__(self, drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.attn = CrossAttention(drop_ratio=drop)
        self.mlp = MLP(dropout=drop)
        
    def forward(self, imu, skeleton):
        imu = imu + self.attn(self.norm1(imu), self.norm2(skeleton))
        imu = imu + self.mlp(self.norm3(imu))
        return imu
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.attn = CrossAttention(drop_ratio=drop)
        self.mlp = MLP(dropout=drop)
        
    def forward(self, imu):
        imu_norm = self.norm1(imu)
        imu = imu + self.attn(imu_norm, imu_norm)
        imu = imu + self.mlp(self.norm2(imu))
        return imu


def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint['model']
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.backbone.'):
            k = k[16:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

class DSTMW2(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False,
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "DSTMW2"
        
        # 初始化特征提取器
        self.dst_extractor = DSTformer(
            maxlen=243,
            dim_feat=512,
            mlp_ratio=2,
            depth=5,
            dim_rep=512,
            num_heads=8,
            att_fuse=True
        )
        self.mw2_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        for param in self.dst_extractor.parameters():
            param.requires_grad = False
        
        # 打印冻结状态
        print(f"Encoder frozen: {self.freeze_encoder}")
        
        if self.freeze_encoder == 'True':
            for param in self.mw2_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        # 加载预训练模型
        if skeleton_pretrain_path != 'random':
            print(f"Loading skeleton pretrained model from: {skeleton_pretrain_path}")
            state_dict = torch.load(skeleton_pretrain_path)
            self.dst_extractor = load_pretrained_weights(self.dst_extractor, state_dict)
            print("Skeleton pretrained model loaded successfully")
            
        if imu_pretrain_path != 'random':
            print(f"Loading IMU pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.mw2_extractor.load_state_dict(state_dict)
            print("IMU pretrained model loaded successfully")
            
        # 初始化GRU
        self.gru = torch.nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
            
        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, skeleton, imu):
        # 打印输入张量的shape
        # print(f"Skeleton input shape: {skeleton.shape}")
        # print(f"IMU input shape: {imu.shape}")
        
        # 提取特征
        skeleton_features = self.dst_extractor.get_representation(skeleton)  # [batch, 243, 17, 512]
        
        # 在17维度上进行平均池化
        skeleton_pooled = torch.mean(skeleton_features, dim=2)  # [batch, 243, 512]
        
        # 在243维度上进行GRU处理
        _, skeleton_features = self.gru(skeleton_pooled)  # [1, batch, 512]
        skeleton_features = skeleton_features.squeeze(0)  # [batch, 512]
        
        # IMU特征提取
        imu_features = self.mw2_extractor(imu)
        
        # 打印特征提取后的shape
        # print(f"Skeleton features shape: {skeleton_features.shape}")
        # print(f"IMU features shape: {imu_features.shape}")
        
        # 合并特征
        combined_features = torch.cat((skeleton_features, imu_features), dim=1)
        # print(f"Combined features shape: {combined_features.shape}")
        
        # 分类
        logits = self.classifier(combined_features)
        # print(f"Logits shape: {logits.shape}")
        
        return {'freeze_logits': logits}


class EnhancedDSTMW2(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False,
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "EnhancedDSTMW2"
        
        # 初始化特征提取器（保持不变）
        self.dst_extractor = ST_GCN_18(in_channels=2)
        self.mw2_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder

        # 参数冻结逻辑（保持不变）  
        if self.freeze_encoder == 'True':
            for param in self.mw2_extractor.parameters():
                param.requires_grad = False

        # 预训练加载（保持不变）            
        if imu_pretrain_path != 'random':
            state_dict = torch.load(imu_pretrain_path)
            self.mw2_extractor.load_state_dict(state_dict)
        

        # 改进的融合模块 --------------------------------------------------
        # 1. 跨模态注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        
        # 2. 动态融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.Sigmoid()
        )
        
        # 3. 分层特征融合
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(512*2, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # 4. 双线性融合
        self.bilinear_fusion = nn.Bilinear(512, 512, 512)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, skeleton, imu):
        # 特征提取（保持不变）
        skeleton_features = self.dst_extractor(skeleton)
        skeleton_features = skeleton_features.squeeze(-1, -2)
        
        imu_features = self.mw2_extractor(imu)

        # 多模态融合改进部分 ----------------------------------------------
        # 维度调整 [B, 512] -> [B, 1, 512]
        skel_feat = skeleton_features.unsqueeze(1)
        imu_feat = imu_features.unsqueeze(1)
        
        # 阶段1：跨模态注意力
        attended_skel, _ = self.cross_attn(
            query=skel_feat,
            key=imu_feat,
            value=imu_feat
        )
        attended_imu, _ = self.cross_attn(
            query=imu_feat,
            key=skel_feat,
            value=skel_feat
        )
        
        # 阶段2：门控融合
        combined = torch.cat([attended_skel, attended_imu], dim=-1)
        gate = self.fusion_gate(combined)
        gated_fusion = gate * attended_skel + (1 - gate) * attended_imu
        
        # 阶段3：双线性交互
        bilinear_out = self.bilinear_fusion(
            attended_skel.squeeze(1), 
            attended_imu.squeeze(1)
        )
        
        # 阶段4：分层融合
        final_fusion = self.hierarchical_fusion(
            torch.cat([gated_fusion.squeeze(1), bilinear_out], dim=1)
        )
        
        # 分类输出
        logits = self.classifier(final_fusion)
        return {'freeze_logits': logits}


class SISD(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False,
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "SISD"
        
        # 初始化特征提取器（保持不变）
        self.static_gcn = ST_GCN_18(in_channels=2)
        self.dynamic_gcn = ST_GCN_18(in_channels=2)
        self.static_mw2 = MW2StackRNNPooling(size_embeddings=512)
        self.dynamic_mw2 = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder

        # 参数冻结逻辑（保持不变）  
        if self.freeze_encoder == 'True':
            for param in self.static_mw2.parameters():
                param.requires_grad = False

        # 预训练加载（保持不变）            
        if imu_pretrain_path != 'random':
            state_dict = torch.load(imu_pretrain_path)
            self.static_mw2.load_state_dict(state_dict)
        

        # 改进的融合模块 --------------------------------------------------
        # 1. 跨模态注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 2. 动态融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. 分层特征融合
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )
        
        # 4. 双线性融合
        self.bilinear_fusion = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, dynamic_ske, static_ske, dynamic_imu, static_imu):
        # 特征提取（保持不变）
        dynamic_ske_features = self.dynamic_gcn(dynamic_ske)
        dynamic_ske_features = dynamic_ske_features.squeeze(-1, -2)
        static_ske_features = self.static_gcn(static_ske)
        static_ske_features = static_ske_features.squeeze(-1, -2)
        
        dynamic_imu_features = self.dynamic_mw2(dynamic_imu)
        static_imu_features = self.static_mw2(static_imu)
        
        skeleton_features = dynamic_ske_features + static_ske_features
        imu_features = dynamic_imu_features + static_imu_features

        # 多模态融合改进部分 ----------------------------------------------
        # 维度调整 [B, 512] -> [B, 1, 512]
        skel_feat = skeleton_features.unsqueeze(1)
        imu_feat = imu_features.unsqueeze(1)
        
        # 阶段1：跨模态注意力
        attended_skel, _ = self.cross_attn(
            query=skel_feat,
            key=imu_feat,
            value=imu_feat
        )
        attended_imu, _ = self.cross_attn(
            query=imu_feat,
            key=skel_feat,
            value=skel_feat
        )
        
        # 阶段2：门控融合
        combined = torch.cat([attended_skel, attended_imu], dim=-1)
        gate = self.fusion_gate(combined)
        gated_fusion = gate * attended_skel + (1 - gate) * attended_imu
        
        # 阶段3：双线性交互
        bilinear_out = self.bilinear_fusion(
            attended_skel.squeeze(1), 
            attended_imu.squeeze(1)
        )
        
        # 阶段4：分层融合
        final_fusion = self.hierarchical_fusion(
            torch.cat([gated_fusion.squeeze(1), bilinear_out], dim=1)
        )
        
        # 分类输出
        logits = self.classifier(final_fusion)
        return {'freeze_logits': logits}


class SISDv2(pl.LightningModule):
    def __init__(self, hidden_dim=1024, num_classes=2, freeze_encoder=False,
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "SISDv2"
        
        # 初始化特征提取器（保持不变）
        self.static_gcn = ST_GCN_18(in_channels=2)
        self.dynamic_gcn = ST_GCN_18(in_channels=2)
        self.static_mw2 = MW2StackRNNPooling(size_embeddings=512)
        self.dynamic_mw2 = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder

        # 参数冻结逻辑（保持不变）  
        if self.freeze_encoder == 'True':
            for param in self.static_mw2.parameters():
                param.requires_grad = False

        # 预训练加载（保持不变）            
        if imu_pretrain_path != 'random':
            state_dict = torch.load(imu_pretrain_path)
            self.static_mw2.load_state_dict(state_dict)
        

        # 改进的融合模块 --------------------------------------------------
        self.mm_fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//8),
            nn.ReLU(),
            nn.Linear(hidden_dim//8, 2),
            nn.Softmax(dim=-1)
        )
        
        # 共享参数的投影层
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)        
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def _fuse_features(self, dynamic, static, fusion_module):
        """
        优化后的融合策略：
        1. 特征对齐：通过共享投影层统一特征空间
        2. 动态权重：生成模态特定融合系数
        3. 残差连接：保留原始特征信息
        """
        # 投影到共享空间 [B, D]
        proj_dynamic = self.projection(dynamic)
        proj_static = self.projection(static)
        
        # 生成融合权重 [B, 2]
        weights = fusion_module(torch.cat([proj_dynamic, proj_static], dim=-1))
        
        # 加权融合
        fused = weights[:, 0:1] * dynamic + weights[:, 1:2] * static
        
        # 残差连接
        return fused + (dynamic + static)/2  # 原始方法作为残差基底

    def forward(self, dynamic_ske, static_ske, dynamic_imu, static_imu):
        # 特征提取（保持不变）
        dynamic_ske_features = self.dynamic_gcn(dynamic_ske)
        dynamic_ske_features = dynamic_ske_features.squeeze(-1, -2)
        static_ske_features = self.static_gcn(static_ske)
        static_ske_features = static_ske_features.squeeze(-1, -2)
        
        dynamic_imu_features = self.dynamic_mw2(dynamic_imu)
        static_imu_features = self.static_mw2(static_imu)
        
        skeleton_features = torch.cat([dynamic_ske_features, static_ske_features], dim=1)
        imu_features = torch.cat([dynamic_imu_features, static_imu_features], dim=1)
        
        mm_features = self._fuse_features(
            skeleton_features,
            imu_features,
            self.mm_fusion
        )

        # 分类输出
        logits = self.classifier(mm_features)
        return {'freeze_logits': logits}


class SKE2IMU(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False,
                 skeleton_pretrain_path='random', imu_pretrain_path='random',
                 fusion_weight=0.3):  # 可调节的融合权重
        super().__init__()
        self.name = "EnhancedFoGClassifier"
        
        # 保持原有的IMU特征提取器不变
        self.imu_extractor = MW2StackRNNPooling(size_embeddings=512)
        
        # 添加骨骼特征提取器
        self.skeleton_extractor = ST_GCN_18(in_channels=2)
        
        # 可学习的门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_weight = fusion_weight
        self.freeze_encoder = freeze_encoder
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))
        
        # 加载预训练模型
        if self.freeze_encoder == 'True':
            for param in self.imu_extractor.parameters():
                param.requires_grad = False
            print("IMU encoder frozen successfully")
            
        if imu_pretrain_path != 'random':
            print(f"Loading IMU pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.imu_extractor.load_state_dict(state_dict)
            print("IMU pretrained model loaded successfully")
            
        # 保持原有的分类器结构
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
        # 用于渐进式训练的开关
        self.use_skeleton = False
        
    def forward(self, skeleton, batch):
        # 获取IMU特征
        imu_features = self.imu_extractor(batch)
        
        # 如果没有开启骨骼模态或没有提供骨骼数据，直接使用IMU特征
        if not self.use_skeleton or skeleton is None:
            return {'freeze_logits': self.classifier(imu_features)}
        
        # 提取骨骼特征
        skeleton_features = self.skeleton_extractor(skeleton).squeeze(-2).squeeze(-1)
        
        # 计算融合权重
        features_concat = torch.cat([imu_features, skeleton_features], dim=-1)
        gate_weights = self.fusion_gate(features_concat)
        
        # 加权融合特征
        fused_features = (gate_weights[:, 0].unsqueeze(-1) * imu_features + 
                         gate_weights[:, 1].unsqueeze(-1) * skeleton_features)
        
        # 渐进式融合
        final_features = (1 - self.fusion_weight) * imu_features + self.fusion_weight * fused_features
        
        logits = self.classifier(final_features)
        return {
            'freeze_logits': logits,
            'imu_features': imu_features,
            'skeleton_features': skeleton_features,
            'gate_weights': gate_weights
        }
    
    def set_fusion_weight(self, weight):
        """设置融合权重"""
        self.fusion_weight = weight
    
    def enable_skeleton(self, enable=True):
        """启用或禁用骨骼模态"""
        self.use_skeleton = enable


class SKE2IMU_old(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                 skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "SKE2IMU"
        
        # Feature extractor
        self.skeleton_extractor = ST_GCN_18(in_channels=2)
        self.feature_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        # 打印冻结状态
        print(f"Encoder frozen: {self.freeze_encoder}")
        
        if self.freeze_encoder == 'True':
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        if imu_pretrain_path != 'random':
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.feature_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        else:
            print("No pretrained model specified, using random initialization")

        self.encoder = nn.ModuleList([
            SelfAttentionBlock(drop=0.2),
            CrossAttentionBlock(drop=0.2),
            SelfAttentionBlock(drop=0.2),
            CrossAttentionBlock(drop=0.2),]
        )
        
        self.gru = torch.nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.position_embedding = nn.Parameter(torch.arange(60).unsqueeze(0).unsqueeze(-1).float())
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, skeleton, imu):
        skeleton_features = self.skeleton_extractor(skeleton, return_t=True).squeeze(-1).transpose(-1, -2)  # b n d
        imu_features = self.feature_extractor(imu, return_t=True)  # b m d
        pos_skeleton = skeleton_features + self.position_embedding[:, :skeleton_features.shape[1], :]
        pos_imu = imu_features + self.position_embedding[:, :imu_features.shape[1], :]
        for i in range(len(self.encoder)):
            if i % 2 == 1:
                features = self.encoder[i](features, pos_skeleton)
            else:
                if i == 0:
                    features = self.encoder[i](pos_imu)
                else:
                    features = self.encoder[i](features)
        fused_features = self.gru(features)[1][0]
        logits = self.classifier(fused_features)
        return {'freeze_logits': logits,
                'skeleton_features': skeleton_features.mean(dim=1),
                'imu_features': imu_features.mean(dim=1)}


class IMU2SKE(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                 skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "IMU2SKE"
        
        # Feature extractor
        self.skeleton_extractor = ST_GCN_18(in_channels=2)
        self.feature_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        # 打印冻结状态
        print(f"Encoder frozen: {self.freeze_encoder}")
        
        if self.freeze_encoder == 'True':
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        if imu_pretrain_path != 'random':
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.feature_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        else:
            print("No pretrained model specified, using random initialization")

        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, skeleton, imu):
        skeleton_features = self.skeleton_extractor(skeleton).squeeze(-1, -2)
        features = self.feature_extractor(imu)
        # fused_features = torch.cat([skeleton_features, features], dim=1)
        # fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(skeleton_features)
        return {'freeze_logits': logits,
                'skeleton_features': skeleton_features,
                'imu_features': features}

class EnhancedAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return x + self.adapter(x)

class IMU2SKEA(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                 skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "IMU2SKEA"
        
        # Feature extractor
        self.skeleton_extractor = ST_GCN_18(in_channels=2)
        self.feature_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        self.ada = EnhancedAdapter()
        
        # 打印冻结状态
        
        if self.freeze_encoder == 'True':
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        if imu_pretrain_path != 'random':
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.feature_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        else:
            print("No pretrained model specified, using random initialization")

        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, skeleton, imu):
        skeleton_features = self.skeleton_extractor(skeleton).squeeze(-1, -2)
        features = self.feature_extractor(imu)
        print(f"IMU features shape: {features.shape}")
        features = self.ada(features)
        # fused_features = torch.cat([skeleton_features, features], dim=1)
        # fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(skeleton_features)
        return {'freeze_logits': logits,
                'skeleton_features': skeleton_features,
                'imu_features': features}


class IMULIP(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                 skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "IMULIP"
        
        # Feature extractor
        self.skeleton_extractor = ST_GCN_18(in_channels=2)
        self.feature_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        # 打印冻结状态
        print(f"Encoder frozen: {self.freeze_encoder}")
        
        if self.freeze_encoder == 'True':
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        if imu_pretrain_path != 'random':
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.feature_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        else:
            print("No pretrained model specified, using random initialization")
        
        self.ca = CrossAttention(embed_size=512, num_heads=8)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, skeleton, imu):
        skeleton_features = self.skeleton_extractor(skeleton).squeeze(-1, -2)
        features = self.feature_extractor(imu)
        
        fused_features = self.ca(features, skeleton_features)
        logits = self.classifier(fused_features)
        return {'freeze_logits': logits,
                'skeleton_features': skeleton_features,
                'imu_features': features}
