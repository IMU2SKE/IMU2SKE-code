# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict


def truncated_normal_(tensor, mean=0, std=0.09):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class AttentionPooling(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.weight = torch.nn.Conv1d(
            in_channels=input_channels, out_channels=1, kernel_size=1
        )

    def forward(self, batch):
        weights = torch.softmax(self.weight(batch), dim=-1)
        return (weights * batch).sum(dim=-1)


class AttentionPooledIMUEncoder(pl.LightningModule):
    """
    Input: [N x n_channels x n_steps]
    Output:
        - forward: [N x n_embeddings]
    """

    def __init__(
        self,
        in_channels=6,
        out_channels=24,
        kernel_size=10,
        dilation=2,
        size_embeddings=512,
        initialize_weights=True,
    ):

        print("Initializing AttentionPooledIMUEncoder ...")
        super(AttentionPooledIMUEncoder, self).__init__()
        self.name = AttentionPooledIMUEncoder

        self.encoder = nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.LeakyReLU(),
            AttentionPooling(input_channels=out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, size_embeddings),
        )
        if initialize_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                truncated_normal_(m.weight, 0, 0.02)
                truncated_normal_(m.bias, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                truncated_normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y_hat = self.encoder(x)
        return y_hat


class TimeDistributedIMUEncoder(pl.LightningModule):
    """
    Input: [N x n_channels x n_steps]
    Output:
        - forward_time_distributed: (
            [N x n_frames x size_embeddings],
            [N x size_embeddings]
        )
        - forward: [N x n_classes]
    """

    def __init__(
        self, n_frames=10, n_channels=6, n_steps_per_frame=128, size_embeddings=512
    ):

        # print("Initializing TimeDistributedIMUEncoder ...")

        super(TimeDistributedIMUEncoder, self).__init__()

        self.name = TimeDistributedIMUEncoder
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.n_steps_per_frame = n_steps_per_frame
        self.size_embeddings = size_embeddings

        self.time_distributed_signal_encoder = nn.Sequential(
            OrderedDict(
                [
                    # x: N x n_channels x n_steps
                    (
                        "conv1",
                        nn.Conv1d(self.n_channels, self.n_channels, 50, stride=1),
                    ),
                    ("relu1", nn.ReLU()),
                    # x: N x 1 x n_steps
                    ("conv2", nn.Conv1d(self.n_channels, 1, 10)),
                    ("relu2", nn.ReLU()),
                    # x: N x 1 x (n_frames * n_steps_per_frame)
                    (
                        "pool",
                        nn.AdaptiveAvgPool1d(self.n_frames * self.n_steps_per_frame),
                    ),
                ]
            )
        )

        self.rnn = nn.Sequential(
            OrderedDict(
                [
                    # x: N x n_frames x size_embeddings
                    (
                        "gru",
                        nn.GRU(
                            input_size=self.n_steps_per_frame,
                            hidden_size=self.size_embeddings,
                            num_layers=1,
                            batch_first=True,
                        ),
                    ),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    # x: N x n_frames x size_embeddings
                    ("linear", nn.Linear(self.size_embeddings, self.size_embeddings)),
                ]
            )
        )

    def forward_time_distributed(self, x):
        # x: N x 1 x (n_frames * n_steps_per_frame)
        x = self.time_distributed_signal_encoder(x)

        # x: N x (n_frames * n_steps_per_frame)
        x = x.reshape((x.shape[0], x.shape[-1]))

        # x: N x n_frames x n_steps_per_frame
        x = x.unflatten(1, (self.n_frames, self.n_steps_per_frame))

        # x:  N x n_frames x size_embeddings
        # hn: N x size_embeddings
        x, hn = self.rnn(x)
        return (x, hn)

    def forward(self, x):
        _, hn = self.forward_time_distributed(x)
        y_hat = self.classifier(hn[0])
        return y_hat


class PatchTransformer(pl.LightningModule):
    """
    Transformer based encoder for IMU.
    Increasing patch_size decrease the sequence length.
    """

    def __init__(
        self,
        patch_size: int = 1,
        size_embeddings: int = 128,
        nhead: int = 1,
        ff_hidden_size: int = 128,
        layers: int = 1,
        cls_token: bool = True,
    ):
        """
        patch_size: as in ViT, split the 6xd tensor
                    in patches of 6xpatch_size. In
                    this case we get 1D patches.
        size_embeddings: embedding size.
        nhead: transformer heads.
        ff_hidden_size: feedforward model size.
        layers: number tranformer layers layers
        cls_token: bool, to return a single [CLS]
                   token as in BERT/RoBERTa. If
                   False, return the average of the
                   embeddings.
        """
        super().__init__()
        self.name = PatchTransformer

        self.cls_token = cls_token
        self.imu_patch_size = patch_size
        imu_patch_dims = self.imu_patch_size * 6

        # number of channel in the signal ==> 6
        self.imu_token_embed = torch.nn.Linear(
            imu_patch_dims, size_embeddings, bias=False
        )

        self.model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=size_embeddings,
                nhead=nhead,
                dim_feedforward=ff_hidden_size,
                batch_first=True,
            ),
            num_layers=layers,
        )

        self.cls = torch.nn.Parameter(torch.zeros(1, 1, size_embeddings))

    def forward(self, batch):
        bsz = batch.shape[0]
        # this create the patches
        x = batch.unfold(-1, self.imu_patch_size, self.imu_patch_size).permute(
            0, 2, 1, 3
        )
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
        # create embeddings for every patch
        x = self.imu_token_embed(x)

        # cat the [CLS] embedding
        if self.cls_token:
            x = torch.cat((self.cls.expand(bsz, -1, -1), x), dim=1)

        # sequence modelling step
        outputs = self.model(x)

        # return the [CLS] embedding in position zero
        if self.cls_token:
            return outputs[:, 0, :]
        else:
            return torch.mean(outputs, dim=1)


class PatchRNN(pl.LightningModule):
    """
    RNN based encoder for IMU.
    Increasing patch_size decrease the sequence length.
    """

    def __init__(
        self,
        patch_size: int = 1,
        size_embeddings: int = 128,
        layers: int = 1,
        bidirectional: bool = True,
    ):
        """
        patch_size: as in ViT, split the 6xd tensor
                    in patches of 6xpatch_size. In
                    this case we get 1D patches.
        size_embeddings: embedding size.
        ff_hidden_size: feedforward model size.
        layers: number RNN layers layers
        bidirectional: bidir-RNN
        """
        super().__init__()
        self.name = PatchRNN
        self.imu_patch_size = patch_size
        self.bidirectional = bidirectional
        imu_patch_dims = self.imu_patch_size * 6

        if bidirectional:
            if size_embeddings % 2 == 0:
                size_embeddings = int(size_embeddings / 2)
            else:
                size_embeddings = int((size_embeddings - 1) / 2)

        # number of channel in the signal ==> 6
        self.imu_token_embed = torch.nn.Linear(
            imu_patch_dims, size_embeddings, bias=False
        )

        self.gru = torch.nn.GRU(
            batch_first=True,
            input_size=size_embeddings,
            hidden_size=size_embeddings,
            bidirectional=bidirectional,
            num_layers=layers,
        )

    def forward(self, batch):
        # create patches of imu_patch_size size
        x = batch.unfold(-1, self.imu_patch_size, self.imu_patch_size).permute(
            0, 2, 1, 3
        )
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
        # create embeddings for every patch
        x = self.imu_token_embed(x)

        # sequence modelling step
        _, state = self.gru(x)

        # merging last hidden states since it is bi-dir
        if self.bidirectional:
            state = torch.cat((state[0, :, :], state[1, :, :]), dim=1)
        else:
            state = state[0]
        return state


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            torch.nn.MaxPool1d(kernel_size=3),
        )

    def forward(self, batch):
        return self.net(batch)


class MW2StackRNNPooling(pl.LightningModule):
    def __init__(self, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )

    def forward(self, batch, return_t=False):
        if return_t:
            return self.net(batch)[0]
        # return the last hidden state
        return self.net(batch)[1][0]

class FoGClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                 skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "FoGClassifier"
        
        # Feature extractor
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
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, batch):
        features = self.feature_extractor(batch)
        logits = self.classifier(features)
        return {'freeze_logits': logits,
                'imu_features': features}


class SDFoG(pl.LightningModule):
    def __init__(self, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        self.name = "SDFoG"
        
        # Feature extractor
        self.static_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.dynamic_extractor = MW2StackRNNPooling(size_embeddings=512)
        self.freeze_encoder = freeze_encoder
        
        # 打印冻结状态
        print(f"Encoder frozen: {self.freeze_encoder}")
        
        if self.freeze_encoder == 'True':
            for param in self.static_extractor.parameters():
                param.requires_grad = False
            for param in self.dynamic_extractor.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen successfully")
            
        if imu_pretrain_path != 'random' and imu_pretrain_path.startswith('data'):
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path)
            self.static_extractor.load_state_dict(state_dict)
            # self.dynamic_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        elif imu_pretrain_path != 'random' and imu_pretrain_path.startswith('both'):
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path[4:])
            self.static_extractor.load_state_dict(state_dict)
            self.dynamic_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        elif imu_pretrain_path != 'random' and imu_pretrain_path.startswith('dynamic'):
            # 打印预训练加载信息
            print(f"Loading pretrained model from: {imu_pretrain_path}")
            state_dict = torch.load(imu_pretrain_path[7:])
            # self.static_extractor.load_state_dict(state_dict)
            self.dynamic_extractor.load_state_dict(state_dict)
            print("Pretrained model loaded successfully")
        else:
            print("No pretrained model specified, using random initialization")
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, imu, limu):
        static_features = self.static_extractor(imu)
        dynamic_features = self.dynamic_extractor(limu)
        combined_features = torch.cat((static_features, dynamic_features), dim=1)
        logits = self.classifier(combined_features)
        return {'freeze_logits': logits}

        

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_dim=6, hidden_dim=512, num_classes=2, freeze_encoder=False, 
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        # LSTM层：input_dim=6，hidden_dim=512，batch_first=True表示输入的shape是 (batch_size, seq_len, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 分类器：全连接层将LSTM的输出映射到类别
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x形状为 (batch_size, 1000, 6)
        
        # LSTM的输出是 (batch_size, seq_len, hidden_dim)
        # 我们使用最后一个时间步的隐藏状态进行分类
        _, (hn, _) = self.lstm(x)  # hn 是最后一个时间步的隐藏状态，形状 (num_layers, batch_size, hidden_dim)
        
        # 取最后一个时间步的隐藏状态作为特征输入分类器
        x = hn[-1, :, :]  # 取最后一层LSTM的隐藏状态 (batch_size, hidden_dim)
        x = self.fc(x)  # 分类层输出预测结果
        return {'freeze_logits': x}


class TCNClassifier(pl.LightningModule):
    def __init__(self, input_dim=6, num_classes=2, hidden_dim=512, freeze_encoder=False, 
                skeleton_pretrain_path='random', imu_pretrain_path='random'):
        super().__init__()
        # 第一层1D卷积，输入维度6，输出维度为hidden_dim，kernel_size为3，padding为1
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # 第二层1D卷积
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # 分类器：全连接层将卷积后的特征映射到类别
        self.fc1 = nn.Linear(hidden_dim * 1000, 256)  # 由于没有池化，输出特征维度保持一致
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x形状为 (batch_size, 6, 1000)
        # 转置为 (batch_size, 1000, 6) 适应Conv1D
        # x = x.permute(0, 2, 1)  # (batch_size, 1000, 6) -> (batch_size, 6, 1000)
        
        # 第一层卷积
        x = torch.relu(self.conv1(x))
        
        # 第二层卷积
        x = torch.relu(self.conv2(x))
        
        # 展平特征
        x = x.view(x.size(0), -1)  # (batch_size, hidden_dim * 1000)
        
        # 分类器
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return {'freeze_logits': x}