import os
import torch
import argparse


# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='这是一个示例程序')

# 添加参数
parser.add_argument('--mode', type=str, help='train or test')
parser.add_argument('--mask', type=str, help='mask joint index')
parser.add_argument('--data-path', type=str, help='train or test path')
parser.add_argument('--finetune', type=str, help='ckp path')
parser.add_argument('--drop_rate', type=float, default=.0)
parser.add_argument('--attn_drop_rate', type=float, default=.0)
parser.add_argument('--drop_path_rate', type=float, default=.0)
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cv', type=str, default='False', choices=['True', 'False'],
                  help='whether to use cross validation')
parser.add_argument('--distribute', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--weight', type=float, default=100)
parser.add_argument('--d_weight', type=float, default=1)
parser.add_argument('--model', type=str, default='ContrastSTGCN')
parser.add_argument('--use', nargs='+', default=['all'],
                   choices=['all', 'none', 'use_skeleton', 'use_imu', 'use_multi', 'use_imu_skeleton'],
                   help='ablation study: all - use all losses, none - use no additional losses, use_skeleton - use skeleton_clip_loss, etc.')
parser.add_argument('--attention_ratio', type=float, default=1.0,
                   choices=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
                   help='ratio of linear attention to softmax attention (0.0: all softmax, 1.0: all linear)')
parser.add_argument('--freeze-encoder', type=str, default='False', choices=['True', 'False'],
                   help='whether to freeze encoder weights during training')
parser.add_argument('--pretrain-path', type=str, default='',
                   help='pretrained encoder path')
parser.add_argument('--skele-path', type=str, default='',
                   help='pretrained encoder path')
parser.add_argument('--clip', type=str, default='False', choices=['True', 'False'],
                   help='whether to clip the input data')
parser.add_argument('--dual-clip', type=str, default='False', choices=['True', 'False'],
                   help='whether to clip the input data')
parser.add_argument('--imu-clip', type=str, default='False', choices=['True', 'False'],
                   help='whether to clip the input data')
parser.add_argument('--skeleton-clip', type=str, default='False', choices=['True', 'False'],
                   help='whether to clip the input data')

# 将字符串参数转换为bool类型
args = parser.parse_args()
args.cv = args.cv == 'True'
args.freeze_encoder = args.freeze_encoder == 'True'


# 解析命令行参数
args = parser.parse_args()
if not args.path.startswith('/'):
    args.path = 'results/train_results/' + args.path if not args.path == './' else args.path
    os.makedirs(args.path, exist_ok=True)