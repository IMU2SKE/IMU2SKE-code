import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# from dataset import PoseTorchDataset

def plot_comparison_skeleton_2d(ax, norm_skeleton, ori_skeleton, joint_pairs, xlim, ylim, format='dwpose'):
    """
    绘制对比骨架的单帧图像。
    
    参数:
    - ax: matplotlib的Axes对象
    - norm_skeleton: numpy array of shape (num_joints, 2)
    - ori_skeleton: numpy array of shape (num_joints, 2)
    - joint_pairs: list of tuples，每个元组包含父关节和子关节的索引
    - xlim: tuple，x轴范围
    - ylim: tuple，y轴范围
    - format: str，骨架格式
    """
    ax.clear()
    for joint_pair in joint_pairs:
        joint_1, joint_2 = joint_pair
        # 绘制 norm 骨架
        x_values_norm = [norm_skeleton[joint_1, 0], norm_skeleton[joint_2, 0]]
        y_values_norm = [norm_skeleton[joint_1, 1], norm_skeleton[joint_2, 1]]
        if not ((x_values_norm[0] <= 0.2 and y_values_norm[0] <= 0.2) or 
                (x_values_norm[1] <= 0.2 and y_values_norm[1] <= 0.2)):
            ax.plot(x_values_norm, y_values_norm, 'b-', marker='o', label='Norm' if joint_pair == joint_pairs[0] else "")
        
        # 绘制 ori 骨架
        x_values_ori = [ori_skeleton[joint_1, 0], ori_skeleton[joint_2, 0]]
        y_values_ori = [ori_skeleton[joint_1, 1], ori_skeleton[joint_2, 1]]
        if not ((x_values_ori[0] <= 0.2 and y_values_ori[0] <= 0.2) or 
                (x_values_ori[1] <= 0.2 and y_values_ori[1] <= 0.2)):
            ax.plot(x_values_ori, y_values_ori, 'r-', marker='x', label='Ori' if joint_pair == joint_pairs[0] else "")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()  # 翻转Y轴


def animate_comparison(i, norm_data, ori_data, joint_pairs, ax, xlim, ylim, format='dwpose'):
    """
    更新动画帧。
    
    参数:
    - i: 当前帧索引
    - norm_data: numpy array of shape (num_frames, num_joints, 2)
    - ori_data: numpy array of shape (num_frames, num_joints, 2)
    - joint_pairs: list of tuples，骨架的连接关系
    - ax: matplotlib的Axes对象
    - xlim: tuple，x轴范围
    - ylim: tuple，y轴范围
    - format: str，骨架格式
    """
    norm_skeleton = norm_data[0, i]
    ori_skeleton = ori_data[0, i]
    plot_comparison_skeleton_2d(ax, norm_skeleton, ori_skeleton, joint_pairs, xlim, ylim, format)
    ax.set_title(f'Frame {i}')

def save_comparison_skeleton_animation(norm_data, ori_data, save_path_mp4='comparison_skeleton_animation.mp4', 
                                      save_path_gif='comparison_skeleton_animation.gif', 
                                      joint_pairs=None, xlim=(0, 1), ylim=(0, 1), format='dwpose'):
    """
    保存 norm 和 ori 数据的对比动画。
    
    参数:
    - norm_data: numpy array of shape (num_frames, num_joints, 2)
    - ori_data: numpy array of shape (num_frames, num_joints, 2)
    - save_path_mp4: str，保存MP4文件的路径
    - save_path_gif: str，保存GIF文件的路径
    - joint_pairs: list of tuples，每个元组包含父关节和子关节的索引
    - xlim: tuple，x轴范围
    - ylim: tuple，y轴范围
    - format: str，骨架格式（'dwpose'或'openpose'）
    """
    if joint_pairs is None:
        if format == 'dwpose':
            joint_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7), 
                (1, 11), (11, 12), (12, 13), (13, 14), (13, 15), (13, 16),
                (1, 8), (8, 9), (9, 10), (10, 17), (10, 18), (10, 19)
            ]
        elif format == 'openpose':
            joint_pairs = [
                (0, 1),  (0, -1), (0, -2),  # 鼻子到脖子
                (1, 2), (2, 3), (3, 4),      # 右肩到右肘到右手腕
                (1, 5), (5, 6), (6, 7),      # 左肩到左肘到左手腕
                (1, 8), (8, 9), (9, 10),     # 右髋到右膝到右脚踝
                (1, 11), (11, 12), (12, 13), # 左髋到左膝到左脚踝
            ]
    
    num_frames = norm_data.shape[1]
    fig, ax = plt.subplots()
    
    ani = FuncAnimation(fig, animate_comparison, frames=num_frames, 
                        fargs=(norm_data, ori_data, joint_pairs, ax, xlim, ylim, format), interval=200)
    
    # 保存动画为MP4和GIF
    ani.save(save_path_gif, writer=PillowWriter(fps=25))
    
    plt.close(fig)  # 关闭图形以避免在交互环境中显示
    return save_path_mp4, save_path_gif

    
def plot_skeleton_2d(ax, skeleton, joint_pairs, xlim, ylim, format='dwpose'):
    ax.clear()
    for joint_pair in joint_pairs:
        joint_1, joint_2 = joint_pair
        x_values = [skeleton[joint_1, 0], skeleton[joint_2, 0]]
        y_values = [skeleton[joint_1, 1], skeleton[joint_2, 1]]
        if (x_values[0] <= 0.2 and y_values[0] <= 0.2) or (x_values[1] <= 0.2 and y_values[1] <= 0.2):
            continue
        ax.plot(x_values, y_values, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()  # 翻转Y轴

def animate(i, data, joint_pairs, ax, xlim, ylim, format='dwpose'):
    batch_index = 0  # assuming batch size is 1
    skeleton = data[batch_index, i]
    plot_skeleton_2d(ax, skeleton, joint_pairs, xlim, ylim, format)
    ax.set_title(f'Frame {i}')

def save_skeleton_animation(data, save_path_mp4='skeleton_animation.mp4', save_path_gif='skeleton_animation.gif', 
                            joint_pairs=None, xlim=(0, 1), ylim=(0, 1), format='dwpose'):
    if joint_pairs is None:
        if format == 'dwpose':
            joint_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7), 
                (1, 11), (11, 12), (12, 13), (13, 14), (13, 15), (13, 16),
                (1, 8), (8, 9), (9, 10), (10, 17), (10, 18), (10, 19)
            ]
        elif format == 'openpose':
            joint_pairs = [
                (0, 1),  (0, -1), (0, -2),# 鼻子到脖子
                (1, 2), (2, 3), (3, 4),  # 右肩到右肘到右手腕
                (1, 5), (5, 6), (6, 7),  # 左肩到左肘到左手腕
                (1, 8), (8, 9), (9, 10),  # 右髋到右膝到右脚踝
                (1, 11), (11, 12), (12, 13),  # 左髋到左膝到左脚踝
            ]
    num_frames = data.shape[1]  # 动态设置帧数
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=num_frames, fargs=(data, joint_pairs, ax, xlim, ylim, format), interval=200)

    # Save the animation as MP4
    # ani.save(save_path_mp4, writer=FFMpegWriter(fps=25))

    # Optionally, save the animation as GIF
    ani.save(save_path_gif, writer=PillowWriter(fps=25))

    plt.close(fig)  # Close the figure to avoid displaying it in interactive environments
    return save_path_mp4, save_path_gif

if __name__ == '__main__':
    dataset = PoseTorchDataset(mode='train', mask='all', dual=False, datanum=120, data_format='openpose')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for data in dataloader:
        attr = data['attr'].numpy()  # Convert to numpy array
        print(f'Loaded data shape: {attr.shape}')  # 打印数据形状以检查帧数
        break  # Load only the first batch for this example

    save_path_mp4 = 'skeleton_animation.mp4'
    save_path_gif = 'skeleton_animation.gif'

    save_skeleton_animation(attr, save_path_mp4, save_path_gif, format='openpose')
    print(f'Animation saved to {save_path_mp4} and {save_path_gif}')
