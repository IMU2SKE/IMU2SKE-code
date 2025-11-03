import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from stats import *
from collections import Counter
from ROC import plot_multiclass_roc, plot_binary_roc
from stats import colormap


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def auc_load(path):
    scores = []
    labels = []
    if isinstance(path, str):
        with open(path, 'r') as file:
            lines = file.readlines()[1:]
        for line in lines:
            id_log_target = line.replace(',', '').replace('[', '').replace(']', '').split()
            score = softmax(np.array(list(map(float, id_log_target[1:-1]))))
            label = int(float(id_log_target[-1]))
            scores.append(score)
            labels.append(label)
        scores = np.array(scores)
        labels = np.array(labels)
    elif isinstance(path, tuple):
        labels = np.load(path[0])
        scores = np.load(path[1])
        print(scores.shape[0])
        for i in range(scores.shape[0]):
            scores[i] = softmax(scores[i])
    print(labels.shape, scores.shape)
    print(roc_auc_score(labels, scores[:, 1], multi_class='ovr', average='macro'))
    # plot_multiclass_roc(labels, scores, 2)
    plot_binary_roc(labels, scores[:, 1])
    
    
# confidence
def load_data(path):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
    for line in lines:
        id_log_target = line.replace(',', '').replace('[', '').replace(']', '').split()
        idx = id_log_target[0]
        logits = list(map(float, id_log_target[1:4]))
        labels = id_log_target[4]
        preds = logits.index(max(logits))
        if idx in res:
            res[idx]['preds'].append(preds)
        else:
            res[idx] = {
                'logits': logits,
                'preds': [int(preds)],
                'labels': int(labels),
            }
    for k in res:
        counter = Counter(reversed(sorted(res[k]['preds'])))
        res[k]['vote'] = counter.most_common(1)[0][0]
    #     print(idx, logits, preds, labels)
    g_res = {}
    for k in res:
        key = k[:-2]
        if key not in g_res:
            g_res[key] = {
                'preds': [res[k]['vote']],
                'labels': res[k]['labels']
            }
        else:
            g_res[key]['preds'].append(res[k]['vote'])
    for k in g_res:
        counter = Counter(reversed(sorted(g_res[k]['preds'])))
        g_res[k]['vote'] = counter.most_common(1)[0][0]
    # print(g_res)
    return g_res
        

def cal_CM(res, name=None):
    if name is not None:
        print(res, sorted(name))
        pred_label = np.array([[res[str(x)]['vote'], res[(str(x))]['labels']] for x in sorted(name)])
    else:
        pred_label = np.array([[res[x]['vote'], res[x]['labels']] for x in res])
    preds = pred_label[:, 0]
    labels = pred_label[:, 1]
    save_test(res.keys(), labels, preds)
    mae = mean_absolute_error(labels, preds)
    print(f"Mean Absolute Error (MAE): {mae}")

    # 计算PCC
    pcc, _ = pearsonr(labels, preds)
    print(f"Pearson Correlation Coefficient (PCC): {pcc}")
    cm = confusion_matrix(labels, preds)
    # 计算每个类别的敏感性
    sensitivity_per_class = np.diag(cm) / np.sum(cm, axis=1)
    print(f"Sensitivity per class: {sensitivity_per_class}")

    # 计算宏平均敏感性
    sensitivity_macro = np.mean(sensitivity_per_class)
    print(f"Macro Sensitivity: {sensitivity_macro}")
    
    # 计算每个类别的 1-Specificity
    
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    print(f"1-Specificity per class: {specificity_per_class}")

    # 计算宏平均 1-Specificity
    specificity_macro = np.mean(specificity_per_class)
    print(f"Macro 1-Specificity: {specificity_macro}")
    # print(cm)
    draw_cm(labels, preds)
    draw_score(labels, preds)
    drawdis(labels, preds)
    get_acc(labels, preds)
        

def crop_test_index(name):
    res = []
    with open('/home/tzh/Project/PD/Pose/split_dwpose_res/archive/com_liu_cheng_hou.csv', 'r') as file:
        data = file.readlines()
    for line in data[1:]:
        index, liu, cheng, hou, vote = line.strip().split(',')
        if int(index) not in name:
            continue 
        else:
            res.append(line)
    with open('/home/tzh/Project/PD/Pose/split_dwpose_res/archive/test_index.csv', 'w') as f:
        f.writelines(res)


def save_test(name, label, pred):
    res = []
    for n, l, p in zip(name, label, pred):
        res.append(f'{n},{l},{p}\n')
    with open('/home/tzh/Project/PD/Pose/split_dwpose_res/archive/model_test.csv', 'w') as f:
        f.writelines(res)


def load_com_index(name=None):
    with open('/home/tzh/Project/PD/Pose/split_dwpose_res/archive/com_liu_cheng_hou.csv', 'r') as file:
        data = file.readlines()
    labels1 = []
    labels2 = []
    for line in data[1:]:
        index, liu, cheng, hou, vote = line.strip().split(',')
        if not name is None and int(index) not in name:
            continue
        if int(vote) == -1 or int(cheng) == -1:
            continue
        labels1.append(int(vote))
        labels2.append(int(cheng))
    y_true = np.array(labels1)
    y_pred = np.array(labels2)
    draw_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    # 计算PCC
    pcc, _ = pearsonr(y_true, y_pred)
    print(f"Pearson Correlation Coefficient (PCC): {pcc}")

    
def get_test():
    data = np.load('test_subset120_2_vote.npy')
    name = [int(x) for x in data[:, 1] if not 'HY' in x]
    return name


def plot_mae_bar_chart(labels, mae_values, title='Mean Absolute Error (MAE)', ylabel='MAE', ylim=(0, 1)):
    """
    绘制均绝对误差（MAE）柱状图。

    参数:
    labels (list of str): x轴标签。
    mae_values (list of float): 每个对比的MAE值。
    title (str): 图表标题。
    ylabel (str): y轴标签。
    ylim (tuple of float): y轴范围。
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制柱状图
    bars = ax.bar(labels, mae_values, color=colormap)

    # 添加数据标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/4.0, yval, round(yval, 2), va='bottom', fontsize=16)  # va: vertical alignment

    # 添加标题和标签
    ax.set_title(title, fontsize=16)
    ax.set_ylim(ylim)  # 设置y轴的范围
    ax.set_ylabel(ylabel, fontsize=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    # 自定义x轴标签
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)

    # 显示图形
    plt.savefig(f'img_st/{ylabel.lower()}.eps')
    plt.savefig(f'img_st/{ylabel.lower()}.png')

def process(x):
    y = x[0:3]
    y.extend([sum(x[:-1])/len(x[:-1]), x[-1]])
    return y


import sys
if __name__ == '__main__':
    if not os.path.exists('results/model_metrics_on_toy_dataset'):
        os.makedirs('results/model_metrics_on_toy_dataset')
    # path = f'results/train_results/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}.txt'
    # res = load_data(path)
    # cal_CM(res)
    labels = np.load('/home/tzh/Project/GaitFreezing/results/train_results/exp/IMU2SKE_v6_mask_lr5e-3_epoch100_seed9_freezeFalse_pretrainrandom_skeletonrandom_clipTrue_dualclipTrue_imuclipTrue_skeletonclipFalse/freeze_labels.npy')
    preds = np.load('/home/tzh/Project/GaitFreezing/results/train_results/exp/IMU2SKE_v6_mask_lr5e-3_epoch100_seed9_freezeFalse_pretrainrandom_skeletonrandom_clipTrue_dualclipTrue_imuclipTrue_skeletonclipFalse/freeze_preds.npy')
    draw_cm(labels, preds)
    path = '/home/tzh/Project/GaitFreezing/results/train_results/exp/IMU2SKE_v6_mask_lr5e-3_epoch100_seed9_freezeFalse_pretrainrandom_skeletonrandom_clipTrue_dualclipTrue_imuclipTrue_skeletonclipFalse/val_0.txt'
    auc_load(path)