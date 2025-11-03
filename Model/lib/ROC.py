import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

# colormap = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']
colormap = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#ff8884']

def plot_multiclass_roc(y_true, y_scores, n_classes):
    """
    绘制多分类问题的ROC曲线
    
    参数:
    y_true (array): 真实标签
    y_scores (array): 预测概率
    n_classes (int): 类别数量
    """
    # Binarize the output
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(7, 7))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=colormap[i], lw=2,
                 label='updrs {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color=colormap[3], linestyle=':', linewidth=2,
             label='average (area = {0:0.2f})'.format(roc_auc["micro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic for Multiclass', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    # plt.savefig('results/model_metrics_on_toy_dataset/roc.eps')
    plt.savefig('results/model_metrics_on_toy_dataset/roc.png')

def plot_binary_roc(y_true, y_scores):
    """
    绘制二分类问题中两个类别的ROC曲线

    参数:
    y_true (array): 真实标签 (0 or 1)
    y_scores (array): 预测概率 (for the positive class, Class 1)
    """
    # Compute ROC curve and AUC for Class 1 (Positive Class)
    fpr_class1, tpr_class1, _ = roc_curve(y_true, y_scores)
    roc_auc_class1 = auc(fpr_class1, tpr_class1)

    # Compute ROC curve and AUC for Class 0 (Negative Class)
    y_true_class0 = 1 - y_true  # Invert labels for Class 0
    y_scores_class0 = 1 - y_scores  # Invert scores for Class 0
    fpr_class0, tpr_class0, _ = roc_curve(y_true_class0, y_scores_class0)
    roc_auc_class0 = auc(fpr_class0, tpr_class0)

    # Plot ROC curves
    plt.figure(figsize=(7, 7))
    plt.plot(fpr_class1, tpr_class1, color=colormap[0], lw=2,
             label='Class 1 (area = {0:0.2f})'.format(roc_auc_class1))
    plt.plot(fpr_class0, tpr_class0, color=colormap[1], lw=2,
             label='Class 0 (area = {0:0.2f})'.format(roc_auc_class0))

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic for Binary Classification', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig('results/model_metrics_on_toy_dataset/roc.eps')
    plt.savefig('results/model_metrics_on_toy_dataset/roc.png')
