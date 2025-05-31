import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
plt.rcParams['font.family'] = 'SimHei'

def load_data(train, test):
    # 读取训练集
    train_labels = train.iloc[:, 0].values.astype(np.uint8)
    train_images = train.iloc[:, 1:].values.astype(np.uint8)

    # 读取测试集 
    test_labels = test.iloc[:, 0].values.astype(np.uint8)
    test_images = test.iloc[:, 1:].values.astype(np.uint8)

    return (train_images, train_labels), (test_images, test_labels)

def load_pred_data():
    test = pd.read_csv("./dataset/fashion-mnist_test.csv")
    test_labels_true = test.iloc[:, 0].values
    results_dir = './预测结果'
    test_labels_pred = {}
    file_pattern = os.path.join(results_dir, '*_results.csv')
    result_files = glob.glob(file_pattern)
    # 遍历所有结果文件
    for file_path in result_files:
        filename = os.path.basename(file_path)
        model_name = filename.replace('_results.csv', '')
        df = pd.read_csv(file_path)
        predictions = df['预测标签'].tolist()
        test_labels_pred[model_name] = predictions
    print("\n===== 预测结果加载完成 =====")
    print()
    print(f"共加载 {len(test_labels_pred)} 个模型的预测结果")
    print("模型列表:", list(test_labels_pred.keys()))
    return test_labels_true, test_labels_pred

def svm_plot_roc(best_model,test_images,test_labels,title):
    class_names = [' T恤', '裤子', '套衫', '连衣裙', '外套','凉鞋', '衬衫', '运动鞋', '包', '踝靴']
    test_prob = best_model.predict_proba(test_images) 
    # 二值化真实标签
    n_classes = len(np.unique(test_labels))
    test_bin = label_binarize(test_labels, classes=np.arange(n_classes))
    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_bin[:, i], test_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'   
    ]
  
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率', fontsize=12)
    plt.ylabel('真正率', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()
        