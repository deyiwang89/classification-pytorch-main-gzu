import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = 'metrics_out\\confusion_matrix.csv'
# print(os.path.dirname(path))
# 读取CSV文件
data = pd.read_csv(path, index_col=0)

# 提取类别名称
classes = data.columns.tolist()

# 转换为 NumPy 数组
confusion_matrix = data.values

# 创建热力图
plt.figure(figsize=(24, 20))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)

# 设置标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 旋转标签
plt.xticks(rotation=45, ha='right')  # x轴标签旋转45度
plt.yticks(rotation=0)  # y轴标签不旋转

# 保存图像，分辨率为300 dpi
cm_img_path = os.path.join(os.path.dirname(path),'confusion_matrix.png')
plt.savefig(cm_img_path, dpi=300)




# 计算整体准确率
correct_predictions = np.sum(np.diag(confusion_matrix))  # 对角线元素之和
total_samples = np.sum(confusion_matrix)  # 所有样本的数量
overall_accuracy = correct_predictions / total_samples  # 计算整体准确率
print(correct_predictions ,total_samples)
print(f"整体准确率为 {overall_accuracy:.4f}")  # 输出结果，保留两位小数

# 存储每个类别的准确率和召回率
precision_list = []
recall_list = []

for i in range(len(classes)):
    tp = confusion_matrix[i, i]  # 真正例
    fp = np.sum(confusion_matrix[:, i]) - tp  # 假正例
    fn = np.sum(confusion_matrix[i, :]) - tp  # 假负例
    
    # 计算准确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)

# 计算宏平均、微平均和加权平均
macro_precision = np.mean(precision_list)  # 宏平均
macro_recall = np.mean(recall_list)        # 宏平均

micro_precision = np.sum(precision_list) / len(classes)  # 微平均
micro_recall = np.sum(recall_list) / len(classes)        # 微平均

weighted_precision = np.average(precision_list, weights=np.sum(confusion_matrix, axis=1))  # 加权平均
weighted_recall = np.average(recall_list, weights=np.sum(confusion_matrix, axis=1))        # 加权平均

# 输出平均值
print(f"宏平均准确率：{macro_precision:.4f}，宏平均召回率：{macro_recall:.4f}")
print(f"微平均准确率：{micro_precision:.4f}，微平均召回率：{micro_recall:.4f}")
print(f"加权平均准确率：{weighted_precision:.4f}，加权平均召回率：{weighted_recall:.4f}")
