import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------#
#   test_annotation_path    测试图片路径和标签
#------------------------------------------------------#
test_annotation_path    = 'cls_test.txt'
#------------------------------------------------------#
#   metrics_out_path        指标保存的文件夹
#------------------------------------------------------#
metrics_out_path        = "metrics_out"

class Eval_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    with open("./cls_test.txt","r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
    
    # 打开文件以写入模式
    with open('metrics_out\\result.txt', 'w') as file:
        # 将信息写入文件
        file.write("top-1 accuracy = %.2f%%\n" % (top1*100))
        file.write("top-5 accuracy = %.2f%%\n" % (top5*100))
        file.write("mean Recall = %.2f%%\n" % (np.mean(Recall)*100))
        file.write("mean Precision = %.2f%%\n" % (np.mean(Precision)*100))



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