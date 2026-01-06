'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，可以参考一下txt_annotation.py文件。
'''
from PIL import Image
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from classification import Classification

classfication = Classification()
count = 0
# while count <1:
#     img = 'datasets\\train\\Airplane\\2016-07-31_182552.png'#input('Input image filename:')
#     count += 1
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         class_name, probability = classfication.detect_image(image)
#         print(class_name)
#         bbb = classfication.get_grad_cam(image)
#         plt.title('Class:%s Probability:%.3f' %(class_name, probability))


# 图片文件夹路径
img_folder = ['temp_present']

save_dir = '88888'  # 指定保存文件夹

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

for i_name in img_folder:
    # 遍历文件夹中的所有图片
    sub_dir = os.path.join(save_dir, i_name)  # 创建子文件夹路径
    os.makedirs(sub_dir, exist_ok=True)  # 创建子文件夹，如果已存在则不报错
    for img_filename in os.listdir(i_name):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            img_path = os.path.join(i_name, img_filename)
            try:
                # 打开图片
                image = Image.open(img_path)
            except Exception as e:
                print('Error opening image! Try again:', e)
                continue
            else:
                # 检测类名和概率
                class_name, probability = classfication.detect_image(image)
                
                # 获取Grad-CAM图像
                bbb = classfication.get_grad_cam(image)
            
                # 绘制图像
                plt.imshow(bbb)
                plt.title(f'Class: {class_name}; Probability: {probability:.3f}')
                plt.axis('off')  # 关闭坐标轴

                # 保存图像到对应的子文件夹中
                save_path = os.path.join(sub_dir, img_filename)  # 修改保存路径为子文件夹下的路径
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()  # 关闭当前图表以释放内存

# for i_name in img_folder:
#     # 遍历文件夹中的所有图片
#     for img_filename in os.listdir(i_name):
#         if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
#             img_path = os.path.join(i_name, img_filename)
#         try:
#             # 打开图片
#             image = Image.open(img_path)
#         except Exception as e:
#             print('Error opening image! Try again:', e)
#             continue
#         else:
#             # 检测类名和概率
#             class_name, probability = classfication.detect_image(image)
#             # print('Class Name:', class_name)
#             # print('Probability:', probability)

#             # 获取Grad-CAM图像
#             bbb = classfication.get_grad_cam(image)
        
#             # 绘制图像
#             plt.imshow(bbb)
#             plt.title(f'Class: {class_name}; Probability: {probability:.3f}')
#             plt.axis('off')  # 关闭坐标轴

#             # 保存图像
#             save_path = os.path.join(save_dir, os.path.basename(img_path))
#             plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
#             plt.close()  # 关闭当前图表以释放内存
