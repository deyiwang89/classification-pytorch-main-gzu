import os
import random
import shutil

# 母文件夹路径
mother_folder = "datasets/train" 
# 指定文件夹路径
target_folder = "temp_present" 

if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# 获取母文件夹下的所有子文件夹
sub_folders = [f for f in os.listdir(mother_folder) if os.path.isdir(os.path.join(mother_folder, f))]

# 从子文件夹中随机选择一张图片并拷贝到目标文件夹
for sub_folder in sub_folders:
    # 子文件夹路径
    sub_folder_path = os.path.join(mother_folder, sub_folder)
    # 获取子文件夹下的所有图片文件（假设图片扩展名为jpg和png）
    image_files = [f for f in os.listdir(sub_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    if image_files:  # 如果子文件夹有图像文件，选择随机一个并拷贝到目标文件夹
        image_file = random.choice(image_files)  # 随机选择一个图片文件
        source_path = os.path.join(sub_folder_path, image_file)  # 图片源路径
        target_path = os.path.join(target_folder, f"{sub_folder}.jpg")  # 目标路径，以子文件夹名称命名，且格式为jpg（不论原文件格式）
        shutil.copy2(source_path, target_path)  # 拷贝文件到目标路径