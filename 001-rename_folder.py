import os


with open("model_data\\cls_classes.txt","r") as f:
    names = f.readlines()

for i,val in enumerate(names):
    # print(i,val[:-1])

    # 定义目标目录
    base_directory = 'datasets'

    # 遍历目录
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            if dir_name == str(i):
                # 构造旧文件夹路径
                old_folder_path = os.path.join(root, dir_name)
                # 定义新的文件夹名称
                new_folder_name = val[:-1]  # 可以根据需要动态生成
                # 构造新文件夹路径
                new_folder_path = os.path.join(root, new_folder_name)
                
                # 重命名文件夹
                os.rename(old_folder_path, new_folder_path)
                print(f"文件夹 '{old_folder_path}' 已被重命名为 '{new_folder_path}'")