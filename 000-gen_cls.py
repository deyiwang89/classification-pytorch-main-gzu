import os 
import json

# 从 JSON 文件加载字典
with open('RSD46-WHU.json', 'r') as f:
    rename_dict = json.load(f)

output_file = "cls_classes.txt"

# 创建并打开输出文件
with open(output_file, mode='w', newline='', encoding='utf-8') as f:

    for key,val in rename_dict.items():
        f.write(f"{key}\n")

f.close()
