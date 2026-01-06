from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from classification import Classification

# 为每个类别预定义颜色
CLASS_COLOR_MAP = {
    "Airplane": (255, 0, 0),            # 红色
    "Airport": (255, 127, 0),          # 橙色
    "Artificial_dense_forest_land": (0, 128, 0),  # 深绿色
    "Artificial_sparse_forest_land": (34, 139, 34),  # 森林绿
    "Bare_land": (210, 180, 140),      # 棕褐色
    "Basketball_court": (255, 165, 0),  # 橙色
    "Blue_structured_factory_building": (0, 0, 255),  # 蓝色
    "Building": (128, 128, 128),        # 灰色
    "Construction_site": (139, 69, 19),  # 棕色
    "Cross_river_bridge": (105, 105, 105),  # 暗灰色
    "Crossroads": (169, 169, 169),      # 暗灰色
    "Dense_tall_building": (100, 100, 100),  # 深灰色
    "Dock": (47, 79, 79),               # 暗青色
    "Fish_pond": (65, 105, 225),        # 皇家蓝
    "Footbridge": (139, 134, 130),      # 灰色
    "Graff": (255, 192, 203),           # 粉红色
    "Grassland": (124, 252, 0),         # 草坪绿
    "Low_scattered_building": (169, 169, 169),  # 暗灰色
    "Lrregular_farmland": (240, 230, 140),  # 象牙色
    "Medium_density_scattered_building": (192, 192, 192),  # 银灰色
    "Medium_density_structured_building": (169, 169, 169),  # 暗灰色
    "Natural_dense_forest_land": (0, 100, 0),  # 暗绿色
    "Natural_sparse_forest_land": (50, 205, 50),  # 浅绿色
    "Oiltank": (139, 69, 19),            # 棕色
    "Overpass": (105, 105, 105),         # 暗灰色
    "Parking_lot": (169, 169, 169),      # 暗灰色
    "Plasticgreenhouse": (173, 216, 230),  # 淡蓝色
    "Playground": (255, 192, 203),       # 粉红色
    "Railway": (105, 105, 105),          # 暗灰色
    "Red_structured_factory_building": (220, 20, 60),  # 深红色
    "Refinery": (139, 69, 19),           # 棕色
    "Regular_farmland": (218, 165, 32),  # 金色
    "Scattered_blue_roof_factory_building": (70, 130, 180),  # 钢蓝色
    "Scattered_red_roof_factory_building": (178, 34, 34),  # 火砖红
    "Sewage_plant-type-one": (139, 125, 107),  # 棕褐色
    "Sewage_plant-type-two": (139, 115, 85),   # 棕褐色
    "Ship": (47, 79, 79),                  # 暗青色
    "Solar_power_station": (255, 215, 0),  # 金色
    "Sparse_residential_area": (192, 192, 192),  # 银灰色
    "Square": (169, 169, 169),             # 暗灰色
    "Steelsmelter": (105, 105, 105),       # 暗灰色
    "Storage_land": (139, 125, 107),       # 棕褐色
    "Tennis_court": (255, 206, 135),       # 桃色
    "Thermal_power_plant": (255, 99, 71),  # 番茄红
    "Vegetable_plot": (154, 205, 50),      # 黄绿色
    "Waste_landfill": (139, 115, 85),      # 棕褐色
    "Water": (0, 191, 255),                # 天蓝色
}


# 创建分类器实例
classfication = Classification()

# 图片文件夹路径
img_folder = ['cropped_images']
save_dir = 'classified_images'  # 指定保存文件夹

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 为每个输入文件夹创建对应的输出子文件夹
for i_name in img_folder:
    sub_dir = os.path.join(save_dir, i_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    # 遍历文件夹中的所有图片
    print(f"正在处理文件夹: {i_name}")
    img_files = [f for f in os.listdir(i_name) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for idx, img_filename in enumerate(img_files):
        img_path = os.path.join(i_name, img_filename)
        
        try:
            # 打开图片
            image = Image.open(img_path)
            
            # 转换为RGB模式以避免调色板限制
            if image.mode != 'RGB':
                original_image = image.convert('RGB')
            else:
                original_image = image.copy()
            
        except Exception as e:
            print(f'无法打开图片 {img_filename}: {e}')
            continue
        
        # 检测类名和概率 - 修改此处以获取多个类别
        try:
            # 假设Classification类支持返回多个类别
            # 格式为[(class_name1, probability1), (class_name2, probability2), ...]
            results = classfication.detect_multiple_classes(image)
            # 按置信度排序
            results.sort(key=lambda x: x[1], reverse=True)
        except AttributeError:
            # 如果不支持多类别检测，使用原方法并模拟多类别结果
            class_name, probability = classfication.detect_image(image)
            results = [(class_name, probability)]
        
        # 使用最高置信度的类别作为边框颜色
        if not results:
            print(f"警告: 图片 {img_filename} 没有检测到任何类别")
            continue
            
        main_class, _ = results[0]
        
        # 检查主要类别是否在映射中
        if main_class not in CLASS_COLOR_MAP:
            print(f"警告: 类别 '{main_class}' 不在预定义颜色映射中，使用默认颜色")
            border_color = (255, 255, 255)  # 默认白色
        else:
            border_color = CLASS_COLOR_MAP[main_class]
        
        # 在原始图像上绘制边框和标签
        draw = ImageDraw.Draw(original_image)
        
        # 设置边框宽度和字体
        width, height = original_image.size
        border_width = max(3, int(min(width, height) * 0.01))  # 边框宽度，最小3像素
        
        # 绘制边框（使用主要类别的颜色）
        draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=border_width)
        
        # 尝试加载字体，若失败则使用默认字体
        try:
            font_size = max(12, int(min(width, height) * 0.03))  # 字体大小
            font = ImageFont.truetype("simhei.ttf", font_size)  # 尝试加载中文字体
        except IOError:
            try:
                # 尝试其他可能的中文字体
                font = ImageFont.truetype("WenQuanYi Micro Hei", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Heiti TC", font_size)
                except IOError:
                    font = None  # 使用默认字体
        
        # # 垂直堆叠显示多个类别标签
        # y_offset = 0
        # for class_name, probability in results:
        #     # 获取当前类别的颜色
        #     if class_name not in CLASS_COLOR_MAP:
        #         color = (255, 255, 255)  # 默认白色
        #     else:
        #         color = CLASS_COLOR_MAP[class_name]
            
        #     # 绘制标签背景和文本
        #     label_text = f"{class_name} ({probability:.2f})"
        #     text_width, text_height = draw.textsize(label_text, font=font)
        #     padding = 5
            
        #     # 标签背景框（垂直堆叠）
        #     draw.rectangle(
        #         [(padding, padding + y_offset), (text_width + 2*padding, text_height + 2*padding + y_offset)],
        #         fill=color
        #     )
            
        #     # 标签文本
        #     draw.text((padding + 2, padding + 2 + y_offset), label_text, fill=(255, 255, 255), font=font)
            
        #     # 更新垂直偏移量
        #     y_offset += text_height + 2 * padding + 2  # 每个标签之间留出一些空间

        # 垂直堆叠显示多个类别标签
        y_offset = 0
        padding = 5
        label_spacing = 2  # 标签之间的垂直间距

        # 存储所有标签的尺寸信息
        label_dimensions = []
        max_text_width = 0

        # 先计算所有标签的尺寸，解决textsize过时问题
        for class_name, probability in results:
            label_text = f"{class_name} ({probability:.2f})"
            # 使用textbbox替代textsize，获取文本边界框
            # textbbox返回(left, top, right, bottom)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]  # 计算宽度
            text_height = bbox[3] - bbox[1]  # 计算高度
            
            label_dimensions.append((text_width, text_height))
            max_text_width = max(max_text_width, text_width)  # 记录最大宽度

        # 计算整体背景框尺寸
        total_height = sum([h for _, h in label_dimensions]) + \
                    padding * 2 + \
                    label_spacing * (len(results) - 1)

        # 绘制统一背景框（可选，根据需要保留或删除）
        # draw.rectangle(
        #     [(padding, padding), 
        #      (max_text_width + 2 * padding, padding + total_height)],
        #     fill=(0, 0, 0, 128)  # 半透明黑色背景
        # )

        # 逐个绘制每个类别标签（单独成行）
        for i, (class_name, probability) in enumerate(results):
            text_width, text_height = label_dimensions[i]
            
            # 获取当前类别的颜色
            if class_name not in CLASS_COLOR_MAP:
                color = (255, 255, 255)  # 默认白色
            else:
                color = CLASS_COLOR_MAP[class_name]
            
            # 绘制当前标签的背景框（每个标签独立背景）
            draw.rectangle(
                [(padding, padding + y_offset), 
                (padding + text_width + padding, padding + y_offset + text_height + padding)],
                fill=color
            )
            
            # 绘制标签文本
            draw.text(
                (padding + 2, padding + 2 + y_offset), 
                label_text, 
                fill=(255, 255, 255),  # 白色文本
                font=font
            )
            
            # 更新垂直偏移量（当前标签高度 + 内边距 + 标签间距）
            y_offset += text_height + padding + label_spacing

        # 保存处理后的图像
        save_path = os.path.join(sub_dir, img_filename)
        original_image.save(save_path)
        
        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(img_files)} 张图片")

print(f"所有图片处理完成，结果保存在: {save_dir}")