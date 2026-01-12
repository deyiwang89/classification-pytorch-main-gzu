from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from classification import Classification

# Predefine colors for each class
CLASS_COLOR_MAP = {
    "Airplane": (255, 0, 0),            # Red
    "Airport": (255, 127, 0),          # Orange
    "Artificial_dense_forest_land": (0, 128, 0),  # Dark Green
    "Artificial_sparse_forest_land": (34, 139, 34),  # Forest Green
    "Bare_land": (210, 180, 140),      # Tan
    "Basketball_court": (255, 165, 0),  # Orange
    "Blue_structured_factory_building": (0, 0, 255),  # Blue
    "Building": (128, 128, 128),        # Gray
    "Construction_site": (139, 69, 19),  # Brown
    "Cross_river_bridge": (105, 105, 105),  # Dim Gray
    "Crossroads": (169, 169, 169),      # Dim Gray
    "Dense_tall_building": (100, 100, 100),  # Dark Gray
    "Dock": (47, 79, 79),               # Dark Cyan
    "Fish_pond": (65, 105, 225),        # Royal Blue
    "Footbridge": (139, 134, 130),      # Gray
    "Graff": (255, 192, 203),           # Pink
    "Grassland": (124, 252, 0),         # Lawn Green
    "Low_scattered_building": (169, 169, 169),  # Dim Gray
    "Lrregular_farmland": (240, 230, 140),  # Ivory
    "Medium_density_scattered_building": (192, 192, 192),  # Silver
    "Medium_density_structured_building": (169, 169, 169),  # Dim Gray
    "Natural_dense_forest_land": (0, 100, 0),  # Dark Green
    "Natural_sparse_forest_land": (50, 205, 50),  # Light Green
    "Oiltank": (139, 69, 19),            # Brown
    "Overpass": (105, 105, 105),         # Dim Gray
    "Parking_lot": (169, 169, 169),      # Dim Gray
    "Plasticgreenhouse": (173, 216, 230),  # Light Blue
    "Playground": (255, 192, 203),       # Pink
    "Railway": (105, 105, 105),          # Dim Gray
    "Red_structured_factory_building": (220, 20, 60),  # Crimson
    "Refinery": (139, 69, 19),           # Brown
    "Regular_farmland": (218, 165, 32),  # Goldenrod
    "Scattered_blue_roof_factory_building": (70, 130, 180),  # Steel Blue
    "Scattered_red_roof_factory_building": (178, 34, 34),  # Firebrick
    "Sewage_plant-type-one": (139, 125, 107),  # Tan
    "Sewage_plant-type-two": (139, 115, 85),   # Tan
    "Ship": (47, 79, 79),                  # Dark Cyan
    "Solar_power_station": (255, 215, 0),  # Gold
    "Sparse_residential_area": (192, 192, 192),  # Silver
    "Square": (169, 169, 169),             # Dim Gray
    "Steelsmelter": (105, 105, 105),       # Dim Gray
    "Storage_land": (139, 125, 107),       # Tan
    "Tennis_court": (255, 206, 135),       # Peach Puff
    "Thermal_power_plant": (255, 99, 71),  # Tomato
    "Vegetable_plot": (154, 205, 50),      # Yellow Green
    "Waste_landfill": (139, 115, 85),      # Tan
    "Water": (0, 191, 255),                # Deep Sky Blue
}


# Create classifier instance
classfication = Classification()

# Image folder path
img_folder = ['cropped_images']
save_dir = 'classified_images'  # Specify save folder

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Create corresponding output subfolders for each input folder
for i_name in img_folder:
    sub_dir = os.path.join(save_dir, i_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    # Traverse all images in the folder
    print(f"Processing folder: {i_name}")
    img_files = [f for f in os.listdir(i_name) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for idx, img_filename in enumerate(img_files):
        img_path = os.path.join(i_name, img_filename)
        
        try:
            # Open image
            image = Image.open(img_path)
            
            # Convert to RGB mode to avoid palette restrictions
            if image.mode != 'RGB':
                original_image = image.convert('RGB')
            else:
                original_image = image.copy()
            
        except Exception as e:
            print(f'Cannot open image {img_filename}: {e}')
            continue
        
        # Detect class name and probability - Modify here to get multiple classes
        try:
            # Assuming Classification class supports returning multiple classes
            # Format is [(class_name1, probability1), (class_name2, probability2), ...]
            results = classfication.detect_multiple_classes(image)
            # Sort by confidence
            results.sort(key=lambda x: x[1], reverse=True)
        except AttributeError:
            # If multiple class detection is not supported, use original method and simulate multiple class results
            class_name, probability = classfication.detect_image(image)
            results = [(class_name, probability)]
        
        # Use the class with the highest confidence as the border color
        if not results:
            print(f"Warning: No class detected for image {img_filename}")
            continue
            
        main_class, _ = results[0]
        
        # Check if the main class is in the map
        if main_class not in CLASS_COLOR_MAP:
            print(f"Warning: Class '{main_class}' not in predefined color map, using default color")
            border_color = (255, 255, 255)  # Default White
        else:
            border_color = CLASS_COLOR_MAP[main_class]
        
        # Draw border and label on the original image
        draw = ImageDraw.Draw(original_image)
        
        # Set border width and font
        width, height = original_image.size
        border_width = max(3, int(min(width, height) * 0.01))  # Border width, minimum 3 pixels
        
        # Draw border (using the color of the main class)
        draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=border_width)
        
        # Try loading font, if failed use default font
        try:
            font_size = max(12, int(min(width, height) * 0.03))  # Font size
            font = ImageFont.truetype("simhei.ttf", font_size)  # Try loading Chinese font
        except IOError:
            try:
                # Try other possible Chinese fonts
                font = ImageFont.truetype("WenQuanYi Micro Hei", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Heiti TC", font_size)
                except IOError:
                    font = None  # Use default font
        # Vertically stack multiple class labels
        y_offset = 0
        padding = 5
        label_spacing = 2  # Vertical spacing between labels

        # Store dimension information of all labels
        label_dimensions = []
        max_text_width = 0

        # Calculate dimensions of all labels first, solve textsize obsolete issue
        for class_name, probability in results:
            label_text = f"{class_name} ({probability:.2f})"
            # Use textbbox instead of textsize, get text bounding box
            # textbbox returns (left, top, right, bottom)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]  # Calculate width
            text_height = bbox[3] - bbox[1]  # Calculate height
            
            label_dimensions.append((text_width, text_height))
            max_text_width = max(max_text_width, text_width)  # Record maximum width

        # Calculate overall background box size
        total_height = sum([h for _, h in label_dimensions]) + \
                    padding * 2 + \
                    label_spacing * (len(results) - 1)

        # Draw each class label one by one (separate lines)
        for i, (class_name, probability) in enumerate(results):
            text_width, text_height = label_dimensions[i]
            
            # Get color of current class
            if class_name not in CLASS_COLOR_MAP:
                color = (255, 255, 255)  # Default White
            else:
                color = CLASS_COLOR_MAP[class_name]
            
            # Draw background box for current label (independent background for each label)
            draw.rectangle(
                [(padding, padding + y_offset), 
                (padding + text_width + padding, padding + y_offset + text_height + padding)],
                fill=color
            )
            
            # Draw label text
            draw.text(
                (padding + 2, padding + 2 + y_offset), 
                label_text, 
                fill=(255, 255, 255),  # White text
                font=font
            )
            
            # Update vertical offset (current label height + padding + label spacing)
            y_offset += text_height + padding + label_spacing

        # Save processed image
        save_path = os.path.join(sub_dir, img_filename)
        original_image.save(save_path)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(img_files)} images")

print(f"All images processed, results saved in: {save_dir}")