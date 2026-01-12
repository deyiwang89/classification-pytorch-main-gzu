'''
predict.py Notes
1. Cannot perform batch prediction. If you want batch prediction, you can use os.listdir() to traverse the folder and use Image.open to open image files for prediction.
2. If you want to save the prediction results as txt, you can use open to open the txt file and use the write method to write to txt. You can refer to the txt_annotation.py file.
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


# Image folder path
img_folder = ['temp_present']

save_dir = '88888'  # Specify save folder

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

for i_name in img_folder:
    # Traverse all images in the folder
    sub_dir = os.path.join(save_dir, i_name)  # Create subfolder path
    os.makedirs(sub_dir, exist_ok=True)  # Create subfolder, no error if it already exists
    for img_filename in os.listdir(i_name):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            img_path = os.path.join(i_name, img_filename)
            try:
                # Open image
                image = Image.open(img_path)
            except Exception as e:
                print('Error opening image! Try again:', e)
                continue
            else:
                # Detect class name and probability
                class_name, probability = classfication.detect_image(image)
                
                # Get Grad-CAM image
                bbb = classfication.get_grad_cam(image)
            
                # Draw image
                plt.imshow(bbb)
                plt.title(f'Class: {class_name}; Probability: {probability:.3f}')
                plt.axis('off')  # Turn off axes

                # Save image to corresponding subfolder
                save_path = os.path.join(sub_dir, img_filename)  # Modify save path to path under subfolder
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()  # Close current chart to release memory

# for i_name in img_folder:
#     # Traverse all images in the folder
#     for img_filename in os.listdir(i_name):
#         if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
#             img_path = os.path.join(i_name, img_filename)
#         try:
#             # Open image
#             image = Image.open(img_path)
#         except Exception as e:
#             print('Error opening image! Try again:', e)
#             continue
#         else:
#             # Detect class name and probability
#             class_name, probability = classfication.detect_image(image)
#             # print('Class Name:', class_name)
#             # print('Probability:', probability)

#             # Get Grad-CAM image
#             bbb = classfication.get_grad_cam(image)
        
#             # Draw image
#             plt.imshow(bbb)
#             plt.title(f'Class: {class_name}; Probability: {probability:.3f}')
#             plt.axis('off')  # Turn off axes

#             # Save image
#             save_path = os.path.join(save_dir, os.path.basename(img_path))
#             plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
#             plt.close()  # Close current chart to release memory

