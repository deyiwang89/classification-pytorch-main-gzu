## Classification: Implementation of Classification Models in PyTorch
---

## Table of Contents
1. [Environment](#Environment)
2. [Download](#Download)
3. [How2train](#How2train)
4. [How2predict](#How2predict)
5. [How2eval](#How2eval)
6. [Reference](#Reference)



## Environment
pytorch == 1.2.0

## Download
Pre-trained weights required for training can be downloaded from Baidu Cloud.
Link: https://pan.baidu.com/s/18Ze7YMvM5GpbTlekYO8bcA
Extraction Code: 5wym

The dataset used for training can be obtained by contacting Yinting Lv. Email: 2112332016@e.gzhu.edu.cn


## How2train
1. The images stored in the `datasets` folder are divided into two parts: `train` contains training images, and `test` contains test images.
2. Before training, you need to prepare the dataset. Create different folders inside the `train` or `test` directories, where each folder name corresponds to the class name, and the images inside are for that class. The file structure is as follows:
```
|-datasets
    |-train
        |-class1
            |-123.jpg
            |-234.jpg
        |-class2
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |- class1
            |-567.jpg
            |-678.jpg
        |- class2
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, run `txt_annotation.py` in the root directory to generate `cls_train.txt` required for training. Before running, modify the `classes` in the script to match your specific classes.
4. Then modify `cls_classes.txt` in the `model_data` folder to also correspond to your specific classes.
5. After adjusting the network and weights you want to choose in `train.py`, you can start training!

## How2predict
### a. Using Pre-trained Weights
1. After downloading and unzipping the repository, there is already a trained model in `model_data`. Run `predict.py` and input:
```python
img/123.jpg
```
### b. Using Your Own Trained Weights
1. Train the model following the "How2train" steps.
2. In the `classification.py` file, modify `model_path`, `classes_path`, `backbone`, and `alpha` in the following section to match your trained files. **`model_path` corresponds to the weight file in the `logs` folder, `classes_path` is the class file corresponding to `model_path`, `backbone` corresponds to the backbone feature extraction network used, and `alpha` is the alpha value when using MobileNet.**
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   To predict using your own trained model, you must modify model_path and classes_path!
    #   model_path points to the weight file in the logs folder, classes_path points to the txt in model_data
    #   If shape mismatch occurs, pay attention to modifying model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'logs/best.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Model type used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used to demonstrate how to use mini_imagenet to train your own pre-trained weights
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if no GPU is available
    #-------------------------------#
    "cuda"          : True
}
```
3. Run `predict.py` and input:
```python
img/cat.jpg
```

## How2eval
1. The images stored in the `datasets` folder are divided into two parts: `train` contains training images, and `test` contains test images. During evaluation, we use the images in the `test` folder.
2. Before evaluation, you need to prepare the dataset. Create different folders inside the `train` or `test` directories, where each folder name corresponds to the class name, and the images inside are for that class. The file structure is as follows:
```
|-datasets
    |-train
        |-class1
            |-123.jpg
            |-234.jpg
        |-class2
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |- class1
            |-567.jpg
            |-678.jpg
        |- class2
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, run `txt_annotation.py` in the root directory to generate `cls_test.txt` required for evaluation. Before running, modify the `classes` in the script to match your specific classes.
4. Then modify `model_path`, `classes_path`, `backbone`, and `alpha` in the `classification.py` file to match your trained files. **`model_path` corresponds to the weight file in the `logs` folder, `classes_path` is the class file corresponding to `model_path`, `backbone` corresponds to the backbone feature extraction network used, and `alpha` is the alpha value when using MobileNet.**
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   To predict using your own trained model, you must modify model_path and classes_path!
    #   model_path points to the weight file in the logs folder, classes_path points to the txt in model_data
    #   If shape mismatch occurs, pay attention to modifying model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'logs/best.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Model type used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used to demonstrate how to use mini_imagenet to train your own pre-trained weights
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if no GPU is available
    #-------------------------------#
    "cuda"          : True
}
```
5. Run the corresponding scripts for evaluation or prediction:
   - **010-eval.py**: Model Evaluation. Calculates Top-1 Accuracy, Top-5 Accuracy, Recall, Precision, and generates a Confusion Matrix. Results are saved in the `metrics_out` folder.
   - **011-predict.py**: Visual Prediction (Grad-CAM). Predicts single or multiple images and generates heatmaps (Grad-CAM) to show areas the model focuses on. Results are saved in the `88888` folder.
   - **012-new_predict.py**: Batch Prediction Application. Classifies images in batch, draws bounding boxes and labels (class, probability) on the original images. Results are saved in the `classified_images` folder.

## Reference
https://github.com/keras-team/keras-applications
https://github.com/bubbliiiing/classification-pytorch
