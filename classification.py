import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)

def apply_gaussian_blur(heatmap, kernel_size=(13, 13), sigma=0):
    """
    对热力图应用高斯滤波进行平滑。
    :param heatmap: 原始热力图。
    :param kernel_size: 高斯核的大小，通常为奇数，如(5, 5)。
    :param sigma: 高斯核的标准差。
    :return: 平滑后的热力图。
    """
    for i in range(5):
        heatmap = cv2.GaussianBlur(heatmap, kernel_size, sigma)
    return heatmap
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
#--------------------------------------------#
class Classification(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs\\GSFF-VIT\\best_epoch_weights.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        #--------------------------------------------------------------------#
        #   输入的图片大小
        #--------------------------------------------------------------------#
        "input_shape"       : [224, 224],
        #--------------------------------------------------------------------#
        #   所用模型种类：
        #   mobilenetv2 , 
        #   resnet18  , resnet34 , resnet50 , resnet101 , resnet152
        #   vgg11 , vgg13 , vgg16 , vgg11_bn , vgg13_bn ,  vgg16_bn  , 
        #   vit_b_16  , 
        #   swin_transformer_tiny , swin_transformer_small ,  swin_transformer_base
        #--------------------------------------------------------------------#
        "backbone"          : 'vit_b_16',
        #--------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   获得种类
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            state_dict = torch.load(self.model_path, map_location=self.device, strict=False)
            # 只保留能够与model中存在的层名对应的部分进行匹配加载
            updated_state_dict = {k: state_dict[k] for k in self.model.state_dict() if k in state_dict}
            # 现在仅包含那些能在两个StateDict中都找到的key（对应模型中已有的层），即"真正"能匹配的层，来加载这些权重
            self.model.load_state_dict(updated_state_dict)
        except:
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
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
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   获得所属种类
        #---------------------------------------------------#
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        #---------------------------------------------------#
        #   绘图并写字
        #---------------------------------------------------#
        # plt.subplot(1, 1, 1)
        # plt.imshow(np.array(image))
        # plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        # plt.show()
        return class_name, probability

    def get_grad_cam(self, image, class_idx=None):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.enable_grad():
            photo = torch.from_numpy(image_data).to(self.device).requires_grad_(True)
            outputs = self.model(photo)
            if class_idx is None:
                class_idx = np.argmax(outputs[0].detach().cpu().numpy())
            outputs = outputs[0, class_idx]

        # 计算梯度
        self.model.zero_grad()
        outputs.backward()
        gradients = photo.grad

        # 获取最后一个卷积层
        modules = list(self.model.modules())
        for i in range(len(modules) - 1, -1, -1):
            module = modules[i]
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                break

        # 提取特征图
        features = photo.detach().clone().requires_grad_(False)

        # 计算梯度的全局平均值
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

        # 加权特征图
        for i in range(features.shape[1]):
            features[:, i, :, :] *= pooled_gradients[:, i, :, :]

        # 应用 ReLU
        features = F.relu(features)

        # 生成 Grad-CAM
        features_avg = torch.mean(features, dim=1).squeeze()
        cam = features_avg.clone()
        cam = cam.clamp(min=0).detach().cpu().numpy()

        # 平滑处理
        # cam = cv2.resize(cam, (image.width // 8, image.height // 8))
        # cam = cv2.GaussianBlur(cam, (5, 5), sigmaX=0)
        cam = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

        # # 归一化
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # 将热力图转为颜色
        cam = np.uint8(255 * cam)
        cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # 合并原图和 Grad-CAM
        heatmap = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        superimposed_img = heatmap.astype(float)*0.8 + np.array(image, dtype=float)*0.6
        superimposed_img = np.clip(superimposed_img / 255, 0, 1)

        return superimposed_img




    # def get_grad_cam(self, image, class_idx=None):
    #     image = cvtColor(image)
    #     image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
    #     image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

    #     with torch.enable_grad():
    #         photo = torch.from_numpy(image_data).to(self.device).requires_grad_(True)
    #         outputs = self.model(photo)
    #         if class_idx is None:
    #             class_idx = np.argmax(outputs[0].detach().cpu().numpy())
    #         outputs = outputs[0, class_idx]

    #     # 计算梯度
    #     self.model.zero_grad()
    #     outputs.backward()
    #     gradients = photo.grad

    #     # 获取最后一个卷积层的梯度和特征图
    #     # 将模块转换为列表
    #     modules = list(self.model.modules())
    #     # print(modules)
    #     # 从模型的末尾向前遍历，找到最后一个卷积层
    #     for i in range(len(modules) - 1, -1, -1):
    #         module = modules[i]
    #         if isinstance(module, nn.Conv2d):
    #             conv_layer = module
    #             break

    #     # 提取特征图
    #     features = photo.detach().clone().requires_grad_(False)

    #     # 计算梯度的全局平均值
    #     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

    #     # 将梯度加权到特征图上
    #     for i in range(features.shape[1]):
    #         features[:, i, :, :] *= pooled_gradients[:, i, :, :]      
        
    #     # 应用ReLU激活函数
    #     features = F.relu(features)

    #     # 生成Grad-CAM
    #     features_avg = torch.mean(features, dim=1).squeeze()  # 对通道维度进行平均，得到平均值
    #     cam = features_avg.clone()  # 创建特征的克隆副本，以便后续修改

    #     # 生成Grad-CAM
    #     cam = torch.mean(features, dim=1).squeeze()  # 对通道维度进行平均
    #     cam = cam.clamp(min=0)  # 只保留正向的贡献
    #     cam = cam.detach().cpu().numpy()

    #     cam = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

    #     # # 归一化CAM
    #     cam = cam - np.min(cam)
    #     cam = cam / np.max(cam)
    #     # 将低于平均值的元素变为0
    #     cam[cam < cam.mean()] = 0  # 找到小于平均值的元素并将其设置为0

    #     # 对每个元素进行转换: x -> 1/x (避免除以零)
    #     cam = np.where(cam > 0, 1 / cam, 0)


    #     # 将热力图转换为OpenCV格式
    #     cam = np.uint8(255 * cam)
    #     cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_HOT)


    #     # 绘制Grad-CAM
    #     heatmap = cam_color#cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #     superimposed_img = heatmap + np.array(image)

    #     return superimposed_img/255