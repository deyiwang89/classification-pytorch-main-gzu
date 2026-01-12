#程序中链接了多个 OpenMP 运行时库的副本
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #----------------------------------------------------#
    #   Whether to use Cuda
    #   Set to False if no GPU is available
    #----------------------------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-card distributed operation
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify graphics cards under Ubuntu.
    #                   Windows system uses DP mode to call all graphics cards by default, DDP is not supported.
    #   DP mode:
    #       Settings            distributed = False
    #       Input in terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Settings            distributed = True
    #       Input in terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available for multi-card in DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce video memory by about half, requires pytorch 1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #----------------------------------------------------#
    #   When training your own dataset, make sure to modify classes_path
    #   Modify to the txt file corresponding to your own classes
    #----------------------------------------------------#
    classes_path    = 'model_data/cls_classes.txt' 
    #----------------------------------------------------#
    #   Input image size
    #----------------------------------------------------#
    input_shape     = [256, 256]
    #------------------------------------------------------#
    #   Model type used:
    #   mobilenetv2 , 
    #   resnet18 , resnet34 , resnet50 , resnet101 , resnet152
    #   vgg11 , vgg13 , vgg16 , vgg11_bn , vgg13_bn , vgg16_bn , 
    #   vit_b_16 , 
    #   swin_transformer_tiny , swin_transformer_small , swin_transformer_base
    #------------------------------------------------------#
    backbone        = "resnet34"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Whether to use the pre-trained weights of the backbone network. Here, the backbone weights are used, so they are loaded during model construction.
    #   If model_path is set, the backbone weights do not need to be loaded, and the value of pretrained is meaningless.
    #   If model_path is not set, pretrained = True, then only the backbone is loaded to start training.
    #   If model_path is not set, pretrained = False, Freeze_Train = False, then training starts from 0, and there is no process of freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Please refer to README for downloading weight files, which can be downloaded via Baidu Netdisk. The pre-trained weights of the model are universal for different datasets because features are universal.
    #   The more important part of the pre-trained weights of the model is the weight part of the backbone feature extraction network, which is used for feature extraction.
    #   Pre-trained weights must be used in 99% of cases. If not used, the weights of the backbone part are too random, the feature extraction effect is not obvious, and the network training results will not be good.
    #
    #   If there is an operation to interrupt training during the training process, you can set model_path to the weight file in the logs folder to reload the weights that have been partially trained.
    #   At the same time, modify the parameters of the Freeze Phase or Unfreeze Phase below to ensure the continuity of the model epoch.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   Here, the weights of the entire model are used, so they are loaded in train.py, and pretrain does not affect the weight loading here.
    #   If you want the model to start training from the pre-trained weights of the backbone, set model_path = '', pretrain = True, and only the backbone is loaded at this time.
    #   If you want the model to start training from 0, set model_path = '', pretrain = False, and start training from 0 at this time.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ""
        
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two phases: the freezing phase and the unfreezing phase. The freezing phase is set to meet the training needs of students with insufficient machine performance.
    #   Freezing training requires less video memory. In the case of very poor graphics cards, Freeze_Epoch can be set equal to UnFreeze_Epoch, and only freezing training is performed at this time.
    #      
    #   Here are some suggestions for parameter settings. Trainers can adjust them flexibly according to their own needs:
    #   (1) Start training from the pre-trained weights of the entire model: 
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3. (Freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3. (No Freeze)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 200, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2. (Freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 200, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2. (No Freeze)
    #       Where: UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Start training from 0:
    #       Adam:
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Unfreeze_batch_size >= 16, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3. (No Freeze)
    #       SGD:
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Unfreeze_batch_size >= 16, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2. (No Freeze)
    #       Where: UnFreeze_Epoch should generally not be less than 300.
    #   (3) Setting of batch_size:
    #       Within the range acceptable by the graphics card, larger is better. Insufficient video memory has nothing to do with the dataset size. If prompted with insufficient video memory (OOM or CUDA out of memory), please reduce batch_size.
    #       Affected by the BatchNorm layer, batch_size must be at least 2 and cannot be 1.
    #       Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times Unfreeze_batch_size. It is not recommended to set the gap too large, because it relates to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freezing phase training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Occupies less video memory, only fine-tunes the network
    #   Init_Epoch          The current starting training epoch of the model, its value can be greater than Freeze_Epoch, e.g., setting:
    #                       Init_Epoch = 60 , Freeze_Epoch = 50 , UnFreeze_Epoch = 100
    #                       Will skip the freezing phase, start directly from epoch 60, and adjust the corresponding learning rate.
    #                       (Used when resuming training from a breakpoint)
    #   Freeze_Epoch        Freeze_Epoch for model freezing training
    #                       (Invalid when Freeze_Train=False)
    #   Freeze_batch_size   batch_size for model freezing training
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   Unfreezing phase training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   Occupies more video memory, all parameters of the network will change
    #   UnFreeze_Epoch          Total training epochs of the model
    #   Unfreeze_batch_size     batch_size of the model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 1
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freezing training
    #                   Default is to freeze the backbone training first and then unfreeze training.
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate of the model
    #                   When using Adam optimizer, it is recommended to set Init_lr=1e-3
    #                   When using SGD optimizer, it is recommended to set Init_lr=1e-2
    #   Min_lr          Minimum learning rate of the model, default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Type of optimizer used, options: adam, sgd
    #                   When using Adam optimizer, it is recommended to set Init_lr=1e-3
    #                   When using SGD optimizer, it is recommended to set Init_lr=1e-2
    #   momentum        momentum parameter used inside the optimizer
    #   weight_decay    Weight decay, can prevent overfitting
    #                   There will be errors when using adam optimizer, recommended to set to 0
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay method used, options: step, cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     How many epochs to save weights once
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        Folder to save weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threading to read data
    #                   Enabling it will speed up data reading, but will occupy more memory
    #                   Computers with small memory can be set to 2 or 0
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   Training image path and labels
    #   test_annotation_path    Validation image path and labels (using test set instead of validation set)
    #------------------------------------------------------#
    train_annotation_path   = "cls_train.txt"
    test_annotation_path    = 'cls_test.txt'


    #------------------------------------------------------#
    #   Set the graphics card used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   Download pre-trained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    #------------------------------------------------------#
    #   Get classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        #------------------------------------------------------#
        #   Please check README for weight files, download from Baidu Netdisk
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load based on the Key of pre-trained weights and the Key of the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show Keys that were not matched
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mTips: It is normal if the head part is not loaded, but it is wrong if the Backbone part is not loaded.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to use fp16 correctly
    #   Therefore, torch 1.2 displays "could not be resolve" here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-card sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-card parallel running
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
        
    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #---------------------------------------------------------#
    #   Total training epochs refers to the total number of times traversing all data
    #   Total training steps refers to the total number of gradient descents
    #   Each training epoch contains several training steps, and each training step performs one gradient descent.
    #   Here only the minimum training epoch is suggested, there is no upper limit, and only the unfreezing part is considered in the calculation
    #----------------------------------------------------------#
    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using %s optimizer, it is recommended to set the total training steps to above %d.\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] The total training data volume for this run is %d, Unfreeze_batch_size is %d, training %d Epochs in total, and the calculated total training steps are %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training steps are %d, which is less than the suggested total steps %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   Backbone feature extraction network features are universal, freezing training can speed up training
    #   It can also prevent weights from being destroyed in the early stages of training.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch for freezing training
    #   UnFreeze_Epoch is the total training epoch
    #   Please reduce Batch_size if OOM or insufficient video memory is prompted
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze a certain part for training
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        #-------------------------------------------------------------------#
        #   If not freezing training, set batch_size directly to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adaptively adjust the learning rate
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            nbs             = 256
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        optimizer = {
            'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        #---------------------------------------#
        #   Get the formula for learning rate decay
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        train_dataset   = DataGenerator(train_lines, input_shape, True)
        val_dataset     = DataGenerator(val_lines, input_shape, False)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
            
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen learning part
            #   Then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adaptively adjust the learning rate
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                    nbs             = 256
                    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get the formula for learning rate decay
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                model.Unfreeze_backbone()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
