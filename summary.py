#--------------------------------------------#
#   This part of the code is only used to view the network structure, not for testing
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets import get_model_from_name

if __name__ == "__main__":
    input_shape = [224, 224]
    num_classes = 1000
    backbone    = "mobilenetv2"
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = get_model_from_name[backbone](num_classes=num_classes, pretrained=False).to(device)
    
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2 because profile does not count convolution as two operations
    #   Some papers count convolution as multiplication and addition, two operations. In this case, multiply by 2
    #   Some papers only consider multiplication operations and ignore addition. In this case, do not multiply by 2
    #   This code chooses to multiply by 2, referring to YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
