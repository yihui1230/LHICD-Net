import torch.nn.functional as F
import torch_dct as dct
import torch.nn as nn



def dct_2d(rgb_tensor, P=8):
    """
    Modified according to https://github.com/VisibleShadow/Implementation-of-Detecting-Camouflaged-Object-in-Frequency-Domain/blob/main/train.py
    """
    #ycbcr_tensor = rgb2ycbcr(rgb_tensor)
    ycbcr_tensor=rgb_tensor

    c = ycbcr_tensor.shape[1]


    num_batchsize = ycbcr_tensor.shape[0]
    size = ycbcr_tensor.shape[2]
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, c, size // P, P, size // P, P).permute(0, 2, 4, 1, 3, 5)
    ycbcr_tensor = dct.dct_2d(ycbcr_tensor, norm='ortho')
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, size // P, size // P, -1).permute(0, 3, 1, 2)
    return ycbcr_tensor

def slide_dct(rgb_tensor, P=7,stride=4):
    """
    Modified according to https://github.com/VisibleShadow/Implementation-of-Detecting-Camouflaged-Object-in-Frequency-Domain/blob/main/train.py
    """
    m = nn.ReflectionPad2d((P-1)//2)
    #ycbcr_tensor = rgb2ycbcr(rgb_tensor)
    ycbcr_tensor=rgb_tensor
    num_batchsize = ycbcr_tensor.shape[0]
    c=ycbcr_tensor.shape[1]
    size = ycbcr_tensor.shape[2]
    kernel_region=F.unfold(m(ycbcr_tensor),kernel_size=P,stride=stride)
    kernel_region=kernel_region.reshape(num_batchsize,c,P,P,size//stride , size//stride).permute(0,  4,5,1, 2, 3)

    ycbcr_tensor = dct_2d(kernel_region, norm='ortho')
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, size//stride , size//stride, -1).permute(0, 3, 1, 2)
    return ycbcr_tensor