from collections import namedtuple
import math
import numpy as np
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
from torch.autograd import Variable

from scipy.spatial import distance
from pprint import pprint
import pdb
# import os
# import sys
# here = os.getcwd()
# # my_root = os.path.abspath(os.path.join(here, "utils"))
# my_root = os.path.abspath(os.path.join(here, "utils"))
# sys.path.insert(0, my_root)
from config_ import cfg
from .prune import *

np.set_printoptions(precision=2)





class RConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.misc = (kernel_size,
                 stride, padding, dilation, groups, bias)
        # self.norm_prune = 1.0
        # self.simi_prune = cfg.SIMI
        # self.register_buffer("filter_mask", torch.zeros([out_channels, 1]))
        # self.prune_num = int(out_channels*self.simi_prune)
        # self.prune_num = None
        # self.fuse_factor = cfg.FUSE
        self.simi_type = cfg.SIMI_TYPE
        # self.simi_ch = cfg.SIMI_CH
        # self.masked = None

        self._to_regu = False
        self.cos_loss = None
        self.weight_dummy = None

        print("init my MConv Module")


    def forward(self, input):
        if self._to_regu:
            self.do_regu()
            # self._to_regu = False

        m_weight = self.weight
        output = F.conv2d(input, m_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return output
    

    def cut_channel(self, dim="out", ratio=0.3):
        ratio = cfg.CUT
        # if dim == 'out':
        #     shrinked = int(self.out_channels*ratio)
        #     self.__init__(shrinked, self.in_channels, *self.misc)
        # elif dim == 'in':
        #     shrinked = int(self.in_channels*ratio)
        #     self.__init__(self.out_channels, shrinked, *self.misc)
        # shrinked = int(self.in_channels*ratio)
        self.__init__(int(self.out_channels*ratio), int(self.in_channels*ratio), *self.misc)
    
    def do_regu(self):
        dim = cfg.SIMI_CH
        # weight = self.weight.detach()
        weight = self.weight.detach()
        self.weight_dummy = weight
        # print(weight.requires_grad)
        self.weight_dummy.requires_grad_(True)
        # print(weight.requires_grad)
        if dim == "out":
            weight_mat = weight.view(weight.shape[0], -1)
        elif dim == 'ikk':
            weight_mat = weight.view(weight.shape[0], -1).t()
        simi = simi_torch(weight_mat)
        mean_simi = torch.abs(simi).mean()
        self.cos_loss = mean_simi

    def to_regu(self):
        self._to_regu = True
    

    def check_grad(self):
        pass
        # grad =  self.weight.grad[:3, :3, 0, 0]
        # print(grad)
    
    def regu_grad(self):
        self.cos_loss.backward(retain_graph=True)
        grad_dummy = self.weight_dummy.grad.data
        print("grad_dummy.device: ", grad_dummy.device)
        self.weight.grad.data += grad_dummy*cfg.SIMI

        if self.weight.shape[1] == 3:
            print(grad_dummy[:3,:3, 0, 0])

        # print(cfg.SIMI, self)

