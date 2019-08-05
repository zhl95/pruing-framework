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





class FConv2d(nn.Conv2d):
    """docstring for FConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.misc = (kernel_size,
                 stride, padding, dilation, groups, bias)
        # self.norm_prune = 1.0
        self.keep = out_channels - int(out_channels*cfg.SIMI)
        # self.register_buffer("filter_mask", torch.zeros([out_channels, 1]))
        # self.prune_num = int(out_channels*self.simi_prune)
        # self.prune_num = None
        # self.fuse_factor = cfg.FUSE
        self.simi_type = cfg.SIMI_TYPE
        # self.simi_ch = cfg.SIMI_CH
        self.masked = None

        self._to_sup = False
        self.cos_loss = None
        self.weight_dummy = None

        self.renew_counter = 0
    

        print("init my MConv Module")


    def forward(self, input):
        if self._to_sup:
            if self.renew_counter % cfg.RENEW == 0:
                self.renew_counter = 0
                if self.simi_type == 'cos':
                    self.renew_mask_cos()
                elif self.simi_type == 'l2':
                    self.renew_mask_norm()
            # print("self.renew_counter: ", self.renew_counter, self)
            self.renew_counter += 1
            self.do_sup()
            # self.check_dummy()

            

            # self._to_sup = False

        m_weight = self.weight
        output = F.conv2d(input, m_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return output
    
    
    def do_sup(self):
        dim = cfg.SIMI_CH
        if self.masked is None:
            return 0
        # weight = self.weight.detach()
        weight = self.weight.detach()
        o_ch = weight.shape[0]
        shape = weight.shape
        
        # # print(weight.requires_grad)
        # self.weight_dummy.requires_grad_(True)
        # print(weight.requires_grad)
        if dim == "out":
            weight_mat = weight.view(o_ch, -1)
            weight_dummy_mat = self.weight_dummy.detach().view(o_ch, -1)
        elif dim == 'ikk':
            weight_mat = weight.view(o_ch, -1).t()
            weight_dummy_mat = self.weight_dummy.view(o_ch, -1).t()
        weight_sup = self.masked*weight_mat + (1-self.masked)*weight_dummy_mat
        if dim == "out":
            weight_sup = weight_sup.view(shape)
        elif dim == 'ikk':
            weight_sup = weight_sup.t().view(shape)
        self.weight_dummy = weight_sup.clone().detach()
        self.weight.data = weight_sup.data



    def to_sup(self):
        self._to_sup = True
    

    def renew_mask_cos(self):
        # pass
        # grad =  self.weight.grad[:3, :3, 0, 0]
        # print(grad)
        if self.weight_dummy is None:
            self.weight_dummy = self.weight.clone().detach()

        dim = cfg.SIMI_CH
        weight = self.weight.detach()
        o_ch = self.weight.shape[0]
        if dim == "out":
            weight_mat = weight.view(o_ch, -1)
            weight_dummy_mat = self.weight_dummy.view(o_ch, -1)
            self.masked = weight.new(o_ch, 1).fill_(0)
            num_f = o_ch
        elif dim == 'ikk':
            weight_mat = weight.view(o_ch, -1).t()
            weight_dummy_mat = self.weight_dummy.view(o_ch, -1).t()
            num_f = np.prod(weight.shape[1:])
            self.masked = weight.new(num_f, 1).fill_(0)
        simi_row = simi_torch(weight_mat).abs()

        # print("weight_mat.shape: ", weight_mat.shape)
        # print("simi_row.shape: ", simi_row.shape)

        simi = simi_row.view(num_f, num_f).sum(dim=1)

        simi_dummy_row = simi_torch(weight_dummy_mat).abs()
        simi_dummy = simi_dummy_row.view(num_f, num_f).sum(dim=1)
        simi_delta = simi_dummy - simi
        simi_idx = torch.argsort(simi_delta)

        
        # print("self.masked.shape: ", self.masked.shape)
        for i in range(self.keep):
            if cfg.POS:
                self.masked[simi_idx[-(i+1)], 0] = 1
            else:
                self.masked[simi_idx[i], 0] = 1
        
        if self.weight.shape[1] == 3:
            print("cfg.POS: ", cfg.POS)
            print("simi: ", simi)
            print("simi_dummy: ", simi_dummy)
            print("simi_delta: ", simi_delta)
            print("simi_idx: ", simi_idx)
            print("self.masked: \n", self.masked.t())


    def renew_mask_norm(self):
        # pass
        # grad =  self.weight.grad[:3, :3, 0, 0]
        # print(grad)
        if self.weight_dummy is None:
            self.weight_dummy = self.weight.clone().detach()

        dim = cfg.SIMI_CH
        weight = self.weight.detach()
        o_ch = self.weight.shape[0]
        if dim == "out":
            weight_mat = weight.view(o_ch, -1)
            weight_dummy_mat = self.weight_dummy.view(o_ch, -1)
            self.masked = weight.new(o_ch, 1).fill_(0)
        elif dim == 'ikk':
            weight_mat = weight.view(o_ch, -1).t()
            weight_dummy_mat = self.weight_dummy.view(o_ch, -1).t()
            self.masked = weight.new(weight_mat.shape[0], 1).fill_(0)
        dist_row = dist_torch(weight_mat)

        # print("weight_mat.shape: ", weight_mat.shape)
        # print("dist_row.shape: ", dist_row.shape)

        dist = dist_row.view(o_ch, o_ch).sum(dim=1)

        dist_dummy_row = dist_torch(weight_dummy_mat)
        dist_dummy = dist_dummy_row.view(o_ch, o_ch).sum(dim=1)
        dist_delta = dist_dummy - dist
        dist_idx = torch.argsort(dist_delta)

        # print("self.masked.shape: ", self.masked.shape)
        for i in range(self.keep):
            if cfg.POS:
                self.masked[dist_idx[-(i+1)], 0] = 1
            else:
                self.masked[dist_idx[i], 0] = 1
        
        # if self.weight.shape[1] == 3:
        #     print("dist: ", dist)
        #     print("dist_dummy: ", dist_dummy)
        #     print("dist_delta: ", dist_delta)
        #     print("dist_idx: ", dist_idx)
        #     print("self.masked: \n", self.masked)
        
    def init_mask(self):
        o_ch = self.weight.shape[0]
        if cfg.SIMI_CH == 'out':
            self.masked = self.weight.new([o_ch, 1]).fill_(1)
        elif cfg.SIMI_CH == 'ikk':
            ikk = np.prod(self.weight.shape[1:])
            self.masked = self.weight.new([ikk, 1]).fill_(1)

    def check_dummy(self):
        if self.weight.shape[1] == 3:
            if self.weight_dummy is None:
                self.weight_dummy = self.weight.clone().data
            print('-----')
            print("self.weight[:3, :3, 0, 0]: ", self.weight[:3, :3, 0, 0])
            print("self.weight_dummy[:3, :3, 0, 0]: ", self.weight_dummy[:3, :3, 0, 0])
            self.weight_dummy.data = self.weight.clone().data
    
    # def regu_grad(self):
    #     self.cos_loss.backward(retain_graph=True)
    #     grad_dummy = self.weight_dummy.grad.data
    #     print("grad_dummy.device: ", grad_dummy.device)
    #     self.weight.grad.data += grad_dummy*cfg.SIMI

    #     if self.weight.shape[1] == 3:
    #         print(grad_dummy[:3,:3, 0, 0])

        # print(cfg.SIMI, self)

