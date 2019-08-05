from collections import namedtuple
import math
import numpy as np
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

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




class MConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.norm_prune = 1.0
        self.simi_prune = cfg.SIMI
        self.register_buffer("filter_mask", torch.zeros([out_channels, 1]))
        # self.prune_num = int(out_channels*self.simi_prune)
        self.prune_num = None
        self.fuse_factor = cfg.FUSE
        self.simi_type = cfg.SIMI_TYPE
        self.simi_ch = cfg.SIMI_CH
        self.masked = None

        self._to_mask = None

        print("init my MConv Module")


    def forward(self, input):
        if self._to_mask:
            self.do_prune()
            self._to_mask = False

        m_weight = self.weight
        output = F.conv2d(input, m_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return output
    
    def do_prune(self):
        if cfg.MODE == 'merge':
            return self.do_merge()
        elif cfg.MODE == 'gm':
            return self.gm_prune()
        else:
            raise NotImplementedError
        

    def do_merge(self):
        # dim = "out"
        dim = cfg.SIMI_CH
        weight = self.weight.detach()

        if dim == "out":
            weight_mat = weight.view(weight.shape[0], -1)
        elif dim == 'ikk':
            weight_mat = weight.view(weight.shape[0], -1).t()
        # 
        # simi_graph = self.simi_measure(weight_mat, weight_mat)
        print(self.simi_type)
        pdb.set_trace()

        norm_then_merge = True
        if norm_then_merge:
            small, large, small_idx, large_idx = split_via_norm(weight_mat, all=True)

        # small, large = split_via_norm(weight_mat)
        # ccc = split_via_norm(weight_mat)
        # print("cccccc", ccc.shape)
        # small = ccc[:5, :]
        # large = ccc[5:, :]
        # small, large = ccc[:5, :], ccc[5:, :]
        # print("small: ", small)
        # print("large: ", large)
        
        # # print(split_via_norm(weight_mat))

        simi_small = simi_cal(small, self.simi_type)
        simi_large = simi_cal(large, self.simi_type)

        simi_graph = simi_cal(weight_mat, self.simi_type)
        self.prune_num = int(weight_mat.shape[0]*self.simi_prune)
        norm = torch.norm(weight_mat, 2, 1)
        # hist,bins = np.histogram(simi_graph, bins =  10)  
        plt.hist(simi_small[simi_small>-900], bins =  30) 
        plt.hist(simi_large[simi_large>-900]+2, bins =  30) 
        # # plt.hist(simi_graph[simi_graph>-900], bins =  30) 
        # # plt.hist(norm+2, bins =  10) 
        # # plt.hist(simi_graph, bins =  30) 
        plt.title("histogram") 
        plt.show()

        if norm_then_merge:
            merge_map = get_merge_map_221(simi_large, large, self.prune_num-())
        else:
            merge_map = get_merge_map_221(simi_graph, weight_mat, self.prune_num)
        pprint(merge_map)
        np.savetxt("./simi.csv", simi_graph, delimiter=',', fmt='%.2f')
        print(simi_graph)
        


        # if self.weight.shape[1] == 64:
        #     pdb.set_trace()

        self.masked = [i for k, i in merge_map.items()]

        if dim == "out":
            new_weight = merge_221(merge_map, weight_mat, self.fuse_factor).view(weight.shape)
        elif dim == 'ikk':
            new_weight = merge_221(merge_map, weight_mat, self.fuse_factor).t().view(weight.shape)

        # if weight.shape[1] == 3:
        #     pprint(self.masked)
            # print("merged weight \n", new_weight[:,:3, 0, 0])
        self.weight.data = new_weight
        # if weight.shape[1] == 64:
        #     pprint(merge_map)
        #     print("merged weight \n", self.weight.data[:,:3, 0, 0])
        # print(simi_graph.sort())
        # print(np.amax(simi_graph, axis=1))
        # print(np.amax(a, axis=0))
    

    def gm_prune(self):
        weight = self.weight.detach()
        # weight_mat = weight.view(weight.shape[0], -1).t()
        weight_mat = weight.view(weight.shape[0], -1)
        # simi_graph = self.simi_measure(weight_mat, weight_mat)
        simi_graph = simi_cal(weight_mat, self.simi_type)
        self.prune_num = int(weight.shape[0]*self.simi_prune)

        norm = torch.norm(weight_mat, 2, 1)
        norm_np = norm.cpu().numpy()

        filter_large_index = norm_np.argsort()[0:]
        filter_small_index = norm_np.argsort()[:0]

        num = self.prune_num
        similar_sum = np.sum(np.abs(simi_graph), axis=0)
        similar_large_index = similar_sum.argsort()[num:]
        # print(similar_sum[similar_large_index])
        # raise ValueError
        similar_small_index = similar_sum.argsort()[:num] 
        similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
        self.masked = similar_index_for_filter
        # here similar is distance, small one need to be pruned.
        for i in similar_small_index:
            weight_mat[i, ...] = 0
        self.weight.data = weight_mat.view(weight.shape)            

    def mask_grad(self):
        assert not self.masked is None
        dummy = self.weight.grad.clone()

        if cfg.SIMI_CH == 'ikk' and cfg.MODE == 'merge':
            dummy = dummy.view(self.weight.shape[0], -1).t()
        else:
            dummy = dummy.view(self.weight.shape[0], -1)
        for i in self.masked:
            dummy[i, ...] = 0
        # dummy = dummy.t().view(self.weight.shape)    
        if cfg.SIMI_CH == 'ikk' and cfg.MODE == 'merge':
            dummy = dummy.t().view(self.weight.shape)
        else:
            dummy = dummy.view(self.weight.shape)

        # dummy = dummy.view(self.weight.shape)    
        # self.weight.grad.data[i, ...] *= 1e-3
        self.weight.grad.data = dummy
        # if self.weight.grad.shape[1] == 3:
        #     print("grad: ", self.weight.grad.data[:, :3, 0, 0])
        # for i in range(len(self.weight.shape[0])):# self.out_channels try it.


    def to_mask(self):
        self._to_mask = True