from collections import namedtuple
import math
import numpy as np
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

np.set_printoptions(precision=2)

# def cos(a, b):
#     a2 = a*a
#     b2 = b*b
#     a_norm = torch.sum(a2, dim=1, keepdim=True).pow(0.5)
#     b_norm = torch.sum(b2, dim=1, keepdim=True).pow(0.5)
#     a_ = a/a_norm
#     b_ = b/b_norm
#     return (a_*b_).sum(dim=1)




cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
def simi_torch(tensor_2d):
    # print("tensor_2d.requires_grad: ", tensor_2d.requires_grad)
    num = tensor_2d.shape[0]
    a = [[tensor_2d[i]]*num for i in range(num)]
    a_ = [i.unsqueeze(0) for j in a for i in j]
    b = [tensor_2d]*num

    a_th = torch.cat(a_)
    b_th = torch.cat(b)
    
    simi = cos(a_th, b_th)
    # var_simi = Variable(simi, requires_grad=True)

    return simi
    # return var_simi


def dist_torch(tensor_2d):
    num = tensor_2d.shape[0]
    a = [[tensor_2d[i]]*num for i in range(num)]
    a_ = [i.unsqueeze(0) for j in a for i in j]
    b = [tensor_2d]*num

    a_th = torch.cat(a_)
    b_th = torch.cat(b)

    dist = torch.norm(a_th - b_th, p=2, dim=1)
    return dist



def simi_cal(tensor_2d, type):
    try:
        a = tensor_2d.cpu()
        if type =='cos':
            if cfg.MODE == 'merge':
                simi = 1-distance.cdist(a, a, 'cosine') # row-wise distance
                inf_ = np.full(simi.shape, -999.)
                inf_tril = np.tril(inf_)
                # return torch.from_numpy(simi).cuda()
                return simi+inf_tril
            elif cfg.MODE == 'regu':
                simi = 1-distance.cdist(a, a, 'cosine')
                return simi
        elif type == 'l2':
            simi = distance.cdist(a, a, 'euclidean')
            norm = torch.norm(a, 2, 1)
            simi = simi/norm.numpy()
            print("l2:--")
            print(simi[:10, :10])
            # pdb.set_trace()
            return simi
    except AttributeError:
        raise NotImplementedError
        # a = tensor_2d
        # simi = 1-distance.cdist(a, a, 'cosine') # row-wise distance
        # return simi

def split_via_norm(tensor_2d, all=False):
    a = tensor_2d.clone()
    num = a.shape[0]
    norm = torch.norm(a, 2, 1)
    print(norm.shape, a.shape)
    ratio = cfg.NORM
    sorted_idx = torch.argsort(norm, dim=-1, descending=False)
    print(norm)
    print(sorted_idx)
    small_idx = sorted_idx[:int(num*ratio)]
    large_idx = sorted_idx[int(num*ratio):]
    print(small_idx)
    print(large_idx)

    tensor_small = a[small_idx, :]
    tensor_large = a[large_idx, :]
    print("split finished")
    print("tensor_small: ", tensor_small)
    print("tensor_large: ", tensor_large)
    # print(torch.cat([tensor_small, tensor_large]).shape)
    if all:
        return tensor_small, tensor_large, small_idx, large_idx
    else:
        return tensor_small, tensor_large
    # return torch.cat([tensor_small, tensor_large])

def get_merge_map_221(simi, weight_mat, num):
    norm2 = torch.norm(weight_mat, 2, 1)
    norms = norm2.cpu().numpy()
    if num <1:
        return {}
    assert len(simi.shape) == 2
    assert len(norms.shape) == 1
    assert num < simi.shape[0]//2
    i = 0
    merge_map = []
    fail = 0
    while i < num:
        if fail > 1e6:
            break
        top = simi.argmax()
        pair = (np.unravel_index(top, simi.shape))
        simi[pair] = -999
        if norms[pair[0]] > norms[pair[1]]:
            if not pair[0] in merge_map and not pair[1] in merge_map:
                merge_map += [pair[0], pair[1]]
                i += 1
            else:
                fail += 1
                continue
        else:
            if not pair[0] in merge_map and not pair[1] in merge_map:
                merge_map += [pair[1], pair[0]]
                i += 1
            else:
                fail += 1
                continue
    # print("len(merge_map):", len(merge_map))
    return {merge_map[i]:merge_map[i+1] for i in range(0, len(merge_map), 2)}


def merge_221(merge_map, weight_2d, fuse_factor=0.5):
    assert len(weight_2d.shape) == 2
    assert fuse_factor >= 0 and fuse_factor <= 1
    for big, small in merge_map.items():
        # big_ft = weight_2d[big]
        # small_ft = weight_2d[small]
        weight_2d[big] = weight_2d[big]*fuse_factor + weight_2d[small]*(1-fuse_factor)
        weight_2d[small] = 0.
    return weight_2d


# def merge_n21(simi, weight_2d, num):

# def pruning_geo_mean(weight_2d, norm_prune, simi_prune):
    