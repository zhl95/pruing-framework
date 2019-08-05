import torch  
import numpy as np
import os
from collections import OrderedDict

# from .. import models
import sys
sys.path.append("..") 
import models

result_dir = "../result/scrath"

print(os.listdir(result_dir))


def get_model(res_dir, regu):
    model_path = os.path.join(res_dir, 'cifar10_res20_regu{:.2f}'.format(regu), 'checkpoint.pth.tar')
    # print(os.listdir(model_dir))
    checkpoint = torch.load(model_path)

    # print(type(model), 'asdf')
    # print(model.type, 'asdfsadf')
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module
    # print(checkpoint.keys())
    # state_dict = checkpoint['arch']
    # print(type(state_dict))

    model = checkpoint['state_dict']
    return model

    
    # print(checkpoint.keys())
    # print(checkpoint['state_dict'])
    # print(checkpoint['recorder'])


get_model(result_dir, 0.1)
