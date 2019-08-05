# from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
from utils import *

import models

import numpy as np
import pickle
from scipy.spatial import distance
import pdb

from trainer2 import Trainer
from config_ import cfg, cfg_from_file, cfg_from_list


model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_simi', type=float, default=0.4, help='the reducing ratio of pruning based on Distance')
parser.add_argument('--fuse', type=float, default=0.8, help='merge factor')
parser.add_argument('--simi_type', type=str, default='l2', help='similiarity measurement')
parser.add_argument('--simi_ch', type=str, default='out', help='out channle or ikk')
parser.add_argument('--mode', type=str, default='merge', help='out channle or ikk')
parser.add_argument('--regu', type=float, default='0.1', help='regu factor')
parser.add_argument('--pos', type=int, help='mask 1 on largerly changed')

parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True



def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_simi), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    # print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)


    
    # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    args.set_cfgs = ['MODE', args.mode] + ['START', args.start_epoch] +\
                    ["SIMI", args.rate_simi] + ["NORM", args.rate_norm] + ["FUSE", args.fuse] +\
                    ['SIMI_TYPE', args.simi_type] + ['SIMI_CH', args.simi_ch] + ["MODE", args.mode] +\
                    ['POS', bool(args.pos)]
    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    from pprint import pprint
    pprint(args)
    print(cfg.MODE)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer

    # import models
    from importlib import reload
    reload(models)
    __import__('models')
    net = models.__dict__[args.arch](num_classes)
    # pdb.set_trace()

    print_log("=> network :\n {}".format(net), log)

    # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)


    # if args.evaluate:
    #     time1 = time.time()
    #     validate(test_loader, net, criterion, log)
    #     time2 = time.time()
    #     print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
    #     return

    # m = Mask(net, args)

    trainer_ = Trainer(net, criterion, (train_loader, test_loader), optimizer, args)
    trainer_.train_all()


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':
    main()
