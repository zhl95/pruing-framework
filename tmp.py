import torch
from scipy.spatial import distance
import numpy as np

# a=torch.arange(9).view(3, 3).float()
# a=torch.randint(9, (3, 3)).float()

# print(a)
# b = a[0].repeat(3,1)
# print(b)
# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# cos2 = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# out = cos(a, b)
# print(out)
# out2 = cos2(a[0], a[2])
# print(out2)

# num = a.shape[0]
# c = [[a[i]]*num for i in range(num)]
# print(c)
# d = [i.unsqueeze(0) for j in c for i in j]
# print(d)
# e = torch.cat(d)
# print(e)
# h = [a]*3
# print(torch.cat(h))

# def simi_torch(tensor_2d):
#     num = tensor_2d.shape[0]
#     a = [[tensor_2d[i]]*num for i in range(num)]
#     a_ = [i.unsqueeze(0) for j in a for i in j]
#     b = [tensor_2d]*num

#     a_th = torch.cat(a_)
#     b_th = torch.cat(b)
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#     simi = cos(a_th, b_th)
    
#     return simi

# res = simi_torch(a)
# print(res)

# out3 = 1-distance.cdist(a, a, 'cosine')
# print(out3)

# inf_ = np.full([3, 3], -np.inf)
# inf_tril = np.tril(inf_)
# print(inf_+1)
# print(inf_tril)

# top = a.numpy().argmax()
# print(np.unravel_index(top, a.shape))
# top2 = a.topk(3)
# print(top)
# print(top2)

# k = {1:2}
# print(k)
# print(1 in k, 2 in k)


# a = [1, 2, 3, 4, 5, 6]

# b = {a[i]:a[i+1] for i in range(0, len(a), 2)}
# print(b)


# a = np.arange(9).reshape([3, 3])
# a = np.random.randint(1, 10, [3, 3])
# print(a)
# dis = distance.cdist(a, a, 'euclidean')
# print(dis)

# print(a[[0, 1, 1], :])

# def split_via_norm(tensor_2d):
#     a = tensor_2d
#     print(a.type())
#     norm = torch.norm(a, 2, 1)
#     print(norm.shape)
#     ratio = 0.5
#     sorted_idx = torch.argsort(norm, dim=-1, descending=True)
#     print(sorted_idx)


# a = torch.randint(10, (3, 3)).float().cuda()
# # a *= 0.1
# print(a)
# split_via_norm(a)

# print(a[[0, 2], :])

# print(type(a.shape[0]))

# import torch
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
# # from pruning_cifar10 import validate, accuracy
# import os

# def validate(val_loader, model, criterion, log):
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda(async=True)
#         input = input.cuda()
#         input_var = torch.autograd.Variable(input, volatile=True)
#         target_var = torch.autograd.Variable(target, volatile=True)

#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)

#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))
#         top5.update(prec5.item(), input.size(0))

#     return top1.avg, losses.avg


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

# train_transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
#         transforms.Normalize(mean, std)])
# test_transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize(mean, std)])


# train_data = dset.CIFAR10('./data/cifar.python', train=True, transform=train_transform, download=True)
# test_data = dset.CIFAR10('./data/cifar.python', train=False, transform=test_transform, download=True)
# num_classes = 10

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,
#                                             num_workers=2, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False,
#                                             num_workers=2, pin_memory=True)

# criterion = torch.nn.CrossEntropyLoss()
# log = open(os.path.join('./', 'log_test.txt'), 'w')

# model_path = os.path.join('./result/scrath', 'cifar10_res20_regu{:.2f}'.format(0.1), 'checkpoint.pth.tar')
# pretrain = torch.load(model_path)
# net = pretrain['state_dict']
# net = net.cuda()
# val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)
# print(" accu before is: %.3f %%" % val_acc_1)


import torch
from models.modules.prune import simi_torch
a_ = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# a_ = torch.randn(64, 64*3*3).cuda()
print(a_.requires_grad)
a_.requires_grad_(True)
print(a_.requires_grad)

b = a_*2
a=a_.detach()
e = a
# print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
# b.back
b = (a * a).sum()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
c = cos(a*a, a)
print(a.shape)
print(c.shape)
d = c.mean()
print("c: ", c)
# print(b.grad_fn)
# print(c.grad_fn)
d.backward()

# print(c.grad)
# print(a.grad)


for i in range(100):
    d = 0
    # print(type(d))
    # c = cos(a*a, a)
    # d = c.mean()
    c = simi_torch(a)
    d += c.mean()
    # print(type(d))
    # print(d.type())
    # print("a: ", a, i)
    print("d: ", d, i)
    d.backward()
    a.data = a.data - a.grad.data*0.1

print(a)
print(a_)
print(e)