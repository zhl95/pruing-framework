import torch
import torch.nn as nn

import os, sys, shutil, time, random

# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
from utils import *
from functions.mask import Mask
from config_ import cfg

from models.modules.prune import simi_cal

if cfg.MODE == 'merge' or cfg.MODE == 'gm':
  print(cfg.MODE, '==')
  from models.modules.mconv import MConv2d
  nn.Conv2d = MConv2d
#   pdb.set_trace()
elif cfg.MODE == 'regu':
  from models.modules.rconv import RConv2d
  nn.Conv2d = RConv2d

def cos_loss(net):
    for m in net.modules():
      if isinstance(m, nn.Conv2d):
        print(m)
        
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class Trainer(object):
    def __init__(self, net, criterion, dloader=(None, None), optimizer=None,
                args=None):  ## too many arguments make a bad paradigm.
        
        if args.use_cuda:
            net.cuda()
            criterion.cuda()
            # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
        self.net = net

        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.train_loader = dloader[0]
        self.test_loader = dloader[1]

        self.optimizer = optimizer
        self.args = args
        

        self.start_time = time.time()
        self.epoch_time = AverageMeter()
        self.small_filter_index = []
        self.large_filter_index = []

        self.log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
        # self.recorder = RecorderMeter(args.epochs)
        self.recorder = RecorderMeter2(args.epochs)
        self.try_resume()

        self.simi_loss = 0

        
    
    def try_resume(self):
        args = self.args
        log = self.log
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print_log("=> loading checkpoint '{}'".format(args.resume), log)
                checkpoint = torch.load(args.resume)
                recorder = checkpoint['recorder']
                args.start_epoch = checkpoint['epoch']
                if args.use_state_dict:
                    self.net.load_state_dict(checkpoint['state_dict'])
                else:
                    self.net = checkpoint['state_dict']

                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
            else:
                print_log("=> no checkpoint found at '{}'".format(args.resume), log)
        else:
            print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    def try_finetune(self):
        args = self.args
        # net = self.net
        log = self.log

        if args.use_pretrain:
            if os.path.isfile(args.pretrain_path):
                print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
            else:
                dir = '/data/yahe/cifar10_base/'
                # dir = '/data/uts521/yang/progress/cifar10_base/'
                whole_path = dir + 'cifar10_' + args.arch + '_base'
                args.pretrain_path = whole_path + '/checkpoint.pth.tar'
                print_log("Pretrain path: {}".format(args.pretrain_path), log)
            pretrain = torch.load(args.pretrain_path)
            if args.use_state_dict:
                self.net.load_state_dict(pretrain['state_dict'])
            else:
                self.net = pretrain['state_dict']


    def train_all(self):
        args = self.args
        optimizer = self.optimizer
        recorder = self.recorder
        log = self.log
        net = self.net
        start_time = self.start_time

        # self.compare_before_after()

        epoch_time = self.epoch_time
        for epoch in range(args.start_epoch, args.epochs):
            self.epoch = epoch
            current_learning_rate = self.adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

            need_hour, need_mins, need_secs = convert_secs2time(self.epoch_time.avg * (args.epochs - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

            print_log(
                '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                        need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                    100 - recorder.max_accuracy(False)), log)

            if (epoch + 1 > cfg.START and epoch % args.epoch_prune == 0) or epoch == args.epochs - 1:  # prune every args.epoch_prune
                
                try:
                    self.net.to_mask()
                except AttributeError:
                    print("net DataParallel wrapped")
                    self.net.module.to_mask()
            # train for one epoch
            train_acc, train_los = self.train_epoch()

            # evaluate on validation set
            val_acc_1, val_los_1 = self.validate()


            val_acc_2, val_los_2 = self.validate()

            # is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
            is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2, self.simi_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net,
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_path, 'checkpoint.pth.tar')

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            recorder.plot_curve(os.path.join(args.save_path, 'curve{}.png'.format(args.manualSeed)))

        log.close()


    def adjust_learning_rate(self, optimizer, epoch, gammas, schedule):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.learning_rate
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def validate(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        val_loader = self.test_loader
        model = self.net
        criterion = self.criterion
        log = self.log
        args = self.args

        # switch to evaluate mode
        model.eval()

        self.simi_loss = model.cos_loss()*1.0

        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                    error1=100 - top1.avg),
                log)

        print_log('  **Simi loss**  {:.3f}'.format(self.simi_loss), log)

        return top1.avg, losses.avg


    # train function (forward, backward, update)
    def train_epoch(self):

        train_loader = self.train_loader
        model = self.net
        criterion = self.criterion
        optimizer = self.optimizer
        # epoch, log, m
        args = self.args
        epoch = self.epoch
        log = self.log

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # cos_loss = model.cos_loss()*args.regu
            # cos_loss = model.cos_loss()*1e2
            self.simi_loss = model.get_simi()
            # loss += cos_loss
            # loss = cos_loss
            # print("loss", loss)
            # print("cos_loss", cos_loss)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            # if i == 0:
            #     loss.backward(retain_graph=True)
            # else:
            #     loss.backward()
            # loss.backward(retain_graph=True)

            # Mask grad for iteration
            # m.do_grad_mask()
            if(epoch + 1 > cfg.START and epoch % args.epoch_prune == 0) or epoch == args.epochs - 1:
                try:
                    self.net.mask_grad()
                except AttributeError:
                    print("DataParallel wrapped")
                    self.net.module.mask_grad()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Simi loss {simi:.4f}   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, simi=self.simi_loss) + time_string(), log)
        print_log(
            '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                error1=100 - top1.avg),
            log)
        return top1.avg, losses.avg


    def compare_before_after(self):
        net = self.net
        # m = self.mask
        # m.init_length()
        args = self.args
        
        print("-" * 10 + "one epoch begin" + "-" * 10)
        print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
        print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
        print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))

        val_acc_1, val_los_1 = self.validate()

        print(" accu before is: %.3f %%" % val_acc_1)
        # m.model = net

        # m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
        # #    m.if_zero()
        # m.do_mask()
        # m.do_similar_mask()
        # net = m.model
        #    m.if_zero()
        if args.use_cuda:
            net = net.cuda()
        val_acc_2, val_los_2 = self.validate()
        print(" accu after is: %s %%" % val_acc_2)

