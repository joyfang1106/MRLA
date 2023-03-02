# -*- coding: UTF-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchmodels
import models

from utils import CosineAnnealingLR, CrossEntropyLabelSmooth, MultiStepLR



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='LANet training and evaluation script')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50_mrla',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50_mrlal)')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--wd-la', '--weight-decay-la', default=1e-4, type=float,
                    metavar='WL', help='weight decay of layer attention (la) layer (default: 1e-4)',
                    dest='weight_decay_la')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.2, metavar='PCT',
                    help='Drop path rate on la layer (default: 0.)')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--action', default='', type=str,
                    help='other information.')
parser.add_argument('--scheduler', default='cos', type=str,
                    help='learning rate scheduler choice (step or cos).')

best_acc1 = 0


def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # # create model
    print("=> creating model '{}' for training".format(args.arch))
    model = models.__dict__[args.arch](drop_rate=args.drop, drop_path=args.drop_path)
        
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            #model.features = torch.nn.DataParallel(model.features)
            #model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        
    # print(model)
    # get the number of models parameters
    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = CrossEntropyLabelSmooth().cuda(args.gpu)
    
    if args.weight_decay_la == args.weight_decay:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else: 
        # extract the parameters names of our layer attention module
        decay = list()
        decay_la = list()
        for name, param in model.named_parameters():
            # print('checking {}'.format(name))
            if 'layer_atten' in name:
                # decay_la[name] = param
                decay_la.append(param)
            else:
                # decay[name] = param
                decay.append(param)

        print('using diff weight dacay for la:', args.weight_decay_la)
        optimizer = optim.SGD([
                    {'params': decay},
                    {'params': decay_la, 'weight_decay': args.weight_decay_la}
                ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = optim.AdamW([
            # {'params': no_decay, 'weight_decay', 0}, 
            # {'params': decay}, **kwargs])
    
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # Imagenet means and standards
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # ===========================================================================
    if args.scheduler == "cos":
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, last_epoch=args.start_epoch-1)
        len_epoch = len(train_loader)
        T_max = 95 * len_epoch
        warmup_iters = 5 * len_epoch
        scheduler = CosineAnnealingLR(optimizer, T_max, warmup='linear', warmup_iters=warmup_iters)
        #T_max = 100 * len_epoch
        #scheduler = CosineAnnealingLR(optimizer, T_max)
    else:
        len_epoch = len(train_loader)
        T_max = 95 * len_epoch
        warmup_iters = 5 * len_epoch
        scheduler = MultiStepLR(optimizer, T_max, milestones=[29*len_epoch+1, 59*len_epoch+1, 89*len_epoch+1],
                                warmup='linear', warmup_iters=warmup_iters)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
    # ===========================================================================

    if args.evaluate:
        m = time.time()
        _, _, _ = validate(val_loader, model, criterion, args)
        n = time.time()
        print("evaluate_time (h): ", (n-m)/3600)
        return

    # create folder
    if not os.path.exists(os.path.abspath(args.work_dir)):
       os.mkdir(os.path.abspath(args.work_dir))  
    # save_path = os.path.join('./'+args.work_dir, args.arch)
    save_path = "%s/%s/"%(args.work_dir, args.arch + '_' + args.action)
    if not os.path.exists(save_path):
       os.mkdir(save_path)
    print ("save checkpoint file and log in: ", save_path)
    
    loss_plot = {}
    valloss_plot = {}
    train_acc1_plot = {}
    train_acc5_plot = {}
    val_acc1_plot = {}
    val_acc5_plot = {}
    lr_list = {}
    
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch)
        # loss_temp, train_acc1_temp, train_acc5_temp = train(train_loader, model, criterion, optimizer, epoch, args)
        loss_temp, train_acc1_temp, train_acc5_temp = train(train_loader, model, criterion, optimizer, epoch, args, scheduler)
        lr_list[epoch]  = optimizer.state_dict()['param_groups'][0]['lr']
        loss_plot[epoch] = loss_temp
        train_acc1_plot[epoch] = train_acc1_temp
        train_acc5_plot[epoch] = train_acc5_temp

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion)
        valloss, acc1, acc5 = validate(val_loader, model, criterion, args)
        valloss_plot[epoch] = valloss
        val_acc1_plot[epoch] = acc1
        val_acc5_plot[epoch] = acc5

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
            if (epoch+1) % 30 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, False, args, filename='checkpoint{}.pth.tar'.format(epoch))
            if epoch > 91 and is_best:
                # print (epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, False, args, filename='checkpoint{}.pth.tar'.format(epoch))

        # save Loss,valloss,train_acc1,train_acc5,val_acc1,val_acc5 in .txt files
        data_save(save_path + 'loss_plot.txt', loss_plot)
        data_save(save_path + 'valloss_plot.txt', valloss_plot)
        data_save(save_path + 'train_acc1.txt', train_acc1_plot)
        data_save(save_path + 'train_acc5.txt', train_acc5_plot)
        data_save(save_path + 'val_acc1.txt', val_acc1_plot)
        data_save(save_path + 'val_acc5.txt', val_acc5_plot)
        data_save(save_path + "lr_plot.txt", lr_list)

        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print("epoch {} train_time (h): {}".format(epoch, time_value))
        print("-" * 80)
        

def train(train_loader, model, criterion, optimizer, epoch, args, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses_batch = {}
    # not use ProgressMeter
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.gpu is None or args.gpu == 0:
            if i % args.print_freq == 0:
                # progress.display(i)
                # lr_temp = gen_lr(epoch, args)
                lr_temp = optimizer.state_dict()['param_groups'][0]['lr']
                print('Epoch: [{0}][{1}/{2}]\t'
                    'LR {3:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), lr_temp,
                    batch_time=batch_time, data_time=data_time, 
                    loss=losses, top1=top1, top5=top5))
                   
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # not use ProgressMeter
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.gpu is None or args.gpu == 0:
                if i % args.print_freq == 0:
                    # progress.display(i)
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
              .format(top1=top1, top5=top5, loss=losses))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    save_path = "%s/%s/"%(args.work_dir, args.arch + '_' + args.action)
    filepath = os.path.join(save_path, filename)
    # './checkpoint_{}_epoch.pkl'.format(epoch)
    bestpath = os.path.join(save_path, 'model_best.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, bestpath)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


if __name__ == '__main__':
    main()
