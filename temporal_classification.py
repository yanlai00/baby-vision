import argparse
import os
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import GaussianBlur
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR

import numpy as np

import wandb

parser = argparse.ArgumentParser(description='Temporal classification with headcam data')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--val-data', help='path to validation dataset')
parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnext101_32x8d', 'resnext50_32x4d',
                                                            'mobilenet_v2'], help='model')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default'
                                                                               ':16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', default='adam', type=str, help='optimizer, Choices: ["adam", "sgd"]')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed '
                                                                                     'training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use tePyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--n_out', default=1000, type=int, help='output dim')
parser.add_argument('--augmentation', default=True, action='store_false', help='whether to use data augmentation?')
parser.add_argument('--partition', default='SAY', type=str, help='which partition to process. Choices: [S, A, Y, SAY]')

SEG_LEN = 288
FPS = 5

def main():
    args = parser.parse_args()

    print(args)
    wandb.init(project="baby-vision", entity="yanlaiy", config=args)
    wandb.config = args

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print('Model:', args.model)
    model = models.__dict__[args.model](pretrained=False)
    if args.model.startswith('res'):
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)
    else:
        model.classifier = torch.nn.Linear(in_features=1280, out_features=args.n_out, bias=True)

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=1)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise NotImplementedError()
    print(f"Using Optimizer {args.optim} with learing rate {args.lr}")
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print(args.resume)
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    exp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments')
    savefile_dir = f'{args.model}_{args.batch_size}_{args.augmentation}_{args.partition}_{FPS}_{SEG_LEN}_{date_time}'
    exp_path = os.path.join(exp_path, savefile_dir)
    Path(exp_path).mkdir(parents=True, exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.augmentation:
        train_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                        transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                        ])
        )
    else:
        train_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        )

    val_dataset = datasets.ImageFolder(
        args.val_data,
        transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )

    # Filter the image folders according to args.partition
    if args.partition != 'SAY':
        assert args.partition in ['S', 'A', 'Y']
        train_dataset.class_to_idx = {k: v for (k, v) in train_dataset.class_to_idx.items() if k.startswith(args.partition)}
        train_idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] in train_dataset.class_to_idx.values()]
        train_dataset = Subset(train_dataset, train_idx)

        val_dataset.class_to_idx = {k: v for (k, v) in val_dataset.class_to_idx.items() if k.startswith(args.partition)}
        val_idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] in val_dataset.class_to_idx.values()]
        val_dataset = Subset(val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    step = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        val(val_loader, model, criterion, step, args)
        # train for one epoch
        step = train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(exp_path, f'epoch_{epoch}.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()

    num_steps = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        step = epoch * num_steps + i

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses = loss.item()
        top1 = acc1[0]
        top5 = acc5[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            wandb.log({'train_loss': losses}, step=step)
            wandb.log({'train_top1': top1}, step=step)
            wandb.log({'train_top5': top5}, step=step)
            print(num_steps)
            print(losses)

    return step

def val(val_loader, model, criterion, step, args):
    # switch to eval mode
    model.eval()

    losses = []
    top1 = []
    top5 = []

    for i, (images, target) in enumerate(val_loader):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        top1.append(acc1[0].cpu())
        top5.append(acc5[0].cpu())

    wandb.log({'val_loss': np.mean(losses)}, step=step)
    wandb.log({'val_top1': np.mean(top1)}, step=step)
    wandb.log({'val_top5': np.mean(top5)}, step=step)

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

def average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    main()