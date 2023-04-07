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
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader, Dataset

import wandb

parser = argparse.ArgumentParser(description='Temporal classification with headcam data')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--val-data', help='path to validation dataset')
parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnext101_32x8d', 'resnext50_32x4d',
                                                            'mobilenet_v2', 'convnext_tiny', 'convnext_large'], help='model')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default'
                                                                               ':16)')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 1000)')
parser.add_argument('-g', '--group_size', default=10, type=int, metavar='N', help='online group size')
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
    wandb.init(project="baby-vision", entity="peiqiliu")
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
    if args.model == 'convnext_tiny':
        #model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model = models.convnext_tiny()
        #model.classifier[-1] = torch.nn.Identity()
    if args.model == 'convnext_large':
        #model = models.convnext_large(weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model = models.convnext_large()
        #model.classifier[-1] = torch.nn.Identity()
    else:
        model = models.__dict__[args.model](pretrained=False)
    if args.model.startswith('res'):
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)
    elif not args.model.startswith('convnext'):
        model.classifier = torch.nn.Linear(in_features=1280, out_features=args.n_out, bias=True)
    else:
        model.classifier.append(torch.nn.Linear(1000, args.group_size))

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
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

    #val_dataset = datasets.ImageFolder(
    #    args.val_data,
    #    transforms.Compose([
    #        transforms.ToTensor(),
    #        normalize
    #    ])
    #)
    # hyperparameters here
    n_groups = len(torch.unique(torch.Tensor(train_dataset.targets))) // args.group_size

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.batch_size, shuffle=True,
    #    num_workers=args.workers, pin_memory=True, sampler=None
    #)
    #indices = (torch.tensor(val_dataset.targets)[..., None] == 0).any(-1).nonzero(as_tuple=True)[0]
    #val_dataset = torch.utils.data.Subset(val_dataset, indices)
    #val_loader = torch.utils.data.DataLoader(
    #    val_dataset, batch_size=args.batch_size,
    #    num_workers=args.workers, pin_memory=True, sampler=None
    #)

    step = 0

    #for epoch in range(args.start_epoch, args.epochs):
    for group in range(n_groups):
        # train for one epoch
        indices = torch.tensor(train_dataset.targets)[..., None] == -1
        for i in range(group * args.group_size, (group + 1) * args.group_size):
            indices = torch.logical_or(torch.tensor(train_dataset.targets)[..., None] == i, indices)
        indices = indices.any(-1).nonzero(as_tuple=True)[0]
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == 'convnext_tiny':
            model.module.classifier[-1] = torch.nn.Linear(1000, args.group_size).to(device)
        val_subset = train(train_subset, model, criterion, optimizer, group, args)
        #val_indices = (torch.tensor(val_dataset.targets)[..., None] == group * args.group_size).any(-1).nonzero(as_tuple=True)[0]
        #val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None
        )
        val(val_loader, model, criterion, group, args)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(exp_path, f'group_{group}.tar'))

class ModifiedSubset(Dataset):
    def __init__(self, subset, new_labels):
        self.subset = subset
        self.new_labels = new_labels
        
    def __getitem__(self, index):
        image, _ = self.subset[index]
        new_label = self.new_labels[index]
        return image, new_label
    
    def __len__(self):
        return len(self.subset)        

def get_prior(label, n, beta):
    return torch.softmax(beta * torch.roll(torch.abs(torch.linspace(1, -1, n + 1))[:-1], label), -1)

def train(train_subset, model, criterion, optimizer, group, args):
    rep = []
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(group), 'KMeans Start')
    model.eval()
    if args.model.startswith('convnext'):
        last_layer = model.module.classifier[-1]
        model.module.classifier[-1] = torch.nn.Identity()
    for idx, (example, label) in enumerate(train_subset):
        if idx % 1000 == 0:
            print(idx, label)
        with torch.no_grad():
            example = model(example.unsqueeze(0).to(device)).squeeze(0).cpu().detach().numpy()
        example = example / np.linalg.norm(example)
        prior = get_prior(label, args.group_size, max(16 - 3 * group, 0))
        example = np.concatenate((example, prior))
        rep.append(example)
    if args.model.startswith('convnext'):
        model.module.classifier[-1] = last_layer
    rep = np.vstack(rep)
    
    # Another hyperparameter to tune
    
    kmeans = KMeans(n_clusters= args.group_size, random_state=0).fit(rep)
    labels = torch.Tensor(kmeans.labels_).long()
    print(str(group), "KMeans end")
    
    train_loader = torch.utils.data.DataLoader(
        ModifiedSubset(train_subset, labels), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    
    # switch to train mode
    model.train()
    for epoch in range(args.epochs):
        num_steps = len(train_loader)
        for i, (images, target) in enumerate(train_loader):
            step = group * args.epochs * num_steps + epoch * num_steps + i

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            #target = torch.Tensor(labels[args.batch_size * i : min(args.batch_size * (i + 1), len(labels))]).cuda()
            target, images = target.cuda(), images.cuda()
            
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
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
                wandb.log({'train_top3': top5}, step=step)
                print(group, epoch)
                print(losses)

    return ModifiedSubset(train_subset, labels)
    #return step

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
    wandb.log({'val_top3': np.mean(top5)}, step=step)

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
