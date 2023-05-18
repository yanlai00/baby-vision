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
from linear_decoding import load_split_train_test, validate
from linear_decoding import train as val_step
from torch.utils.data import DataLoader, Dataset

import time
import numpy as np

import wandb

parser = argparse.ArgumentParser(description='Temporal classification with headcam data')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--val-data', help='path to validation dataset')
parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnext101_32x8d', 'resnext50_32x4d',
                                                            'mobilenet_v2', 'convnext_tiny', 'convnext_large'], help='model')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default'
                                                                               ':16)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer, Choices: ["adam", "sgd"]')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--lp_lr', '--lp_learning-rate', default=0.0005, type=float, help='initial learning ratefor downstream task')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--cluster_resume', default='', type=str, metavar='PATH', help='path to KMeans clustering result')
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
parser.add_argument('--num-classes', default=26, type=int, help='number of classes in downstream classification task')
parser.add_argument('--lp_epochs', default=40, type=int, metavar='N', help='number of total epochs to run in linear probing')
parser.add_argument('--close_pre_probing', default=True, action='store_false',
                    help='whether to do a linear probing before actually training the network')

def set_parameter_requires_grad(model, train=True):
    for param in model.parameters():
        param.requires_grad = train

def main():
    args = parser.parse_args()

    print(args)
    wandb.init(project="baby-vision", name = "tc", entity="yanlaiy", config=args)
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print('Model:', args.model)
    model = models.__dict__[args.model](pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)

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
    savefile_dir = f'{args.model}_{args.batch_size}_{args.augmentation}_{args.partition}_{date_time}'
    exp_path = os.path.join(exp_path, savefile_dir)
    print(exp_path)
    args.exp_path = exp_path
    Path(exp_path).mkdir(parents=True, exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    kmeans_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )

    # Filter the image folders according to args.partition
    if args.partition != 'SAY':
        assert args.partition in ['S', 'A', 'Y']
        class_to_idx = {k: v for (k, v) in train_dataset.class_to_idx.items() if k.startswith(args.partition)}
        #class_to_idx = {k: v for (k, v) in train_dataset.class_to_idx.items() if k.startswith("A_class_000")}
        train_idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] in class_to_idx.values()]
        train_dataset = Subset(train_dataset, train_idx)
        train_dataset.class_to_idx = class_to_idx
        kmeans_dataset = Subset(kmeans_dataset, train_idx)
        kmeans_dataset.class_to_idx = class_to_idx
    else:
        class_to_idx = train_dataset.class_to_idx
        train_dataset = Subset(train_dataset, range(len(train_dataset)))
        kmeans_dataset = Subset(kmeans_dataset, range(len(kmeans_dataset)))
        train_dataset.class_to_idx = class_to_idx
        kmeans_dataset.class_to_idx = class_to_idx

    step = args.start_epoch * len(train_dataset) // args.batch_size

    for epoch in range(args.start_epoch, args.epochs):
        
        if epoch % 2 == 1:
            set_parameter_requires_grad(model, False)
            val(args.val_data, model, criterion, step, args)
            set_parameter_requires_grad(model, True)
        if epoch % args.epochs == 0 or epoch == args.start_epoch:
            start_time = time.time()
            last_layer = model.module.fc
            model.module.fc = torch.nn.Identity()
        
            set_parameter_requires_grad(model, False)
            if args.cluster_resume:
                print("load clusters from " + args.cluster_resume)
                labels = torch.load(args.cluster_resume)
            else:
                print("KMeans start")
                labels = get_labels(kmeans_dataset, model)
                print("KMeans end")
                end_time = time.time()
                print("KMeans takes " + str(- start_time + end_time))
            torch.save(labels, os.path.join(args.exp_path, f'cluster_{epoch}.tar'))
            
            model.module.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True).cuda()
            #model.module.fc = last_layer
            rename_optimizer = torch.optim.Adam([model.module.fc.weight], args.lp_lr, weight_decay=args.weight_decay)
            #set_parameter_requires_grad(model, True)
    
        train_loader = torch.utils.data.DataLoader(
            ModifiedSubset(train_dataset, labels), batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None
        )
        
        # train for one epoch
        if epoch % args.epochs == 0 and args.close_pre_probing:
            print("linear probing")
            step = train(train_loader, model, criterion, rename_optimizer, epoch, args)
            set_parameter_requires_grad(model, True)
        else:
            set_parameter_requires_grad(model, True)
            print("train")
            step = train(train_loader, model, criterion, optimizer, epoch, args)
            scheduler.step()
            torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(exp_path, f'epoch_{epoch}.tar'))
            
def cluster_loss(embs):
    embs_mean = torch.mean(embs, dim=0)
    return torch.sum((embs - embs_mean) ** 2)

def total_cluster_loss(embs, changepoint):
    cls1 = embs[:changepoint]
    cls2 = embs[changepoint:]
    return cluster_loss(cls1) + cluster_loss(cls2)

def temporal_cluster(embs):
    losses = []
    for changepoint in range(1, len(embs) - 1):
        losses.append(total_cluster_loss(embs, changepoint))
    losses = torch.tensor(losses)
    best_changepoint = torch.argmin(losses)

    return best_changepoint
    
def get_labels(train_dataset, model):
    device = torch.device('cuda')
    labels = torch.tensor(list(train_dataset.class_to_idx.values()))
    targets = torch.tensor(())
    model.eval()
    print(len(labels))

    all_train_idxs = []
    for group_label in range(len(labels)):
         all_train_idxs.append([])
    for i in range(len(train_dataset.dataset)):
        if train_dataset.dataset.imgs[i][1] < len(all_train_idxs):
            all_train_idxs[train_dataset.dataset.imgs[i][1]].append(i)

    for group_label in range(len(labels)):
        if group_label % 50 == 0:
            print(group_label)
        one_train_dataset = Subset(train_dataset.dataset, all_train_idxs[group_label])
        train_loader = torch.utils.data.DataLoader(one_train_dataset, batch_size=128, shuffle=False, pin_memory=True)
        rep = []
        for idx, (images, tc_label) in enumerate(train_loader):
            with torch.no_grad():
                output = model(images.to(device))
            output = (output.permute(1, 0) / torch.linalg.norm(output, dim = -1)).permute(1, 0) # (batch_size, feature_dim)
            rep.append(output)
        rep = torch.vstack(rep) # (batch_size, feature_dim)
    
        if group_label == 0:
            # First iteration
            emb1 = None
            emb2 = rep
        else:
            emb1 = emb2
            emb2 = rep
            # import pdb; pdb.set_trace()
            emb = torch.cat((emb1, emb2), dim=0) # Check the dim
            best_changepoint = temporal_cluster(rep)
            emb2 = emb[best_changepoint:]
            curr_targets = torch.full((best_changepoint, ), group_label - 1).cpu()
            targets = torch.cat((targets, curr_targets))
            if group_label == len(labels) - 1:
                # Last iteration
                curr_targets = torch.full((len(emb2), ), group_label).cpu()
                targets = torch.cat((targets, curr_targets))
    return targets.long()

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
            print(epoch, losses)

    return step

def val(val_dir, model, criterion, step, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_layer = model.module.fc
    model.module.fc = torch.nn.Linear(2048, args.num_classes).to(device)
    optimizer = torch.optim.Adam([model.module.fc.weight], args.lp_lr, weight_decay=args.weight_decay)
    args.subsample = False
    val_train, val_test = load_split_train_test(val_dir, args)
    for epoch in range(args.lp_epochs):
        train_top1 = val_step(val_train, model, criterion, optimizer, epoch, args)
    val_top1 = validate(val_test, model, args)[0]
    wandb.log({'val_top1': val_top1}, step=step)
    if args.model.startswith('res'):
        model.module.fc = last_layer
    elif not args.model.startswith('convnext'):
        model.module.classifier = last_layer
    else:
        model.module.classifier[-1] = last_layer

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
