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
from linear_decoding import AverageMeter, ProgressMeter, load_split_train_test, validate
from linear_decoding import train as val_step
from sklearn.cluster import KMeans
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
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer, Choices: ["adam", "sgd"]')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--lp_lr', '--lp_learning-rate', default=0.0005, type=float, help='initial learning ratefor downstream task')
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
parser.add_argument('--num-classes', default=26, type=int, help='number of classes in downstream classification task')
parser.add_argument('--lp_epochs', default=50, type=int, metavar='N', help='number of total epochs to run in linear probing')

SEG_LEN = 288
FPS = 5

def main():
    args = parser.parse_args()

    print(args)
    wandb.init(project="baby-vision", entity="peiqiliu", config=args)
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
        model = models.convnext_tiny()
    if args.model == 'convnext_large':
        model = models.convnext_large()
    else:
        model = models.__dict__[args.model](pretrained=False)
    if args.model.startswith('res'):
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)
    elif not args.model.startswith('convnext'):
        model.classifier = torch.nn.Linear(in_features=1280, out_features=args.n_out, bias=True)
    else:
        model.classifier.append(torch.nn.Linear(1000, args.n_out))

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
    print(exp_path)
    args.exp_path = exp_path
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

    # Filter the image folders according to args.partition
    if args.partition != 'SAY':
        assert args.partition in ['S', 'A', 'Y']
        train_dataset.class_to_idx = {k: v for (k, v) in train_dataset.class_to_idx.items() if k.startswith(args.partition)}
        train_idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] in train_dataset.class_to_idx.values()]
        train_dataset = Subset(train_dataset, train_idx)

    step = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        val(args.val_data, model, criterion, step, args)
        # train for one epoch
        step = train(train_dataset, model, criterion, optimizer, epoch, args)
        scheduler.step()
        #wandb.log({'lr': scheduler.get_last_lr()}, step=step)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(exp_path, f'epoch_{epoch}.tar'))

        
def get_labels(train_dataset, group_size, model):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = torch.tensor(())
    existing_labels = []
    rep = []
    model.eval()
    cur_time = time.time()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=200, shuffle=False, pin_memory=True
    )
    for idx, (example, label) in enumerate(train_loader):
        if label[-1] not in existing_labels:
            existing_labels.append(label[-1])
            if (len(existing_labels) - 1) % group_size == 0 and len(existing_labels) != 1:
                group = len(existing_labels) // group_size - 1
                rep = np.vstack(rep)
                kmeans = KMeans(n_clusters= group_size, random_state=0, n_init = "auto").fit(rep)
                cur_labels = torch.Tensor(kmeans.labels_).long()
                labels = torch.concatenate((labels, cur_labels + group * group_size))
                rep = []
        if idx % 120 == 0:
            print(idx, time.time() - cur_time)
            cur_time = time.time()
        with torch.no_grad():
            example = model(example.to(device)).cpu().detach().numpy()
        example = example / np.linalg.norm(example)
        rep.append(example)
    if len(rep) != 0:
        group = (len(existing_labels) - 1) // group_size
        n_clusters = len(existing_labels) % group_size if len(existing_labels) % group_size != 0 else group_size
        rep = np.vstack(rep)
        kmeans = KMeans(n_clusters= n_clusters, random_state=0, n_init = "auto").fit(rep)
        cur_labels = torch.Tensor(kmeans.labels_).long()
        labels = torch.concatenate((labels, cur_labels + group * group_size))   
    return labels.long()

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
    
def train(train_subset, model, criterion, optimizer, epoch, args):
    start_time = time.time()
    print("KMeans start")
    if args.model.startswith('convnext'):
        last_layer = model.module.classifier[-1]
        model.module.classifier[-1] = torch.nn.Identity()
    elif args.model.startswith('res'):
        last_layer = model.module.fc
        model.module.fc = torch.nn.Identity()
    else:
        last_layer = model.module.classifier
        model.module.classifier = torch.nn.Identity()
        
    labels = get_labels(train_subset, 10, model)
    torch.save(labels, os.path.join(args.exp_path, f'cluster_{epoch}.tar'))
    
    if args.model.startswith('convnext'):
        model.module.classifier[-1] = last_layer
    elif args.model.startswith('res'):
        model.module.fc = last_layer
    else:
        model.module.classifier = last_layer
    print("KMeans end")
    end_time = time.time()
    print("KMeans takes " + str(- start_time + end_time))
    
    train_loader = torch.utils.data.DataLoader(
        ModifiedSubset(train_subset, labels), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
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

def val(val_dir, model, criterion, step, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model.startswith('res'):
        last_layer = model.module.fc
        model.module.fc = torch.nn.Linear(2048, args.num_classes).to(device)
        optimizer = torch.optim.Adam([model.module.fc.weight], args.lp_lr, weight_decay=args.weight_decay)
    elif not args.model.startswith('convnext'):
        last_layer = model.module.classifier
        model.module.classifier = torch.nn.Linear(1280, args.num_classes).to(device)
        optimizer = torch.optim.Adam([model.module.classifier.weight], args.lp_lr, weight_decay=args.weight_decay)
    else:
        last_layer = model.module.classifier[-1]
        model.module.classifier[-1] = torch.nn.Linear(1000, args.num_classes).to(device)
        optimizer = torch.optim.Adam([model.module.classifier[-1].weight], args.lp_lr, momentum = 0.9, weight_decay=args.weight_decay)
    args.subsample = False
    val_train, val_test = load_split_train_test(val_dir, args)
    for epoch in range(args.lp_epochs):
        top1 = val_step(val_train, model, criterion, optimizer, epoch, args)
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
