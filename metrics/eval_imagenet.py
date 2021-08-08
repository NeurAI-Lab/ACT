from __future__ import print_function
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import numpy as np
import pandas as pd
import time
from utilities.utils import set_torch_seeds
from utilities import dist_utils
from models.resnet import ResNet50


parser = argparse.ArgumentParser(description='PyTorch ImageNet PGD Attack Evaluation')
parser.add_argument('--data_path', default='/data/input/datasets/ImageNet_2012/', help='dataset')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for testing (default: 64)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--checkpoint', default='trained_model')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--model_architecture", type=str, default="resnet50")
# Adversarial Training
parser.add_argument('--step_size', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=2/255)
parser.add_argument('--num_steps', type=int,  default=10)
parser.add_argument('--random', default=True, help='random initialization for PGD')
parser.add_argument('--random_init', type=int,  default=1)
parser.add_argument('--condition_on_correct', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
set_torch_seeds(0)

# set up data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# ======================================================================================
# Configurations
# ======================================================================================
def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(data_dir, cache_dataset, distributed):

    # Data loading code
    print("Loading validation data")
    cache_path = _get_cache_path(data_dir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            dist_utils.mkdir(os.path.dirname(cache_path))
            dist_utils.save_on_master((dataset, data_dir), cache_path)

    print("Creating data loaders")
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return dataset, sampler


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    print('Correct:', correct)
    print('Total:', len(data_loader.dataset))
    print('Accuracy:', accuracy)
    print('Loss:', loss)

    return loss, accuracy, correct


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  condition_on_correct=False):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    if condition_on_correct:
        select_idx = out.data.max(1)[1] == y.data
        X = X[select_idx]
        y = y[select_idx]

    X_pgd = Variable(X.data, requires_grad=True)

    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out_pdg = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_pdg, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    out_pdg = model(X_pgd)

    err_pgd = (out_pdg.data.max(1)[1] != y.data).float().sum()
    attacks_conducted = X.shape[0]

    batch_size = len(y)

    nat_acc = (batch_size - err) / batch_size
    adv_acc = (batch_size - err_pgd) / batch_size

    return nat_acc, adv_acc


def eval_adv_test_whitebox(model, device, test_loader, epsilon, num_steps, condition_on_correct=True):
    """
    evaluate model by white-box attack
    """
    model.eval()

    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    header = 'Robustness'
    for data, target in metric_logger.log_every(test_loader, 1, header):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        nat_acc, adv_acc = _pgd_whitebox(model, X, y, epsilon=epsilon, num_steps=num_steps, condition_on_correct=condition_on_correct)
        batch_size = data.shape[0]
        metric_logger.meters['nat_acc'].update(nat_acc.item(), n=batch_size)
        metric_logger.meters['rob_acc'].update(adv_acc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    nat_acc = metric_logger.nat_acc.global_avg
    rob_acc = metric_logger.rob_acc.global_avg

    print('natural accuracy: ', nat_acc)
    print('adversarial accuracy: ', rob_acc)

    return nat_acc, rob_acc


def eval_checkpoint(checkpoint_path, epsilon=2/255, num_steps=10, num_random_init=1, condition_on_correct=False):

    dist_utils.init_distributed_mode(args)
    print(args)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    val_dir = os.path.join(args.data_path, 'val')

    dataset_test, test_sampler = load_data(val_dir, args.cache_dataset, args.distributed)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    # initialize models
    adv_model = torchvision.models.__dict__[args.model_architecture]().to(device)

    checkpoint_dict = torch.load(checkpoint_path)
    adv_model.load_state_dict(checkpoint_dict['adv_model'])

    args.num_steps = num_steps
    print('*' * 60 + '\nAttack: %s Step PGD\n' % num_steps + '=' * 60)
    set_torch_seeds(0)

    lst_rob = []
    for i in range(num_random_init):
        print('*' * 30 + '\nRandom Init %s\n' % i + '*' * 30)
        nat_acc, rob_acc = eval_adv_test_whitebox(adv_model, device, test_loader, epsilon, args.num_steps, condition_on_correct)
        lst_rob.append(rob_acc)

        rob_arr = np.array(lst_rob)

    print('PGD-avg:', np.mean(rob_arr))
    print('PGD-best:', np.max(rob_arr))
    print('PGD-worst:', np.min(rob_arr))
    print('Clean-acc:', nat_acc)


if __name__ == '__main__':
    eval_checkpoint(
        args.checkpoint_path,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        num_random_init=args.random_init,
        condition_on_correct=args.condition_on_correct
    )
