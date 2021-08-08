
import torch
import torch.nn.functional as F


def print_decorated(string, char='='):
    boundary = char * 75
    print("\n" + boundary)
    print("%s" % string)
    print(boundary)


def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct


def adjust_learning_rate(epoch, epoch_steps, epoch_decay, optimizer):
    """decrease the learning rate"""
    
    if epoch in epoch_steps:
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = current_lr * epoch_decay
        print('=' * 60 + '\nChanging learning rate to %g\n' % (current_lr * epoch_decay) + '=' * 60)
