import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
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
from utilities.utils import set_torch_seeds

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
set_torch_seeds(0)

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor()])


# ======================================================================================
# Configurations
# ======================================================================================
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


def _pgd_whitebox(
    model,
    X,
    y,
    epsilon=0.031,
    num_steps=20,
    step_size=0.003,
    random=True,
    condition_on_correct=False,
):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    if condition_on_correct:
        select_idx = out.data.max(1)[1] == y.data
        X = X[select_idx]
        y = y[select_idx]

    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
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

    return err, err_pgd, attacks_conducted


def eval_adv_test_whitebox(
    model,
    device,
    test_loader,
    num_steps=20,
    epsilon=0.031,
    step_size=0.003,
    random=True,
    condition_on_correct=False,
):
    """
    evaluate model by white-box attack
    """
    model.eval()
    num_attacks_total = 0
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, num_attacks = _pgd_whitebox(
            model,
            X,
            y,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
            random=random,
            condition_on_correct=condition_on_correct,
        )
        robust_err_total += err_robust
        natural_err_total += err_natural
        num_attacks_total += num_attacks

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_attacks = num_attacks_total
    rob_acc = (total_attacks - successful_attacks) / total_attacks
    nat_acc = (total_attacks - nat_err) / total_attacks

    print("natural_accuracy: ", nat_acc)
    print("Successful Attacks: ", successful_attacks)
    print("Total Attacks:", total_attacks)
    print("robustness_accuracy:", rob_acc)

    return nat_err, successful_attacks, total_attacks, rob_acc


def _pgd_blackbox(
    model_target, model_source, X, y, epsilon=0.031, num_steps=20, step_size=0.003
):

    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural

    print("natural_err_total: ", natural_err_total)
    print("robust_err_total: ", robust_err_total)


def eval_pgd(
    lst_models,
    model_dir,
    output,
    lst_steps=[10, 15, 20],
    num_random_init=5,
    dataset="CIFAR10",
    condition_on_correct=True,
    epsilon=0.031,
    step_size=0.003,
):

    if dataset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform_test
        )
    elif dataset == "CIFAR100":
        testset = torchvision.datasets.CIFAR100(
            root="data", train=False, download=True, transform=transform_test
        )
    else:
        raise Exception("Invalid dataset")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, **kwargs
    )

    rob_analysis_dict = dict()
    rob_analysis_dict["method"] = []
    rob_analysis_dict["num_steps"] = []
    rob_analysis_dict["acc"] = []
    rob_analysis_dict["PGD-avg"] = []
    rob_analysis_dict["PGD-best"] = []
    rob_analysis_dict["PGD-worst"] = []
    count = 0

    for num_steps in lst_steps:

        for model_basename, method in lst_models:
            print("=" * 60 + "\nModel Name: %s\n" % model_basename + "=" * 60)
            try:
                model = torch.load(
                    os.path.join(model_dir, model_basename, "final_model.pt")
                ).to(device)
                model.eval()

                rob_analysis_dict["method"].append(method)
                print("*" * 60 + "\nAttack: %s Step PGD\n" % num_steps + "=" * 60)
                set_torch_seeds(0)

                lst_rob = []

                for i in range(num_random_init):
                    print("*" * 30 + "\nRandom Init %s\n" % i + "*" * 30)
                    nat_acc, _, _, rob_acc = eval_adv_test_whitebox(
                        model,
                        device,
                        test_loader,
                        num_steps,
                        epsilon,
                        step_size,
                        condition_on_correct,
                    )
                    lst_rob.append(rob_acc)

                rob_arr = np.array(lst_rob)
                rob_analysis_dict["acc"].append(nat_acc)
                rob_analysis_dict["num_steps"].append(num_steps)
                rob_analysis_dict["PGD-avg"].append(np.mean(rob_arr))
                rob_analysis_dict["PGD-best"].append(np.max(rob_arr))
                rob_analysis_dict["PGD-worst"].append(np.min(rob_arr))

                count += 1
                df = pd.DataFrame(rob_analysis_dict)
                df.to_csv(output + "_interim.csv", index=False)

            except Exception as e:
                print(e)
                pass

    df = pd.DataFrame(rob_analysis_dict)
    df.to_csv(output + ".csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch PGD Attack Evaluation")
    parser.add_argument("--test-batch-size", type=int, default=200)
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--epsilon", default=0.031)
    parser.add_argument("--num-steps", default=20)
    parser.add_argument("--step-size", default=0.003)
    parser.add_argument("--condition_on_correct", action="store_true", default=False)
    parser.add_argument("--random", default=True)
    parser.add_argument("--model-path")
    parser.add_argument("--source-model-path")
    parser.add_argument("--target-model-path")
    parser.add_argument("--black-box-attack", action="store_true", default=False)

    args = parser.parse_args()

    # Load Dataset
    if args.dataset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform_test
        )
    elif args.dataset == "CIFAR100":
        testset = torchvision.datasets.CIFAR100(
            root="data", train=False, download=True, transform=transform_test
        )
    else:
        raise Exception("Invalid dataset")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    if not args.black_box_attack:
        # white-box attack
        print("pgd white-box attack")
        model = torch.load(args.model_path).to(device)
        eval_adv_test_whitebox(
            model,
            device,
            test_loader,
            args.num_steps,
            args.epsilon,
            args.step_size,
            args.random,
            args.condition_on_correct,
        )

    else:
        # black-box attack
        print("pgd black-box attack")
        model_target = torch.load(args.target_model_path).to(device)
        model_source = torch.load(args.source_model_path).to(device)

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)
