import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utilities.losses import get_adv_images, cross_entropy, distillation
from utilities import utils
from models.selector import select_model
from metrics.pgd_attack import eval_adv_test_whitebox

parser = argparse.ArgumentParser(description="Knowledge Distillation Methods")
# Model options
parser.add_argument("--exp_identifier", type=str, default="")
parser.add_argument("--nat_student_architecture", type=str, default="ResNet18")
parser.add_argument("--adv_student_architecture", type=str, default="ResNet18")
parser.add_argument("--dataset", default="CIFAR10", type=str)
# Training options
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--epoch_step", nargs="*", type=int, default=[60, 120, 160])
parser.add_argument("--lr_decay_ratio", type=float, default=0.2)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--seeds", nargs="*", type=int, default=[0, 10])
# Training Params
parser.add_argument("--use_same_init", action="store_true", default=False)
parser.add_argument("--temperature", type=float, default=4)
parser.add_argument("--nat_alpha", type=float, default=0.9)
parser.add_argument("--adv_alpha", type=float, default=0.9)
# Device options
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--device_id", default=0, type=int)
# evaluation options
parser.add_argument("--save_freq", default=1, type=int)
parser.add_argument("--train_eval_freq", default=10, type=int)
parser.add_argument("--test_eval_freq", default=10, type=int)
parser.add_argument("--pgd_eval_freq", default=4, type=int)
parser.add_argument("--monitor_robustness", default=180, type=int)
# storage options
parser.add_argument("--dataroot", default="data", type=str)
parser.add_argument("--output_dir", default="experiments", type=str)
parser.add_argument("--checkpoint", default="", type=str)
# Adversarial training
parser.add_argument("--epsilon", type=float, default=0.031)
parser.add_argument("--perturb_steps", type=int, default=10)
parser.add_argument("--step_size", type=float, default=0.007)


# =============================================================================
# Helper Functions
# =============================================================================
def train_mutual_learning(
    args,
    nat_s_net,
    adv_s_net,
    train_loader,
    nat_optimizer,
    adv_optimizer,
    epoch,
    writer,
):

    nat_s_net.train()
    adv_s_net.train()

    nat_train_loss = 0
    nat_correct = 0

    adv_train_loss = 0
    adv_correct = 0

    total = 0

    num_batches = len(train_loader)

    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), desc="batch training", total=num_batches
    ):

        data, target = data.to(args.device), target.to(args.device)

        # reset the gradients
        nat_optimizer.zero_grad()
        adv_optimizer.zero_grad()

        iteration = (epoch * num_batches) + batch_idx

        # get natural model output
        nat_out = nat_s_net(data)

        # generate adversarial examples for the adversarial model
        x_adv = get_adv_images(args, nat_out.detach(), adv_s_net, data, target)

        # get adversarial model output
        adv_out = adv_s_net(x_adv)

        # losses for the natural model
        l_ce_nat = cross_entropy(nat_out, target)
        l_kl_nat = distillation(nat_out, adv_out.detach(), args.temperature)
        loss_nat = (1.0 - args.nat_alpha) * l_ce_nat + args.nat_alpha * l_kl_nat

        # losses for the adversarial model
        l_ce_adv = cross_entropy(adv_out, target)
        l_kl_adv = distillation(adv_out, nat_out.detach(), args.temperature)
        loss_adv = (1.0 - args.adv_alpha) * l_ce_adv + args.adv_alpha * l_kl_adv

        # Update models
        loss_nat.backward()
        nat_optimizer.step()

        loss_adv.backward()
        adv_optimizer.step()

        # Natural Student Logs
        writer.add_scalar("nat_student/l_ce", l_ce_nat.item(), iteration)
        writer.add_scalar("nat_student/l_kl", l_kl_nat.item(), iteration)
        writer.add_scalar("nat_student/loss", loss_nat.item(), iteration)

        # Adversarial Student Logs
        writer.add_scalar("adv_student/l_ce", l_ce_adv.item(), iteration)
        writer.add_scalar("adv_student/l_kl", l_kl_adv.item(), iteration)
        writer.add_scalar("adv_student/loss", loss_adv.item(), iteration)

        nat_train_loss += loss_nat.data.item()
        adv_train_loss += loss_adv.data.item()

        _, predicted_nat = torch.max(nat_out.data, 1)
        _, predicted_adv = torch.max(adv_out.data, 1)

        nat_correct += predicted_nat.eq(target.data).cpu().float().sum()
        adv_correct += predicted_adv.eq(target.data).cpu().float().sum()

        total += target.size(0)

    nat_train_loss /= num_batches + 1
    nat_acc = 100.0 * nat_correct / total

    adv_train_loss /= num_batches + 1
    adv_acc = 100.0 * adv_correct / total

    print(
        "Natural Model Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (nat_train_loss, nat_acc, nat_correct, total)
    )
    print(
        "Adversarial Model Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (adv_train_loss, adv_acc, adv_correct, total)
    )


# =============================================================================
# Training Function
# =============================================================================
def solver(args):

    print(args.experiment_name)
    log_dir = os.path.join(args.experiment_name, "logs")
    model_dir = os.path.join(args.experiment_name, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M"))
    writer = SummaryWriter(log_path)

    use_cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(args.device_id)
        cudnn.benchmark = True

    # load dataset
    if args.dataset == "MNIST":
        transform_train = transforms.Compose([transforms.ToTensor()])

    else:
        transform_train = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
            ]
        )

    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.dataset == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10
        args.num_classes = 10
    elif args.dataset == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100
        args.num_classes = 100
    elif args.dataset == "MNIST":
        dataset = torchvision.datasets.MNIST
        args.num_classes = 10
    else:
        raise ValueError("Invalided dataset argument")

    print("==> Preparing data..")

    trainset = dataset(
        root=args.dataroot, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    testset = dataset(
        root=args.dataroot, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print("Dataset Loaded")

    # Load Natural Model
    nat_s_net = select_model(args.nat_student_architecture, args.num_classes).to(
        args.device
    )

    # Load Adversarial Model
    adv_s_net = select_model(args.adv_student_architecture, args.num_classes).to(
        args.device
    )

    if args.use_same_init:
        adv_s_net.load_state_dict(nat_s_net.state_dict())

    # Initialize Optimizer
    nat_optimizer = SGD(
        nat_s_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    adv_optimizer = SGD(
        adv_s_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    checkpoint_epoch = 0
    if args.checkpoint:

        checkpoint_dict = torch.load(args.checkpoint)

        nat_s_net.load_state_dict(checkpoint_dict["nat_model"])
        adv_s_net.load_state_dict(checkpoint_dict["adv_model"])

        nat_optimizer.load_state_dict(checkpoint_dict["nat_optimizer"])
        adv_optimizer.load_state_dict(checkpoint_dict["adv_optimizer"])

        checkpoint_epoch = checkpoint_dict["epoch"]

        print("Successfully loaded checkpoints from %s epoch" % checkpoint_epoch)

    for epoch in tqdm(range(1, args.epochs + 1), desc="training epochs"):

        # adjust learning rate for SGD
        utils.adjust_learning_rate(
            epoch, args.epoch_step, args.lr_decay_ratio, nat_optimizer
        )
        utils.adjust_learning_rate(
            epoch, args.epoch_step, args.lr_decay_ratio, adv_optimizer
        )

        if epoch < checkpoint_epoch:
            continue

        # training step
        train_mutual_learning(
            args,
            nat_s_net,
            adv_s_net,
            train_loader,
            nat_optimizer,
            adv_optimizer,
            epoch,
            writer,
        )

        # evaluation on natural examples
        if epoch % args.train_eval_freq == 0:
            train_loss, train_accuracy, correct = utils.eval(
                nat_s_net, args.device, train_loader
            )
            utils.print_decorated(
                "Natural Student | Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                    train_loss, correct, len(train_loader.dataset), train_accuracy * 100
                )
            )

            train_loss, train_accuracy, correct = utils.eval(
                adv_s_net, args.device, train_loader
            )
            utils.print_decorated(
                "Adversarial Student | Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                    train_loss, correct, len(train_loader.dataset), train_accuracy * 100
                )
            )

            writer.add_scalar("nat_student/train_loss", train_loss, epoch)
            writer.add_scalar("nat_student/train_accuracy", train_accuracy, epoch)
            writer.add_scalar("adv_student/train_loss", train_loss, epoch)
            writer.add_scalar("adv_student/train_accuracy", train_accuracy, epoch)

        if epoch % args.test_eval_freq == 0:
            test_loss, test_accuracy, correct = utils.eval(
                nat_s_net, args.device, test_loader
            )
            utils.print_decorated(
                "Natural Student | Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                    test_loss, correct, len(test_loader.dataset), test_accuracy * 100
                )
            )

            test_loss, test_accuracy, correct = utils.eval(
                adv_s_net, args.device, test_loader
            )
            utils.print_decorated(
                "Adversarial Student | Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                    test_loss, correct, len(test_loader.dataset), test_accuracy * 100
                )
            )
            writer.add_scalar("nat_student/test_loss", test_loss, epoch)
            writer.add_scalar("nat_student/test_accuracy", test_accuracy, epoch)
            writer.add_scalar("adv_student/test_loss", test_loss, epoch)
            writer.add_scalar("adv_student/test_accuracy", test_accuracy, epoch)

        if epoch >= args.monitor_robustness:

            if epoch % args.pgd_eval_freq == 0:
                nat_err, successful_attacks, total_attacks, rob_acc = eval_adv_test_whitebox(
                    nat_s_net,
                    args.device,
                    test_loader,
                    20,
                    args.epsilon,
                    args.step_size,
                )
                utils.print_decorated(
                    "Natural Student | PGD Robustness: {:.4f}%, Successful Attacks: {}/{}".format(
                        rob_acc * 100, successful_attacks, total_attacks
                    )
                )

                nat_err, successful_attacks, total_attacks, rob_acc = eval_adv_test_whitebox(
                    adv_s_net,
                    args.device,
                    test_loader,
                    20,
                    args.epsilon,
                    args.step_size,
                )
                utils.print_decorated(
                    "Adversarial Student | PGD Robustness: {:.4f}%, Successful Attacks: {}/{}".format(
                        rob_acc * 100, successful_attacks, total_attacks
                    )
                )
                writer.add_scalar("nat_student/pgd_robustness", rob_acc, epoch)
                writer.add_scalar("adv_student/pgd_robustness", rob_acc, epoch)

        if epoch % args.save_freq == 0:

            checkpoint_dict = {
                "nat_model": nat_s_net.state_dict(),
                "adv_model": adv_s_net.state_dict(),
                "nat_optimizer": nat_optimizer.state_dict(),
                "adv_optimizer": adv_optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(
                checkpoint_dict,
                os.path.join(model_dir, "checkpoint-epoch{}.pt".format(epoch)),
            )

    # get final test accuracy
    nat_test_loss, nat_test_accuracy, nat_correct = utils.eval(
        nat_s_net, args.device, test_loader
    )
    adv_test_loss, adv_test_accuracy, adv_correct = utils.eval(
        adv_s_net, args.device, test_loader
    )
    writer.close()

    # save model
    checkpoint_dict = {
        "nat_model": nat_s_net.state_dict(),
        "adv_model": adv_s_net.state_dict(),
        "nat_optimizer": nat_optimizer.state_dict(),
        "adv_optimizer": adv_optimizer.state_dict(),
    }

    torch.save(
        checkpoint_dict,
        os.path.join(model_dir, "final_checkpoint-epoch{}.pt".format(epoch)),
    )

    torch.save(nat_s_net, os.path.join(model_dir, "nat_model.pt"))
    torch.save(adv_s_net, os.path.join(model_dir, "adv_model.pt"))

    return nat_test_accuracy, adv_test_accuracy


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    prefix = ""
    if args.exp_identifier:
        prefix = "%s_" % args.exp_identifier

    base_name = "%sact_%s_nat_%s_adv_%s_%sepochs_%s_nat_alpha_%s_adv_alpha" % (
        prefix,
        args.nat_student_architecture,
        args.adv_student_architecture,
        args.dataset,
        args.epochs,
        args.nat_alpha,
        args.adv_alpha,
    )
    base_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # save training arguments
    args_path = os.path.join(base_dir, "args.txt")
    z = vars(args).copy()
    with open(args_path, "w") as f:
        f.write("arguments: " + json.dumps(z) + "\n")

    if len(args.seeds) > 1:

        lst_nat_test_accs = []
        lst_adv_test_accs = []

        for seed in args.seeds:

            print("\n\n----------- SEED {} -----------\n\n".format(seed))
            utils.set_torch_seeds(seed)

            args.experiment_name = os.path.join(
                args.output_dir, base_name, base_name + "_seed" + str(seed)
            )
            txt_path = args.experiment_name + ".txt"

            # check if the seed has been trained
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    next(f)
                    nat_test_accuracy, adv_test_accuracy = f.readline().strip().split()
                    nat_test_accuracy, adv_test_accuracy = (
                        float(nat_test_accuracy),
                        float(adv_test_accuracy),
                    )

                print(
                    "Seed %s already trained with %s test accuracy and %s test loss"
                    % (seed, nat_test_accuracy, adv_test_accuracy)
                )
            else:
                nat_test_accuracy, adv_test_accuracy = solver(args)

            lst_nat_test_accs.append(nat_test_accuracy)
            lst_adv_test_accs.append(adv_test_accuracy)

            with open(args.experiment_name + ".txt", "w+") as f:
                f.write("nat_test_acc\tadv_test_acc\n")
                f.write("%g\t%g\n" % (nat_test_accuracy, adv_test_accuracy))

        file_name = "performance.txt"

        print(len(args.seeds))

        with open(os.path.join(args.output_dir, base_name, file_name), "w+") as f:
            f.write("seed\ttest_acc\ttest_loss\n")
            for i in range(len(args.seeds)):
                f.write(
                    "%d\t%g\t%g\n"
                    % (args.seeds[i], lst_nat_test_accs[i], lst_adv_test_accs[i])
                )

    else:
        utils.set_torch_seeds(args.seeds[0])
        args.experiment_name = os.path.join(
            args.output_dir, base_name, base_name + "_seed" + str(args.seeds[0])
        )
        test_loss, test_accuracy = solver(args)
        # test_loss, test_accuracy = 0, 1
        print("\n\nFINAL TEST ACC RATE: {:02.2f}".format(test_accuracy))
        file_name = "final_test_acc_{:02.2f}".format(test_accuracy)
        with open(os.path.join(args.output_dir, base_name, file_name), "w") as f:
            f.write("NA")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
