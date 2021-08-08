import argparse
import json
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch import nn
import torchvision
import torchvision.transforms as transforms
from utilities.losses import cross_entropy, distillation, get_adv_images
from utilities import utils
from utilities import dist_utils
import os


parser = argparse.ArgumentParser(description="ACT for ImageNet")
# Model options
parser.add_argument("--exp_identifier", type=str, default="")
parser.add_argument("--nat_student_architecture", type=str, default="resnet50")
parser.add_argument("--adv_student_architecture", type=str, default="resnet50")
parser.add_argument("--device", default="cuda", help="device")
parser.add_argument("--workers", default=4, type=int)
# Training options
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--epoch_step", nargs="*", type=int, default=[30, 60, 90])
parser.add_argument("--lr_decay_ratio", default=0.1, type=float)
parser.add_argument('--lr-step-size', default=30, type=int)
parser.add_argument('--lr-gamma', default=0.1, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20])
# Data
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--cache-dataset", action="store_true", default=False)
parser.add_argument("--sync-bn", action="store_true", default=False)
parser.add_argument("--num_classes", type=int, default=1000)
# Training Params
parser.add_argument("--use_same_init", action="store_true", default=False)
parser.add_argument("--temperature", type=float, default=4)
parser.add_argument("--nat_alpha", type=float, default=0.9)
parser.add_argument("--adv_alpha", type=float, default=0.9)
# evaluation options
parser.add_argument("--save_freq", type=int, default=1)
# storage options
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--output_dir", type=str, default="experiments")
parser.add_argument("--checkpoint", type=str)
# distributed training parameters
parser.add_argument("--world-size", default=1, type=int)
parser.add_argument("--dist-url", default="env://")
# Adversarial training
parser.add_argument("--use_targeted_attack", action="store_true", default=False)
parser.add_argument("--adv_alpha_targeted", type=float, default=0.5)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--epsilon", type=float, default=2 / 255)
parser.add_argument("--perturb_steps", type=int, default=10)
parser.add_argument("--distance", default="l_inf", choices=["l_2", "l_inf"])
# Device options
parser.add_argument("--cuda", action="store_true")
params = parser.parse_args()


# ======================================================================================
# Helper Functions
# ======================================================================================
def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):

    # Data loading code
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # normalize,
                ]
            ),
        )
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            dist_utils.mkdir(os.path.dirname(cache_path))
            dist_utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            dist_utils.mkdir(os.path.dirname(cache_path))
            dist_utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = dist_utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        " * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    return metric_logger.acc1.global_avg


def train_mutual_learning(
    args,
    nat_s_net,
    adv_s_net,
    device,
    train_loader,
    nat_optimizer,
    adv_optimizer,
    epoch,
):

    nat_s_net.train()
    adv_s_net.train()

    print(nat_s_net.device_ids)

    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        "img/s", dist_utils.SmoothedValue(window_size=4, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "nat_ce", dist_utils.SmoothedValue(window_size=4, fmt="{value:2.4f}")
    )
    metric_logger.add_meter(
        "nat_kl", dist_utils.SmoothedValue(window_size=4, fmt="{value:2.4f}")
    )
    metric_logger.add_meter(
        "adv_ce", dist_utils.SmoothedValue(window_size=4, fmt="{value:2.4f}")
    )
    metric_logger.add_meter(
        "adv_kl", dist_utils.SmoothedValue(window_size=4, fmt="{value:2.4f}")
    )
    metric_logger.add_meter(
        "loss_nat", dist_utils.SmoothedValue(window_size=4, fmt="{global_avg:2.4f}")
    )
    metric_logger.add_meter(
        "loss_adv", dist_utils.SmoothedValue(window_size=4, fmt="{global_avg:2.4f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for data, target in metric_logger.log_every(train_loader, print_freq, header):

        start_time = time.time()
        data, target = data.to(device), target.to(device)

        # reset the gradients
        nat_optimizer.zero_grad()
        adv_optimizer.zero_grad()

        # get natural model output
        nat_out = nat_s_net(data)

        # generate adversarial examples for the adversarial model
        x_adv = get_adv_images(
            args, nat_out.detach(), adv_s_net, data, target
        )

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

        batch_size = data.shape[0]
        nat_acc1, nat_acc5 = dist_utils.accuracy(nat_out, target, topk=(1, 5))
        adv_acc1, adv_acc5 = dist_utils.accuracy(adv_out, target, topk=(1, 5))

        metric_logger.meters["lr"].update(nat_optimizer.param_groups[0]["lr"])
        metric_logger.meters["nat_acc1"].update(nat_acc1.item(), n=batch_size)
        metric_logger.meters["nat_acc5"].update(nat_acc5.item(), n=batch_size)
        metric_logger.meters["adv_acc1"].update(adv_acc1.item(), n=batch_size)
        metric_logger.meters["adv_acc5"].update(adv_acc5.item(), n=batch_size)
        metric_logger.meters["nat_ce"].update(l_ce_nat.item())
        metric_logger.meters["nat_kl"].update(l_kl_nat.item())
        metric_logger.meters["loss_nat"].update(loss_nat.item())
        metric_logger.meters["adv_ce"].update(l_ce_adv.item())
        metric_logger.meters["adv_kl"].update(l_kl_adv.item())
        metric_logger.meters["loss_adv"].update(loss_adv.item())
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


# ======================================================================================
# Training Function
# ======================================================================================
def solver(args):

    print(args.experiment_name)
    log_dir = os.path.join(args.experiment_name, "logs")
    model_dir = os.path.join(args.experiment_name, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args.cache_dataset, args.distributed
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Load Natural Model
    nat_s_net = torchvision.models.__dict__[args.nat_student_architecture]()
    nat_s_net = nat_s_net.to(args.device)

    # Load Adversarial Model
    adv_s_net = torchvision.models.__dict__[args.adv_student_architecture]()
    adv_s_net = adv_s_net.to(args.device)

    if args.use_same_init:
        adv_s_net.load_state_dict(nat_s_net.state_dict())

    # Initialize Optimizer
    nat_optimizer = SGD(
        nat_s_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    adv_optimizer = SGD(
        adv_s_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    lr_scheduler_nat = torch.optim.lr_scheduler.StepLR(nat_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler_adv = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    nat_s_net_without_ddp = nat_s_net
    adv_s_net_without_ddp = adv_s_net

    if args.distributed:
        nat_s_net = torch.nn.parallel.DistributedDataParallel(
            nat_s_net, device_ids=[args.gpu]
        )
        adv_s_net = torch.nn.parallel.DistributedDataParallel(
            adv_s_net, device_ids=[args.gpu]
        )
        nat_s_net_without_ddp = nat_s_net.module
        adv_s_net_without_ddp = adv_s_net.module

    checkpoint_epoch = 0
    if args.checkpoint:
        checkpoint_dict = torch.load(args.checkpoint)
        nat_s_net_without_ddp.load_state_dict(checkpoint_dict["nat_model"])
        adv_s_net_without_ddp.load_state_dict(checkpoint_dict["adv_model"])
        nat_optimizer.load_state_dict(checkpoint_dict["nat_optimizer"])
        adv_optimizer.load_state_dict(checkpoint_dict["adv_optimizer"])
        lr_scheduler_nat.load_state_dict(checkpoint_dict['lr_scheduler_nat'])
        lr_scheduler_adv.load_state_dict(checkpoint_dict['lr_scheduler_adv'])
        checkpoint_epoch = checkpoint_dict["epoch"]
        print("Successfully loaded checkpoints from %s epoch" % checkpoint_epoch)

    print("*" * 60 + "\nTraining Mode: %s\n" % args.mode + "*" * 60)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch < checkpoint_epoch:
            continue

        # training step
        train_mutual_learning(
            args,
            nat_s_net,
            adv_s_net,
            args.device,
            train_loader,
            nat_optimizer,
            adv_optimizer,
            epoch,
        )

        # adjust learning rate for SGD
        lr_scheduler_nat.step()
        lr_scheduler_adv.step()

        # utils.adjust_learning_rate(
        #     epoch, args.epoch_step, args.lr_decay_ratio, nat_optimizer
        # )
        # utils.adjust_learning_rate(
        #     epoch, args.epoch_step, args.lr_decay_ratio, adv_optimizer
        # )

        print("Natural Model:")
        nat_test_accuracy = evaluate(
            nat_s_net, criterion, test_loader, device=args.device
        )
        print("Adversarial Model:")
        adv_test_accuracy = evaluate(
            adv_s_net, criterion, test_loader, device=args.device
        )

        if dist_utils.is_main_process():
            if epoch % args.save_freq == 0:
                checkpoint_dict = {
                    "nat_model": nat_s_net_without_ddp.state_dict(),
                    "adv_model": adv_s_net_without_ddp.state_dict(),
                    "nat_optimizer": nat_optimizer.state_dict(),
                    "adv_optimizer": adv_optimizer.state_dict(),
                    "lr_scheduler_nat": lr_scheduler_nat.state_dict(),
                    "lr_scheduler_adv": lr_scheduler_adv.state_dict(),
                    "epoch": epoch,
                }
                dist_utils.save_on_master(
                    checkpoint_dict,
                    os.path.join(model_dir, "checkpoint-epoch{}.pt".format(epoch)),
                )

    if dist_utils.is_main_process():
        # save model
        checkpoint_dict = {
            "nat_model": nat_s_net.state_dict(),
            "adv_model": adv_s_net.state_dict(),
            "nat_optimizer": nat_optimizer.state_dict(),
            "adv_optimizer": adv_optimizer.state_dict(),
        }

        dist_utils.save_on_master(
            checkpoint_dict,
            os.path.join(model_dir, "final_checkpoint-epoch{}.pt".format(epoch)),
        )
        dist_utils.save_on_master(
            nat_s_net_without_ddp, os.path.join(model_dir, "nat_model.pt")
        )
        dist_utils.save_on_master(
            adv_s_net_without_ddp, os.path.join(model_dir, "adv_model.pt")
        )

    return nat_test_accuracy, adv_test_accuracy


def main(args):

    # effective epochs
    args.mode = "ACT"
    args.dataset = "Imagenet"

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.exp_identifier:
        base_name = "%s_%s_%s_nat_%s_adv_%s_%sepochs_%s_nat_alpha_%s_adv_alpha" % (
            args.exp_identifier,
            args.mode,
            args.nat_student_architecture,
            args.adv_student_architecture,
            args.dataset,
            args.epochs,
            args.nat_alpha,
            args.adv_alpha,
        )
    else:
        base_name = "%s_%s_nat_%s_adv_%s_%sepochs_%s_nat_alpha_%s_adv_alpha" % (
            args.mode,
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
    print(args)
    with open(args_path, "w") as f:
        f.write("arguments: " + json.dumps(z) + "\n")

    use_cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize distributed arguments
    dist_utils.init_distributed_mode(args)

    if len(args.seeds) > 1:

        lst_nat_test_accs = []
        lst_adv_test_accs = []

        for seed in args.seeds:
            print("\n\n----------- SEED {} -----------\n\n".format(seed))
            utils.set_torch_seeds(seed)
            torch.manual_seed(seed)

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
        print("\n\nFINAL TEST ACC RATE: {:02.2f}".format(test_accuracy))
        file_name = "final_test_acc_{:02.2f}".format(test_accuracy)
        with open(os.path.join(args.output_dir, base_name, file_name), "w") as f:
            f.write("NA")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
