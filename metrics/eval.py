import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pandas as pd

parser = argparse.ArgumentParser(description="PyTorch Test Set Evaluation")
parser.add_argument("--test-batch-size", type=int, default=200)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--model-path")
parser.add_argument("--dataset", default="CIFAR10")
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor()])


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
    print("Accuracy:", accuracy)
    print("loss:", loss)

    return loss, accuracy, correct


def eval_accuracy(
    lst_models, model_dir, output, dataset="CIFAR10", test_batch_size=200
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
        testset, batch_size=test_batch_size, shuffle=False, **kwargs
    )

    rob_analysis_dict = dict()
    rob_analysis_dict["method"] = []
    rob_analysis_dict["test_acc"] = []
    count = 0
    for model_basename, method in lst_models:

        print("=" * 60 + "\nModel Name: %s\n" % model_basename + "=" * 60)
        try:
            model = torch.load(
                os.path.join(model_dir, model_basename, "final_model.pt")
            ).to(device)
            model.eval()

            loss, accuracy, correct = eval(model, device, test_loader)

            rob_analysis_dict["method"].append(method)
            rob_analysis_dict["test_acc"].append(accuracy)

            print("Accuracy:", accuracy * 100)

        except Exception as e:
            print(e)
            pass
        count += 1
        df = pd.DataFrame(rob_analysis_dict)
        print(output + "_interim.csv")
        df.to_csv(output + "_interim.csv", index=False)

    os.remove(output + "_interim.csv")
    df = pd.DataFrame(rob_analysis_dict)
    df.to_csv(output + ".csv", index=False)


if __name__ == "__main__":

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

    # Load model
    model = torch.load(args.model_path).to(device)

    # evaluate on test set
    eval(model, device, test_loader)
