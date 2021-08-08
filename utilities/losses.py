import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def cross_entropy(y, labels):
    l_ce = F.cross_entropy(y, labels)
    return l_ce


def distillation(student_scores, teacher_scores, T):

    p = F.log_softmax(student_scores / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / student_scores.shape[0]

    return l_kl


def get_adv_images(args, nat_model_out, adv_model, x_natural, y):

    # define KL-loss
    criterion = nn.KLDivLoss(size_average=False)

    adv_model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if args.use_targeted_attack:
        # generate the random targeted classes
        y_targeted = torch.zeros(y.shape, device=y.device).uniform_(0, args.num_classes).floor_()
        y_targeted = y_targeted.type(y.dtype)
        y_targeted = torch.fmod(y_targeted + y, args.num_classes)
    else:
        y_targeted = y

    for _ in range(args.perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():

            adv_model_out = adv_model(x_adv)
            ce_loss = cross_entropy(adv_model_out, y_targeted)
            kl_loss = criterion(F.log_softmax(adv_model_out, dim=1), F.softmax(nat_model_out, dim=1))

            if args.use_targeted_attack:
                # targeted attack (minimize CE and maximize KL)
                loss = -(1-args.adv_alpha_targeted) * ce_loss + args.adv_alpha_targeted * kl_loss
            else:
                loss = (1-args.adv_alpha) * ce_loss + args.adv_alpha * kl_loss

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
        x_adv = torch.min(
            torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon
        )
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    adv_model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    return x_adv
