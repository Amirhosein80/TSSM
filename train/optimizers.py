import torch
import torch.cuda.amp as amp
import torch.optim as optim

from argparse import Namespace
from typing import Tuple

from train.model_ema import ModelEma


def get_optimizer(model: torch.nn.Module, num_iters: int, args: Namespace, device: torch.device) \
        -> Tuple[optim.Optimizer, amp.GradScaler, optim.lr_scheduler.LRScheduler, ModelEma]:
    """
    load optimizer + scaler + scheduler + model ema for model :)
    :param model: model
    :param num_iters: number of iteration pre epoch
    :param args: arguments
    :param device: device
    :return:
    """
    # select optimizer
    if hasattr(model, "get_params"):
        params = model.get_params(lr=args.LR,
                                  weight_decay=args.WEIGHT_DECAY if not args.OVERFIT_TEST else 0)
    else:
        params = model.parameters()

    if args.OPTIMIZER == "ADAMW":
        optimizer = optim.AdamW(params, lr=args.LR,
                                weight_decay=args.WEIGHT_DECAY if not args.OVERFIT_TEST else 0,
                                betas=tuple(args.ADAMW_BETAS))

    elif args.OPTIMIZER == "SGD":
        optimizer = optim.SGD(params, lr=args.LR,
                              weight_decay=args.WEIGHT_DECAY if not args.OVERFIT_TEST else 0,
                              momentum=args.SGD_MOMENTUM)

    else:
        raise NotImplemented

    # add AMP
    scaler = amp.GradScaler(enabled=not args.QAT)

    # add scheduler
    total_iters = num_iters * (args.EPOCHS - args.WARMUP_EPOCHS)
    warm_iters = num_iters * args.WARMUP_EPOCHS

    if args.SCHEDULER_METHOD == "POLY":
        main_lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters,
                                                            power=args.POLY_POWER)

    elif args.SCHEDULER_METHOD == "COS":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-5)

    elif args.SCHEDULER_METHOD == "CONSTANT":
        main_lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, total_iters=total_iters, factor=1.0)

    else:
        raise NotImplemented

    # set warmup scheduler if you use
    if args.WARMUP_EPOCHS > 0 and not args.OVERFIT_TEST and not args.QAT:
        warm_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.WARMUP_FACTOR, total_iters=warm_iters)

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_lr_scheduler, main_lr_scheduler], milestones=[warm_iters])

    else:
        scheduler = main_lr_scheduler

    # set EMA if you use
    if args.USE_EMA:
        model_ema = ModelEma(model, decay=args.EMA_DECAY, device=device)
    else:
        model_ema = None

    return optimizer, scaler, scheduler, model_ema
