import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import tabulate
import tensorboardX as tb
import torch
import yaml


def set_seed(seed: int) -> None:
    """
    set random seed for modules :)
    :param seed: random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def setup_env() -> None:
    """
    setup backend defaults :)
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.quantized.engine = "x86"
    set_seed(0)


def add_yaml_2_args_and_save_configs_and_get_device(parser: ArgumentParser,
                                                    yaml_path: str, log_path: str) -> torch.device:
    """
    load configs from yaml file and add it to args also get device :)
    :param parser: parser
    :param yaml_path: yaml file path
    :param log_path: log path to save train configs
    :return: device
    """
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs_list = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for key, value in configs.items():
        parser.add_argument(f'--{key}', default=value, help=f'{key} value')
        configs_list.append([key, value])
    configs_list.append(["device", device])
    print("Load Configs")
    table = tabulate.tabulate(configs_list, headers=["name", "config"])
    print(table)
    with open(os.path.join(log_path, "configs.txt"), "w") as file:
        file.write(table)
    return device


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    get learning rate from optimizer :)
    :param optimizer: training optimizer
    :return: learning rate
    """
    return optimizer.param_groups[-1]['lr']


def count_parameters(model: torch.nn.Module) -> float:
    """
    count number of trainable parameters per million :)
    :param model: model
    :return: number of trainable parameters per million
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


def create_log_dir(name: str, parser) -> tb.SummaryWriter:
    """
    create log directory :)
    :param name: experiment name
    :param parser: parser
    :return:
    """
    os.makedirs(f'./train_log/{name}', exist_ok=True)
    num_train = len(os.listdir(f'./train_log/{name}')) + 1
    os.makedirs(f'./train_log/{name}/{num_train}/checkpoint', exist_ok=True)
    os.makedirs(f'./train_log/{name}/{num_train}/predicts', exist_ok=True)
    writer = tb.SummaryWriter(f'./train_log/{name}/{num_train}/tensorboard')
    parser.add_argument(f'--log', default=f"./train_log/{name}/{num_train}/", help=f'log path')
    return writer


def resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
           scheduler: torch.optim.lr_scheduler.LRScheduler, model_ema: torch.nn.Module, args: Namespace):
    """
    load parameters to continue training :)
    :param model: model
    :param optimizer: model optimizer
    :param scaler:  scaler (float 16)
    :param scheduler: learning rate scheduler
    :param model_ema: model ema
    :param args: arguments
    :return: model + optimizer + scaler + scheduler + start_epoch + best_acc
    """
    best_ckpt = os.path.join(args.log, f"checkpoint/best_{args.name}.pth")
    final_ckpt = os.path.join(args.log, f"checkpoint/final_{args.name}.pth")
    if os.path.isfile(best_ckpt) and os.path.isfile(final_ckpt) and not args.OVERFIT_TEST:
        try:
            loaded = torch.load(final_ckpt)
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            scaler.load_state_dict(loaded["scaler"])
            scheduler.load_state_dict(loaded["scheduler"])
            start_epoch = loaded["epoch"]
            loaded = torch.load(best_ckpt)
            best_acc = loaded["acc"]
            if model_ema is not None:
                model_ema.load_state_dict(loaded["ema"])
                model_ema.module.load_state_dict(loaded["model_ema"])
            print(f"Load all parameters from last checkpoint :)")
            print(f"Train start from epoch {start_epoch + 1} epoch :)")
            print(f"Best Accuracy is {best_acc} :)")
            print()
            return model, optimizer, scaler, scheduler, start_epoch, best_acc
        except:
            print(f"Something is wrong! We train from Scratch :( ")


def save(model, acc, best_acc, scaler, optimizer, scheduler, model_ema, epoch, args):
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'scaler': scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if model_ema is not None:
            state.update({
                "ema": model_ema.state_dict(),
                "model_ema": model_ema.module.state_dict()
            })

        best_ckpt = os.path.join(args.log, f"checkpoint/best_{args.name}.pth")
        torch.save(state, best_ckpt)

        best_acc = acc
        return best_acc
