import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, Dict

import comet_ml
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
    comet_ml.init(
        project_name="torch-semantic-segmentation-models",
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.quantized.engine = "fbgemm"
    set_seed(0)


def add_yaml_2_args_and_save_configs_and_get_device(parser: ArgumentParser,
                                                    yaml_path: str, log_path: str)\
                                                    -> Tuple[torch.device, comet_ml.Experiment]:

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
    configs["device"] = device

    if os.path.isfile(os.path.join("./train_log/", "experiment.json")):
        with open(os.path.join("./train_log/", "experiment.json"), 'r') as openfile:
            keys_dict = json.load(openfile)
        if parser.parse_args().name in list(keys_dict.keys()):
            print("Use Existing Experiment")
            experiment = comet_ml.ExistingExperiment(previous_experiment=keys_dict[parser.parse_args().name],
                                                     log_code=True)
        else:
            experiment = comet_ml.Experiment(log_code=True)
            experiment.set_name(parser.parse_args().name)
            experiment.log_parameters(configs)
            keys_dict[parser.parse_args().name] = experiment.get_key()
            with open(os.path.join("./train_log/", "experiment.json"), "w") as outfile:
                json.dump(keys_dict, outfile)

    else:
        with open(os.path.join("./train_log/", "experiment.json"), "w") as outfile:
            json.dump({}, outfile)
        with open(os.path.join("./train_log/", "experiment.json"), 'r') as openfile:
            keys_dict = json.load(openfile)
        experiment = comet_ml.Experiment(log_code=True)
        experiment.set_name(parser.parse_args().name)
        experiment.log_parameters(configs)
        keys_dict[parser.parse_args().name] = experiment.get_key()
        with open(os.path.join("./train_log/", "experiment.json"), "w") as outfile:
            json.dump(keys_dict, outfile)

    print("Load Configs")
    table = tabulate.tabulate(configs_list, headers=["name", "config"])
    print(table)
    with open(os.path.join(log_path, "configs.txt"), "w") as file:
        file.write(table)
    return device, experiment


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
    :return: tensorboard writer, number of training experiment
    """
    os.makedirs(f'./train_log/{name}', exist_ok=True)
    os.makedirs(f'./train_log/{name}/checkpoint', exist_ok=True)
    os.makedirs(f'./train_log/{name}/predicts', exist_ok=True)
    writer = tb.SummaryWriter(f'./train_log/{name}/tensorboard')
    parser.add_argument(f'--log', default=f"./train_log/{name}/", help=f'log path')
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
    :return: model, optimizer, scaler, scheduler, start_epoch, best_acc, log_dict
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
            log_dict = loaded["log_dict"]
            loaded = torch.load(best_ckpt)
            best_acc = loaded["acc"]
            if model_ema is not None:
                model_ema.load_state_dict(loaded["ema"])
                model_ema.module.load_state_dict(loaded["model_ema"])
            print(f"Load all parameters from last checkpoint :)")
            print(f"Train start from epoch {start_epoch + 1} epoch :)")
            print(f"Best Accuracy is {best_acc} :)")
            print()
            return model, optimizer, scaler, scheduler, start_epoch, best_acc, log_dict
        except Exception as error:
            print(f"Something is wrong! :( ")
            print(error)
            exit()


def save(model: torch.nn.Module, acc: float, best_acc: float,
         scaler: torch.cuda.amp.GradScaler, optimizer: torch.optim.Optimizer, model_ema: Optional[torch.nn.Module],
         scheduler: torch.optim.lr_scheduler.LRScheduler, qat_model: Optional[torch.nn.Module], epoch: int,
         args: Namespace, log_dict: Dict) -> float:
    """
    save model and others
    :param model: model
    :param acc: mIOU
    :param best_acc: best archived mIOU
    :param scaler: gradient scaler (float 16)
    :param optimizer: optimizer
    :param model_ema: model_ema if enabled
    :param scheduler: learning rate scheduler
    :param qat_model: quantization aware training model
    :param epoch: last epoch
    :param args: arguments
    :param log_dict: log dictionary
    :return: best mIOU
    """
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'scaler': scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'log_dict': log_dict
        }
        if model_ema is not None:
            state.update({
                "ema": model_ema.state_dict(),
                "model_ema": model_ema.module.state_dict()
            })

        torch.save(state, os.path.join(args.log, f"checkpoint/best_{args.name}.pth"))
        torch.jit.save(torch.jit.script(model),
                       os.path.join(args.log, f"checkpoint/best_scripted_{args.name}.pt"))
        if qat_model is not None:
            torch.save(qat_model.state_dict(), os.path.join(args.log, f"checkpoint/best_{args.name}.pth"))
            torch.jit.save(torch.jit.script(qat_model),
                           os.path.join(args.log, f"checkpoint/last_qat_scripted_{args.name}.pt"))
        best_acc = acc
    return best_acc
