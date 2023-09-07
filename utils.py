import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, Dict, OrderedDict

import cv2
import tqdm.autonotebook as tqdm

import comet_ml
import numpy as np
import tabulate
import tensorboardX as tb
import torch
import yaml

from datasets import unnormalize_image


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
    set_seed(0)


def add_yaml_2_args_and_save_configs_and_get_device(parser: ArgumentParser,
                                                    yaml_path: str, log_path: str) \
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


def create_log_dir(name: str, parser, evaluate: bool = False) -> tb.SummaryWriter:
    """
    create log directory :)
    :param evaluate: use in train or evaluate code
    :param name: experiment name
    :param parser: parser
    :return: tensorboard writer, number of training experiment
    """
    if not evaluate:
        os.makedirs(f'./train_log/{name}', exist_ok=True)
        os.makedirs(f'./train_log/{name}/checkpoint', exist_ok=True)
        os.makedirs(f'./train_log/{name}/predicts', exist_ok=True)
        os.makedirs(f'./train_log/{name}/grad_cam', exist_ok=True)
    writer = tb.SummaryWriter(f'./train_log/{name}/tensorboard')
    parser.add_argument(f'--log', default=f"./train_log/{name}/", help=f'log path')
    return writer


def resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
           scheduler: torch.optim.lr_scheduler.LRScheduler, model_ema: torch.nn.Module, args: Namespace,
           evaluate: bool = False):
    """
    load parameters to continue training :)
    :param evaluate: load for train or evaluate
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
    if not evaluate:
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
    else:
        if os.path.isfile(best_ckpt):
            try:
                loaded = torch.load(best_ckpt)
                model.load_state_dict(loaded["model"])
                optimizer.load_state_dict(loaded["optimizer"])
                best_acc = loaded["acc"]
                print(f"Load all parameters from last checkpoint :)")
                print(f"Best Accuracy is {best_acc} :)")
                print()
                return model, optimizer, None, None, None, best_acc, None
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

        if qat_model is not None:
            torch.save(qat_model.state_dict(), os.path.join(args.log, f"checkpoint/best_qat_{args.name}.pth"))
        best_acc = acc
    return best_acc


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')


def run_benchmark(model, img_loader):
    elapsed = 0
    model.eval()
    num_batches = 100
    loop = tqdm.tqdm(img_loader, total=num_batches)
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(loop):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end - start)
        else:
            break
        loop.set_description(f"Benchmark: Batch {i}, Time {end - start}")
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed / num_images * 1000))
    return elapsed


def grad_cam(model, optimizer, dataloader, device, writer, experiment, clss, args):
    model.eval()
    classes = []
    for v in clss.values():
        classes.append(v["name"])
    loop = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs = inputs.to(device)
        if (batch_idx + 1) % 25 == 0:
            for idx, cls_name in enumerate(classes):
                optimizer.zero_grad()
                outputs = model(inputs)
                if type(outputs) is OrderedDict:
                    outputs = outputs["out"]
                if idx in torch.unique(targets):
                    cls_mask = torch.where(outputs[0].argmax(0) == idx, 1.0, 0.0)
                    grad_loss = (outputs[0][idx] * cls_mask).sum()
                    with torch.no_grad():
                        activations = model.get_activations(inputs)
                    grad_loss.backward(retain_graph=True)
                    gradients = model.get_activations_gradient()
                    for index, (key, value) in enumerate(activations.items()):
                        pooled_gradients = torch.mean(gradients[index],
                                                      dim=[0, 2, 3]).reshape(1, -1, 1, 1).detach().cpu()
                        val = value.detach().cpu()
                        val *= pooled_gradients
                        heatmap = torch.mean(val, dim=1).squeeze().detach().cpu().numpy()
                        heatmap = np.maximum(heatmap, 0)
                        heatmap = heatmap / (np.max(heatmap) + 1e-8)
                        heatmap = (heatmap * 255).astype(np.uint8)

                        image = unnormalize_image(inputs[0].detach().cpu(),
                                                  mean=args.MEAN, std=args.STD).permute(1, 2,
                                                                                        0).numpy()
                        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        superimposed_img = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

                        writer.add_image(f"step{batch_idx + 1}_class_{args.name}_grad{key}",
                                         torch.tensor(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB
                                                                   )).permute(2, 0, 1))
                        experiment.log_image(superimposed_img, f"step{batch_idx + 1}_class_{cls_name}_grad{key}")
                        cv2.imwrite(
                            f'./train_log/{args.name}/grad_cam/'
                            f'step{batch_idx + 1}_class_{cls_name}_grad{key}.jpg',
                            superimposed_img)
