import argparse
import gc
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchinfo import summary

import datasets
import models
import train
import transforms
import utils


def get_args() -> argparse.ArgumentParser:
    """
    get arguments of program :)
    :return: parser
    """
    parser = argparse.ArgumentParser(description="A Good Python Code for train your semantic segmentation models",
                                     add_help=True)
    parser.add_argument("--name", default="regseg_run11", type=str,
                        help=f"experiment name ")
    parser.add_argument("--dataset", default="cityscapes", type=str,
                        help=f"datasets name ")

    return parser


def main() -> None:
    """
    Hi I am Amirhosein Feiz I write this code to evaluate my semantic segmentation ideas :)
    """
    gc.collect()
    utils.setup_env()
    parser = get_args()
    args = parser.parse_args()

    if args.dataset == "cityscapes":
        yaml_file = "./configs/cityscapes.yaml"
    else:
        raise NotImplemented

    writer = utils.create_log_dir(name=args.name, parser=parser, evaluate=True)
    args = parser.parse_args()
    device, experiment = utils.add_yaml_2_args_and_save_configs_and_get_device(parser=parser, yaml_path=yaml_file,
                                                                               log_path=args.log)
    args = parser.parse_args()
    _, val_transforms = transforms.get_augs(args)
    dataset, dataset_classes = datasets.DATASETS[args.dataset]
    valid_ds = dataset(phase="val", root=args.DIR, transforms=val_transforms)

    valid_sampler = torch.utils.data.SequentialSampler(valid_ds)

    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=1, sampler=valid_sampler,
                                           num_workers=args.NUM_WORKER, drop_last=True,
                                           collate_fn=datasets.collate_fn, pin_memory=True)

    model = models.MODELS_COLLECTIONS[args.MODEL](args.NUM_CLASSES, quantization=args.QAT,
                                                  inference=False, grad_cam=True)

    model.grad_cam = False
    summary(model, (1, 3, args.VALID_SIZE[0], args.VALID_SIZE[1]),
            device=device, col_width=16, col_names=["output_size", "num_params", "mult_adds"], verbose=1)
    experiment.set_model_graph(model)

    print()

    optimizer, scaler, scheduler, model_ema = train.get_optimizer(model=model, num_iters=len(valid_dl),
                                                                  args=args, device=device)
    criterion = train.Criterion(args=args)

    model, optimizer, scaler, scheduler, start_epoch, best_acc, log_dict = utils.resume(model=model,
                                                                                        optimizer=optimizer,
                                                                                        scaler=scaler,
                                                                                        scheduler=scheduler,
                                                                                        model_ema=model_ema,
                                                                                        args=args,
                                                                                        evaluate=True)
    model.to(device)
    model.grad_cam = True
    print("Grad Cam")
    utils.grad_cam(model=model, optimizer=optimizer, dataloader=valid_dl, device=device, writer=writer,
                   experiment=experiment, clss=dataset_classes, args=args)

    model.grad_cam = False
    with torch.inference_mode():
        valid_acc, valid_loss = train.evaluate(model=model, epoch=1, dataloader=valid_dl,
                                               criterion=criterion, args=args, writer=writer,
                                               experiment=experiment, device=device, classes=dataset_classes,
                                               save_preds=True)
    print(f"Evaluation: Valid Loss is  {valid_loss} & Valid mIOU is {valid_acc}")


if __name__ == "__main__":
    main()
