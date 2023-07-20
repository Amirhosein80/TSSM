import os
from argparse import Namespace
from collections import OrderedDict
from typing import Tuple, Dict

import tensorboardX as tb
import torch
import tqdm
import comet_ml
from PIL import Image
from torchvision.utils import draw_segmentation_masks

from datasets.utils import unnormalize_image
from train.metrics import AverageMeter, ConfusionMatrix
from utils import set_seed


def train_one_epoch(model: torch.nn.Module, epoch: int, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, criterion: torch.nn.Module,
                    model_ema: torch.nn.Module, scheduler: torch.optim.lr_scheduler.LRScheduler,
                    args: Namespace, device: torch.device) -> Tuple[float, float]:
    """
    train model for one epoch :)
    :param model: model
    :param epoch: epoch
    :param dataloader: train data loader
    :param optimizer: train optimizer
    :param scaler: train scaler
    :param criterion: train loss function
    :param model_ema: model ema
    :param scheduler: learning rate scheduler
    :param args: arguments
    :param device: device
    :return: mIOU, loss value
    """
    model.train()
    loss_total = AverageMeter()
    semantic_total = AverageMeter()
    aux_total = AverageMeter()
    edge_total = AverageMeter()

    metric = ConfusionMatrix(num_classes=args.NUM_CLASSES)
    set_seed(epoch - 1)
    loop = tqdm.tqdm(dataloader, total=len(dataloader))

    # one epoch loop
    for batch_idx, (inputs, targets) in enumerate(loop):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):

            # forward step & calc loss & metric
            outputs = model(inputs)
            semantic_loss, semantic_aux, semantic_edge = criterion(outputs, targets)
            loss = semantic_loss + (0.4 * semantic_aux) + semantic_edge
            if type(outputs) is OrderedDict:
                outputs = outputs["out"]

        # backward step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)

        optimizer.zero_grad()
        scheduler.step()

        # calculate average of loss & metric
        metric.update(targets=targets, outputs=outputs.argmax(dim=1))
        loss_total.update(loss)
        semantic_total.update(semantic_loss)
        aux_total.update(semantic_aux)
        edge_total.update(semantic_edge)

        # print details
        loop.set_description(f"Train ====>> Epoch:{epoch}    Loss:{loss_total.avg:.4}"
                             f"    Out Loss: {semantic_total.avg:.4}    Aux Loss: {0.4 * aux_total.avg:.4}"
                             f"    Edge Loss: {edge_total.avg:.4}")

    torch.cuda.empty_cache()
    miou = metric.calculate()

    # save current model
    state = {
        'model': model.state_dict(),
        'acc': miou,
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
    path = os.path.join(args.log, f"checkpoint/final_{args.name}.pth")
    torch.save(state, path)

    return miou, loss_total.avg.item()


def evaluate(model: torch.nn.Module, epoch: int, dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, args: Namespace, writer: tb.SummaryWriter, experiment: comet_ml.Experiment,
             device: torch.device, classes: Dict, save_preds: bool = True) -> Tuple[float, float]:
    """
    evaluate model for one epoch :)
    :param model: model
    :param epoch: epoch
    :param dataloader: train data loader
    :param criterion: train loss function
    :param args: arguments
    :param writer: tensor board summary writer
    :param device: device
    :param classes: dictionary {class label: {name: ..., color: (...)}
    :param save_preds: save predictions
    :param experiment: comet experiment
    :return: mIOU, loss value
    """
    model.eval()

    loss_total = AverageMeter()

    metric = ConfusionMatrix(num_classes=args.NUM_CLASSES)
    set_seed(epoch - 1)
    loop = tqdm.tqdm(dataloader, total=len(dataloader))

    # one epoch loop
    with torch.inference_mode():

        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss, _, _ = criterion(outputs, targets)

            # update average of loss & metric
            if type(outputs) is OrderedDict:
                outputs = outputs["out"]
            metric.update(targets=targets, outputs=outputs.argmax(dim=1))
            loss_total.update(loss)

            loop.set_description(f"Valid ====>> Epoch:{epoch}    Loss:{loss_total.avg:.4}")

            if (batch_idx + 1) % 50 == 0 and save_preds:
                mask = outputs[0].detach().cpu().argmax(dim=0)
                mask = torch.nn.functional.one_hot(mask, args.NUM_CLASSES).to(torch.bool).permute(2, 0, 1)

                image = inputs[0].detach().cpu()
                colors = []
                for v in classes.values():
                    colors.append(v["color"])

                image = unnormalize_image(image, mean=args.MEAN, std=args.STD)

                img_mask = draw_segmentation_masks(
                    image=image, masks=mask, colors=colors, alpha=0.8)

                writer.add_image(f"mask{batch_idx + 1}", img_mask, epoch)

                img_mask = img_mask.permute(1, 2, 0).numpy()
                img_mask = Image.fromarray(img_mask)
                experiment.log_image(img_mask, f"mask{batch_idx + 1}", step=epoch)
                img_mask.save(args.log + f"predicts/mask{batch_idx + 1}.jpg")

    miou = metric.calculate()
    torch.cuda.empty_cache()

    return miou, loss_total.avg.item()


def overfit(model: torch.nn.Module, epoch: int, batch: Tuple[torch.Tensor, torch.Tensor],
            optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, criterion: torch.nn.Module,
            model_ema: torch.nn.Module, args: Namespace, device: torch.device) -> None:
    """
    overfit test :)
    :param model: model
    :param epoch: current epoch
    :param batch: batch to overfit
    :param optimizer: train optimizer
    :param scaler: train scaler
    :param criterion: train loss function
    :param model_ema: model ema
    :param args: arguments
    :param device: device
    """
    # set model for train
    model.train()

    # save losses of each step
    loss_total = AverageMeter()
    semantic_total = AverageMeter()
    aux_total = AverageMeter()

    # Confusion matrix to calculate iou & init
    metric = ConfusionMatrix(num_classes=args.NUM_CLASSES)
    set_seed(epoch - 1)

    # one training step
    inputs, targets = batch[0].to(device), batch[1].to(device)
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        # forward step & calc loss & metric
        outputs = model(inputs)
        semantic_loss, semantic_aux, edge_loss = criterion(outputs, targets)
        loss = semantic_loss + (0.4 * semantic_aux) + edge_loss
        if type(outputs) is OrderedDict:
            outputs = outputs["out"]
        metric.update(targets=targets, outputs=outputs.argmax(dim=1))

    # backward step
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if model_ema is not None:
        model_ema.update(model)

    # calculate average of loss & metric

    loss_total.update(loss)
    semantic_total.update(semantic_loss)
    aux_total.update((0.4 * semantic_aux) + edge_loss)
    torch.cuda.empty_cache()

    # print details
    print(f"Train -> Epoch:{epoch} Loss:{loss_total.avg:.4}"
          f" Out Loss: {semantic_total.avg:.4} Aux Loss: {aux_total.avg:.4}")
