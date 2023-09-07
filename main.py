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
    parser.add_argument("--name", default="regseg_cityv2_run2", type=str,
                        help=f"experiment name ")
    parser.add_argument("--dataset", default="cityscapes", type=str,
                        help=f"datasets name ")

    return parser


def main() -> None:
    """
    Hi I am Amirhosein Feiz I write this code to train my semantic segmentation ideas :)
    """
    gc.collect()
    utils.setup_env()
    parser = get_args()
    args = parser.parse_args()

    if args.dataset == "cityscapes":
        yaml_file = "./configs/cityscapes.yaml"
    elif args.dataset == "voc":
        yaml_file = "./configs/voc.yaml"
    else:
        raise NotImplemented

    writer = utils.create_log_dir(name=args.name, parser=parser)
    args = parser.parse_args()
    device, experiment = utils.add_yaml_2_args_and_save_configs_and_get_device(parser=parser, yaml_path=yaml_file,
                                                                               log_path=args.log)
    args = parser.parse_args()
    train_transforms, val_transforms = transforms.get_augs(args)
    dataset, dataset_classes = datasets.DATASETS[args.dataset]
    train_ds = dataset(phase="train", root=args.DIR, transforms=train_transforms, version=args.VERSION)
    valid_ds = dataset(phase="val", root=args.DIR, transforms=val_transforms, version=args.VERSION)

    train_sampler = torch.utils.data.RandomSampler(train_ds)
    valid_sampler = torch.utils.data.SequentialSampler(valid_ds)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.BATCH_SIZE,
                                           sampler=train_sampler, num_workers=args.NUM_WORKER, drop_last=True,
                                           collate_fn=datasets.collate_fn, pin_memory=True)

    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=1, sampler=valid_sampler,
                                           num_workers=args.NUM_WORKER, drop_last=True,
                                           collate_fn=datasets.collate_fn, pin_memory=True)

    model = models.MODELS_COLLECTIONS[args.MODEL](args.NUM_CLASSES, quantization=args.QAT, inference=False)

    if args.QAT:
        print("Quantization Aware Training")
        model.load_state_dict(torch.load(args.QAT_PRETRAIN_WEIGHTS)["model"])
        print(f"Float32 model size{utils.print_model_size(model)}")
        model.fuse_model(is_qat=True)
        qconfig = torch.quantization.get_default_qat_qconfig()
        model.qconfig = qconfig
        model = torch.quantization.prepare_qat(model)
        args.name = args.name + "_qat"
        experiment.set_name(args.name)

    summary(model, (args.BATCH_SIZE, 3, args.TRAIN_SIZE[0], args.TRAIN_SIZE[1]),
            device=device, col_width=16, col_names=["output_size", "num_params", "mult_adds"], verbose=1)
    experiment.set_model_graph(model)

    print()

    optimizer, scaler, scheduler, model_ema = train.get_optimizer(model=model, num_iters=len(train_dl),
                                                                  args=args, device=device)
    criterion = train.Criterion(args=args)

    if args.RESUME:
        model, optimizer, scaler, scheduler, start_epoch, best_acc, log_dict = utils.resume(model=model,
                                                                                            optimizer=optimizer,
                                                                                            scaler=scaler,
                                                                                            scheduler=scheduler,
                                                                                            model_ema=model_ema,
                                                                                            args=args)
    else:
        start_epoch, best_acc = 0, 0.0
        log_dict = {
            "Train Loss": [],
            "Valid Loss": [],
            "Train Metric": [],
            "Valid Metric": [],
            "LR": []
        }

    model.to(device)

    early_stopping = train.EarlyStopping(tolerance=args.EARLY_STOPPING_TOLERANCE,
                                         min_delta=args.EARLY_STOPPING_DELTA)

    for epoch in range(start_epoch + 1, args.EPOCHS + 1):
        utils.set_seed(epoch)
        with experiment.train():
            train_acc, train_loss = train.train_one_epoch(model=model, epoch=epoch, dataloader=train_dl,
                                                          optimizer=optimizer, scaler=scaler, criterion=criterion,
                                                          model_ema=model_ema, scheduler=scheduler, args=args,
                                                          device=device, log_dict=log_dict)

        with torch.inference_mode():
            with experiment.validate():
                valid_acc, valid_loss = train.evaluate(model=model, epoch=epoch, dataloader=valid_dl,
                                                       criterion=criterion, args=args, writer=writer,
                                                       experiment=experiment, device=device, classes=dataset_classes,
                                                       save_preds=True)

                if args.QAT:
                    print()
                    model.to(torch.device("cpu"))
                    model.eval()
                    quantized_eval_model = torch.quantization.convert(model)
                    # utils.run_benchmark(quantized_eval_model, valid_dl)
                    model.to(device)

                else:
                    quantized_eval_model = None

        # add infos to tensorboard
        experiment.log_metric("Loss_train", train_loss, epoch=epoch)
        experiment.log_metric("Metric_train", train_acc, epoch=epoch)
        experiment.log_metric("Loss_valid", valid_loss, epoch=epoch)
        experiment.log_metric("Metric_valid", valid_acc, epoch=epoch)

        log_dict["Train Loss"].append(train_loss)
        log_dict["Valid Loss"].append(valid_loss)
        log_dict["Train Metric"].append(train_acc)
        log_dict["Valid Metric"].append(valid_acc)
        log_dict["LR"].append(utils.get_lr(optimizer))

        plt.clf()
        plt.plot(log_dict["Train Loss"], label="Train Loss")
        plt.plot(log_dict["Valid Loss"], label="Valid Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.log, f"loss_log.png"))

        plt.clf()
        plt.plot(log_dict["Train Metric"], label="Train Metric")
        plt.plot(log_dict["Valid Metric"], label="Valid Metric")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.log, f"metric_log.png"))

        plt.clf()
        plt.plot(log_dict["LR"], label="LR")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.log, f"lr_log.png"))

        writer.add_scalar('Loss/train', train_loss, epoch, walltime=epoch,
                          display_name="Training Loss", )
        writer.add_scalar('Metric/train', train_acc, epoch, walltime=epoch,
                          display_name="Training Metric", )
        writer.add_scalar('LR/train', utils.get_lr(optimizer), epoch, walltime=epoch,
                          display_name="Learning rate", )
        writer.add_scalar('Loss/valid', valid_loss, epoch, walltime=epoch,
                          display_name="Validation Loss", )
        writer.add_scalar('Metric/valid', valid_acc, epoch, walltime=epoch,
                          display_name="Validation Metric", )

        best_acc = utils.save(model=model, acc=valid_acc, best_acc=best_acc, scaler=scaler, optimizer=optimizer,
                              scheduler=scheduler, model_ema=model_ema, epoch=epoch, args=args,
                              qat_model=quantized_eval_model, log_dict=log_dict)

        early_stopping(train_loss=train_loss, validation_loss=valid_loss)
        if early_stopping.early_stop:
            print(f"Early Stop at Epoch: {epoch}")

            break

        if epoch == 1:
            write_mode = "w"
        else:
            write_mode = "a"
        log_path = os.path.join(args.log, f"log.txt")
        with open(log_path, write_mode) as f:
            f.write(f"Epoch: {epoch},"
                    f" Train mIOU: {train_acc}, Train loss: {train_loss},"
                    f" Valid mIOU: {valid_acc}, Valid loss: {valid_loss}\n")

        print()
        log_path = os.path.join(args.log, f"log.csv")
        df = pd.DataFrame(log_dict)
        df.to_csv(log_path, index=False)
        model.to(torch.device("cpu"))
        for name, param in model.named_parameters():
            if param.dim() != 1:
                experiment.log_histogram_3d(param, name, epoch)
        model.to(device)

    writer.add_hparams(
        hparam_dict={
            "lr": args.LR,
            "weight_decay": args.WEIGHT_DECAY,
            "optimizer": args.OPTIMIZER,
            "batch_size": args.BATCH_SIZE,
            "loss": args.LOSS,
        },
        metric_dict={
            "loss": valid_loss,
            "acc": valid_acc,
        })


    writer.close()
    experiment.end()
    print(f"Training finished best mIOU is {best_acc:.4}")


if __name__ == "__main__":
    main()
