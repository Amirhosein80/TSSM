import copy
import os
import comet_ml

import torch
import argparse
import utils

from torchinfo import summary

import train
import transforms
import datasets
import models
import gc


def get_args() -> argparse.ArgumentParser:
    """
    get arguments of program :)
    :return: parser
    """
    parser = argparse.ArgumentParser(description="A Good Python Code for train your semantic segmentation models",
                                     add_help=True)
    parser.add_argument("--name", default="deeplab", type=str,
                        help=f"experiment name ")
    parser.add_argument("--dataset", default="cityscapes", type=str,
                        help=f"datasets name ")

    return parser


def main() -> None:
    """
    main function
    """
    gc.collect()
    utils.setup_env()
    parser = get_args()
    args = parser.parse_args()

    if args.dataset == "cityscapes":
        yaml_file = "./configs/cityscapes.yaml"
    else:
        raise NotImplemented

    writer = utils.create_log_dir(name=args.name, parser=parser)
    args = parser.parse_args()
    device, experiment = utils.add_yaml_2_args_and_save_configs_and_get_device(parser=parser, yaml_path=yaml_file,
                                                                               log_path=args.log)
    args = parser.parse_args()
    train_transforms, val_transforms = transforms.get_augs(args)
    dataset, dataset_classes = datasets.DATASETS[args.dataset]
    train_ds = dataset(phase="train", root=args.DIR, transforms=train_transforms)
    valid_ds = dataset(phase="val", root=args.DIR, transforms=val_transforms)

    train_sampler = torch.utils.data.RandomSampler(train_ds)
    valid_sampler = torch.utils.data.SequentialSampler(valid_ds)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.BATCH_SIZE,
                                           sampler=train_sampler, num_workers=args.NUM_WORKER, drop_last=True,
                                           collate_fn=datasets.collate_fn, pin_memory=True)

    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=1, sampler=valid_sampler,
                                           num_workers=args.NUM_WORKER, drop_last=True,
                                           collate_fn=datasets.collate_fn, pin_memory=True)

    model = models.MODELS_COLLECTIONS[args.MODEL](args.NUM_CLASSES, quantization=args.QAT)

    if args.QAT:
        print("Quantization Aware Training")
        model.load_state_dict(torch.load(args.QAT_PRETRAIN_WEIGHTS)["model"])
        model.fuse_model(is_qat=True)
        qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        model.qconfig = qconfig
        torch.ao.quantization.prepare_qat(model, inplace=True)
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
        model, optimizer, scaler, scheduler, start_epoch, best_acc = utils.resume(model=model, optimizer=optimizer,
                                                                                  scaler=scaler, scheduler=scheduler,
                                                                                  model_ema=model_ema, args=args)
    else:
        start_epoch, best_acc = 0, 0.0

    model.to(device)

    early_stopping = train.EarlyStopping(tolerance=args.EARLY_STOPPING_TOLERANCE,
                                         min_delta=args.EARLY_STOPPING_DELTA)

    if args.QAT:
        model.apply(torch.quantization.enable_observer)
        model.apply(torch.quantization.enable_fake_quant)

    for epoch in range(start_epoch + 1, args.EPOCHS + 1):
        utils.set_seed(epoch)
        with experiment.train():
            train_acc, train_loss = train.train_one_epoch(model=model, epoch=epoch, dataloader=train_dl,
                                                          optimizer=optimizer, scaler=scaler, criterion=criterion,
                                                          model_ema=model_ema, scheduler=scheduler, args=args,
                                                          device=device)

        with torch.inference_mode():
            with experiment.validate():

                if epoch >= args.QAT_OBSERVER_EPOCH and args.QAT:
                    model.apply(torch.quantization.disable_observer)
                if epoch >= args.QAT_BATCHNORM_EPOCH and args.QAT:
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

                valid_acc, valid_loss = train.evaluate(model=model, epoch=epoch, dataloader=valid_dl,
                                                       criterion=criterion, args=args, writer=writer,
                                                       experiment=experiment, device=device, classes=dataset_classes,
                                                       save_preds=True)

                if args.QAT:
                    print()
                    quantized_eval_model = copy.deepcopy(model)
                    quantized_eval_model.eval()
                    quantized_eval_model.to(torch.device("cpu"))
                    quantized_eval_model = torch.quantization.convert(quantized_eval_model, inplace=False)
                    print("Evaluate QAT model")
                    qat_acc, qat_loss = train.evaluate(model=quantized_eval_model, epoch=epoch, dataloader=valid_dl,
                                                       criterion=criterion, args=args, writer=writer,
                                                       experiment=experiment, device=torch.device("cpu"),
                                                       classes=dataset_classes, save_preds=False)

                else:
                    quantized_eval_model = None

        # add infos to tensorboard
        experiment.log_metric("Loss_train", train_loss, epoch=epoch)
        experiment.log_metric("Metric_train", train_acc, epoch=epoch)
        experiment.log_metric("Loss_valid", valid_loss, epoch=epoch)
        experiment.log_metric("Metric_valid", valid_acc, epoch=epoch)

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

        if args.QAT:
            writer.add_scalar('Loss/QAT', qat_loss, epoch, walltime=epoch,
                              display_name="QAT Loss", )
            writer.add_scalar('Metric/QAT', qat_acc, epoch, walltime=epoch,
                              display_name="QAT Metric", )
            experiment.log_metric("Loss_QAT", qat_loss, epoch=epoch)
            experiment.log_metric("Metric_QAT", qat_acc, epoch=epoch)

        best_acc = utils.save(model=model, acc=valid_acc, best_acc=best_acc, writer=writer,
                              scaler=scaler, optimizer=optimizer, scheduler=scheduler, device=device,
                              model_ema=model_ema, epoch=epoch, args=args, qat_model=quantized_eval_model)

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
                    f" Valid mIOU: {valid_acc}, Valid loss: {valid_loss}")

        print()

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

    torch.jit.save(torch.jit.script(model),
                   os.path.join(args.log, f"checkpoint/last_scripted_{args.name}.pt"))

    model.to(torch.device("cpu"))
    if quantized_eval_model is not None:
        torch.jit.save(torch.jit.script(quantized_eval_model),
                       os.path.join(args.log, f"checkpoint/last_qat_scripted_{args.name}.pt"))

    writer.close()
    experiment.end()
    print("Training finished")


if __name__ == "__main__":
    main()
