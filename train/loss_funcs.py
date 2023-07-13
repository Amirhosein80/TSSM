from typing import Optional, Tuple
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn as nn


class BootstrappedCE(nn.Module):
    """
    OHEM Cross Entropy for semantic segmentation :)
    Code Inspired from: "https://arxiv.org/pdf/1604.03540"
    """

    def __init__(self, loss_th: float = 0.3, ignore_index: int = 255, label_smoothing: float = 0.0,
                 weight: Optional[torch.Tensor] = None) -> None:
        """
        :param loss_th: ohem loss threshold. default is 0.3
        :param ignore_index: ignore value in target. default is 255
        :param label_smoothing: epsilon value in label smoothing. default is 0.0
        :param weight: weight of each class in loss function. default is None
        """
        super().__init__()
        self.threshold = loss_th
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none", label_smoothing=label_smoothing, weight=weight
        )

    def forward(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        forward loss function :)
        :param output: model predicts
        :param labels: real labels
        """
        pixel_losses = self.criterion(output, labels).contiguous().view(-1)
        k = torch.numel(labels) // 16
        mask = (pixel_losses > self.threshold)
        if torch.sum(mask).item() > k:
            pixel_losses = pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, k)
        return pixel_losses.mean()


class Criterion(nn.Module):
    """
    loss class for train :)
    """

    def __init__(self, args: Namespace) -> None:
        """
        :param args: arguments
        """
        super().__init__()
        if args.LOSS == "CROSS":
            self.criterion = nn.CrossEntropyLoss(
                weight=args.CLASS_WEIGHTS if args.USE_CLASS_WEIGHTS else None,
                ignore_index=args.IGNORE_LABEL,
                label_smoothing=args.LABEL_SMOOTHING
            )
        elif args.LOSS == "OHEM":
            self.criterion = BootstrappedCE(loss_th=args.OHEM_THRESH, ignore_index=args.IGNORE_LABEL,
                                            label_smoothing=args.LABEL_SMOOTHING,
                                            weight=args.CLASS_WEIGHTS if args.USE_CLASS_WEIGHTS else None
                                            )
        else:
            raise NotImplemented

    def forward(self, outputs: OrderedDict | torch.Tensor, target: torch.Tensor) \
            -> Tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
        """
        forward function for loss:)
        :param outputs: model outputs
        :param target: segment labels
        :param edge: SAB labels
        :return: output loss, auxiliary loss, SAB loss
        """
        semantic_loss = 0.0
        semantic_aux = 0.0
        semantic_edge = 0.0
        if type(outputs) is OrderedDict:
            for key, value in outputs.items():
                if "out" in key:
                    semantic_loss += self.criterion(value, target)
                elif "aux" in key:
                    semantic_aux += self.criterion(value, target)

        elif type(outputs) is torch.Tensor:
            semantic_loss += self.criterion(outputs, target)

        else:
            raise NotImplemented

        return semantic_loss, semantic_aux, semantic_edge


if __name__ == "__main__":
    pass
