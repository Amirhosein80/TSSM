from argparse import Namespace
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


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


class FocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation :)
    Focal Loss paper: https://arxiv.org/pdf/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        """
        :param alpha: alpha value for focal loss. default is 0.25
        :param gamma: gamma value for focal loss. default is 2.0
        :param ignore_index: ignore value in target. default is 255
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        calculate loss :)
        :param output: output of nural network shape: (B, C, H, W)
        :param labels: true labels shape: (B, H, W)
        :return: loss value
        """
        _, c, _, _ = output.shape

        ce_loss = self.ce(output, labels)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


class SobelFilter(nn.Module):
    """
    Pytorch Sobel Filter for edge from labels
    """

    def __init__(self, ignore_label: int = 255, semantic: bool = True) -> None:
        """
        :param ignore_label: ignore value for loss
        :param semantic: semantic labels or edge labels
        """
        super().__init__()
        self.semantic = semantic
        self.ignore_label = ignore_label
        self.sobel_x_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                           dtype=torch.float16, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_y_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                           dtype=torch.float16, requires_grad=False).view(1, 1, 3, 3)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param labels: model outputs
        :return: edges
        """
        x = torch.unsqueeze(labels, dim=1)
        x = x.to(torch.float16)
        gradient_x = nn.functional.conv2d(x, self.sobel_x_filter.to(labels.device), stride=1, padding=1)
        gradient_y = nn.functional.conv2d(x, self.sobel_y_filter.to(labels.device), stride=1, padding=1)

        gradient_magnitude = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        if self.semantic:
            gradient_magnitude = torch.squeeze(gradient_magnitude, dim=1)
            gradient_magnitude = torch.where(gradient_magnitude > 0, labels, self.ignore_label).to(torch.int64)
        else:
            gradient_magnitude = torch.where(gradient_magnitude > 0, 1.0, 0.0)

        return gradient_magnitude


class Criterion(nn.Module):
    """
    loss class for train :)
    """

    def __init__(self, args: Namespace) -> None:
        """
        :param args: arguments
        """
        super().__init__()
        self.sobel_edge = SobelFilter(args.IGNORE_LABEL, semantic=False)
        self.edge_criterion = nn.BCEWithLogitsLoss()
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
        elif args.LOSS == "FOCAL":
            self.criterion = FocalLoss(alpha=args.FOCAL_ALPHA, ignore_index=args.IGNORE_LABEL,
                                       gamma=args.FOCAL_GAMMA)
        else:
            raise NotImplemented

        self.args = args

    def forward(self, outputs: OrderedDict | torch.Tensor, target: torch.Tensor) \
            -> Tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
        """
        forward function for loss:)
        :param outputs: model outputs
        :param target: segment labels
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
                elif "edge" in key:
                    semantic_edge += self.edge_criterion(value, self.sobel_edge(target))

        elif type(outputs) is torch.Tensor:
            semantic_loss += self.criterion(outputs, target)

        else:
            raise NotImplemented

        return semantic_loss, semantic_aux, semantic_edge


if __name__ == "__main__":
    pass
