import torch
from typing import List, Tuple


class AverageMeter:
    """
    save & calculate metric average :)
    """

    def __init__(self) -> None:
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self) -> None:
        """
        reset values :)
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """
        update average :)
        :param val: metric value
        :param n: number of values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix:
    """
    Confusion Matrix for calculate mIOU :)
    """

    def __init__(self, num_classes: int) -> None:
        """
        :param num_classes: number of classes
        """
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, targets, outputs) -> None:
        """
        update metric :)
        :param targets: labels
        :param outputs:  predicts
        """
        targets = targets.cpu().flatten()
        outputs = outputs.cpu().flatten()
        k = (targets >= 0) & (targets < self.num_classes)
        inds = self.num_classes * targets + outputs
        inds = inds[k]
        self.mat += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        """
        reset metric :)
        """
        self.mat.zero_()

    def compute(self) -> Tuple[float, List]:
        """
        compute accuracy and iou for datas :)
        :return: accuracy, ious list
        """
        h = self.mat.float() + 1e-8
        acc_global = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global = acc_global.item() * 100
        iu = (iu * 100).tolist()
        return acc_global, iu

    def calculate(self) -> float:
        """
        calculate & show mIOU :)
        :return: mIOU
        """
        acc_global, iu = self.compute()
        acc_global = round(acc_global, 2)
        IOU = [round(i, 2) for i in iu]
        mIOU = sum(iu) / len(iu)
        mIOU = round(mIOU, 2)
        print(f"IOU: {IOU}\nmIOU: {mIOU}, accuracy: {acc_global}")
        return mIOU
