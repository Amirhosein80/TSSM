import torch
import torchvision.models.segmentation as models
import torch.nn as nn
from typing import List, Dict


class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 19, quantization: bool = False) -> None:
        """
        :param num_classes: number of output classes
        :param quantization: use quantization
        """
        super().__init__()
        if quantization:
            print("DeepLabV3 doesn't support quantization")
        self.model = models.deeplabv3_mobilenet_v3_large(
            weights=models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
            aux_loss=True
        )
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
        self.model.aux_classifier[-1] = nn.Conv2d(10, num_classes, 1)

    def forward(self, x: torch.Tensor):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps or OrderedDict
        """
        x = self.model(x)
        return x

    def get_params(self, lr: float, weight_decay: float) -> List[Dict]:
        """
        get model parameters (*doesn't use WD for BN & Bias) :)
        :param lr: learning rate
        :param weight_decay: weight decay for conv or linear layers
        :return: list of parameters
        """
        params_wd = []
        params_nwd = []

        for p in self.parameters():
            if p.dim == 1:
                params_nwd.append(p)
            else:
                params_wd.append(p)

        params = [
            {"params": params_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": params_nwd, "lr": lr, "weight_decay": 0},
        ]

        return params


if __name__ == "__main__":
    # model = DeepLabV3()
    # import torchinfo
    #
    # print(torchinfo.summary(model, (1, 3, 1024, 2048), device="cuda"))
    dump = torch.randn(2, 3, 768, 768).cuda()
    model = DeepLabV3().cuda()
    out = model(dump)
