from collections import OrderedDict

import torch
import torchvision.models.segmentation as models
import torch.nn as nn
from typing import List, Dict
from torchvision.models._utils import IntermediateLayerGetter


class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 19, quantization: bool = False) -> None:
        """
        :param num_classes: number of output classes
        :param quantization: use quantization
        """
        super().__init__()
        if quantization:
            print("DeepLabV3 doesn't support quantization")
        model = models.deeplabv3_mobilenet_v3_large(
            weights=models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
            aux_loss=True
        )
        model.classifier[-1] = nn.Conv2d(256, 128, 1)
        model.aux_classifier[-1] = nn.Conv2d(10, num_classes, 1)
        self.backbone = IntermediateLayerGetter(model.backbone, return_layers={"3": "inter", "6": "aux", "16": "out"})
        self.aspp = model.classifier
        self.aux_classifier = model.aux_classifier
        del model
        self.head = nn.Sequential(
            nn.Conv2d(128 + 24, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1))

    def forward(self, x: torch.Tensor):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps or OrderedDict
        """
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.aspp(x)
        x = nn.functional.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, features["inter"]], dim=1)
        x = nn.functional.interpolate(self.head(x), size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.training:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result

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
    import torchinfo

    #
    dump = torch.randn(2, 3, 768, 768).cuda()
    deeplab = DeepLabV3().cuda()
    out = deeplab(dump)
    print(torchinfo.summary(deeplab, (1, 3, 1024, 2048), device="cuda"))
    # for name, module in deeplab.backbone.named_children():
    #     print(name)
