from collections import OrderedDict

import torch
import torchvision.models.segmentation as models
import torch.nn as nn
from typing import List, Dict
from torchvision.models._utils import IntermediateLayerGetter

from models.model_utils import AddLayer, CatLayer, ConvBNAct


class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 19, quantization: bool = False, inference: bool = False) -> None:
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
        model.aux_classifier[-1] = nn.Conv2d(10, num_classes, 1)
        model.classifier[-1] = nn.Conv2d(64, num_classes, 1)

        # pretrained parts
        self.backbone = IntermediateLayerGetter(model.backbone, return_layers={"3": "inter", "6": "aux", "16": "out"})
        self.aspp = model.classifier[0]

        del model

        # scratch parts
        self.add = AddLayer(use_relu=True)
        self.cat = CatLayer(dim=1)

        self.conv_4 = ConvBNAct(24, 8, 1)
        self.conv_8 = ConvBNAct(40, 40, 1)
        self.aspp.project[0] = nn.Conv2d(256 * 5, 40, kernel_size=1, bias=False)
        self.aspp.project[1] = nn.BatchNorm2d(40)

        self.conv_sum = ConvBNAct(40, 40, kernel_size=3, stride=1, padding=1)

        self.edge_classifier = nn.Sequential(
            ConvBNAct(24, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 1, 1)
        )
        self.aux_classifier = nn.Sequential(
            ConvBNAct(40, 40, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(40, num_classes, 1)
        )
        self.head = nn.Sequential(
            ConvBNAct(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, num_classes, 1)
        )
        self.inference = inference

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
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.add(x, self.conv_8(features["aux"]))
        x = self.conv_sum(x)
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.cat([x, self.conv_4(features["inter"])])
        x = nn.functional.interpolate(self.head(x), size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if not self.inference:
            x = features["inter"]
            x = self.edge_classifier(x)
            x = nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["edge"] = x

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
            if p.dim() == 1:
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
    deeplab = DeepLabV3(19).cuda()
    out = deeplab(dump)
    print(torchinfo.summary(deeplab, (1, 3, 1024, 2048), device="cuda"))
