from collections import OrderedDict
from typing import List, Optional, Dict

import torch
import torch.nn as nn

from models.model_utils import ConvBNAct, CatLayer


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    block in unet
    :param in_channels: input channels
    :param out_channels: output channels
    :return:
    """
    return nn.Sequential(
        ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class Unet(nn.Module):
    """
    Unet Model + Quantization :)
    Paper: "https://arxiv.org/pdf/1505.04597v1.pdf"
    """
    def __init__(self, num_classes: int = 19, quantization: bool = False) -> None:
        """
        :param num_classes: number of output classes
        :param quantization: use quantization
        """
        super().__init__()

        if quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        self.stage1 = block(3, 32)
        self.stage2 = block(32, 64)
        self.stage3 = block(64, 128)
        self.stage4 = block(128, 256)
        self.bottleneck = nn.Sequential(
            ConvBNAct(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBNAct(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBNAct(512, 256, kernel_size=3, stride=1, padding=1),
        )
        self.decoder4 = block(512, 128)
        self.decoder3 = block(256, 64)
        self.decoder2 = block(128, 32)
        self.decoder1 = block(64, 32)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.head = nn.Conv2d(32, num_classes, 1)

        self.cat4 = CatLayer(quantization=quantization)
        self.cat3 = CatLayer(quantization=quantization)
        self.cat2 = CatLayer(quantization=quantization)
        self.cat1 = CatLayer(quantization=quantization)

    def forward(self, x: torch.Tensor):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps or OrderedDict
        """
        output = OrderedDict()
        if self.quant is not None:
            x = self.quant(x)
        x1 = self.stage1(x)
        x2 = self.stage2(self.max_pool_2(x1))
        x3 = self.stage3(self.max_pool_2(x2))
        x4 = self.stage4(self.max_pool_2(x3))

        x = self.bottleneck(self.max_pool_2(x4))

        x = self.cat4([self.up_sample_2(x), x4])
        x = self.decoder4(x)
        x = self.cat3([self.up_sample_2(x), x3])
        x = self.decoder3(x)
        x = self.cat2([self.up_sample_2(x), x2])
        x = self.decoder2(x)
        x = self.cat1([self.up_sample_2(x), x1])
        x = self.decoder1(x)

        x = self.head(x)
        if self.dequant is not None:
            x = self.dequant(x)
        output["out"] = x

        return output

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

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        """
        fuse model for quantization :)
        :param is_qat: use quantization aware training
        """
        for m in self.modules():
            if type(m) is ConvBNAct:
                m.fuse_model(is_qat)


if __name__ == "__main__":
    dump = torch.randn(1, 3, 1024, 2048).cuda()
    model = Unet().cuda()
    out = model(dump)
    import torchinfo

    # model.eval()
    print(torchinfo.summary(model, (1, 3, 1024, 2048), device="cuda"))
