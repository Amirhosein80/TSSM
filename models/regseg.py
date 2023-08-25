from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn

from models.model_utils import ConvBNAct, CatLayer, AddLayer, SEBlock


class DilatedConv(nn.Module):
    """
    Dilated Conv of RegSeg
    """

    def __init__(self, channels: int, stride: int = 1, dilations: List = [1], quantization: bool = False) -> None:
        """
        :param channels: number of input channels
        :param dilations: dilations list
        :param stride: conv stride
        :param quantization: use quantization
        """
        super().__init__()
        self.num_splits = len(dilations)
        assert channels % len(dilations) == 0
        assert channels % 16 == 0

        mid_channel = channels // len(dilations)

        self.convs = [
            ConvBNAct(mid_channel, mid_channel, kernel_size=3, use_act=True,
                      stride=stride, padding=dilation, dilation=dilation,
                      groups=mid_channel // 16) for dilation in dilations
        ]

        self.convs = nn.ModuleList(self.convs)
        self.cat = CatLayer(dim=1, quantization=quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = list(torch.chunk(x, self.num_splits, 1))
        for idx, conv in enumerate(self.convs):
            x[idx] = conv(x[idx])
        x = self.cat(x)
        return x


class DilatedBlock(nn.Module):
    """
    D block RegSeg
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 quantization: bool = False, dilations: List = [1]) -> None:
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param quantization: use quantization
        :param stride: conv stride
        :param dilations: dilations list
        """
        super().__init__()
        self.stride = stride

        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.conv2 = DilatedConv(out_channels, dilations=dilations, stride=stride, quantization=quantization)
        self.conv3 = ConvBNAct(out_channels, out_channels, kernel_size=1, use_act=False)
        self.se = SEBlock(out_channels, quantization=quantization)
        self.add = AddLayer(quantization=quantization, use_relu=True)

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                ConvBNAct(in_channels, out_channels, kernel_size=1, use_act=False)
            )
        else:
            self.conv_stride = None
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        y = self.conv1(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        y = self.conv2(y)
        y = self.conv3(y)
        return self.add(x, y)


class BackBone(nn.Module):
    """
    BackBone of RegSeg
    """

    def __init__(self, quantization: bool = False) -> None:
        """
        :param quantization: use quantization
        """
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            DilatedBlock(32, 48, stride=2, quantization=quantization),
        )
        self.stage8 = nn.Sequential(
            DilatedBlock(48, 128, stride=2, quantization=quantization),
            DilatedBlock(128, 128, stride=1, quantization=quantization),
            DilatedBlock(128, 128, stride=1, quantization=quantization),
        )
        self.stage16 = nn.Sequential(
            DilatedBlock(128, 256, stride=2, quantization=quantization),
            DilatedBlock(256, 256, stride=1, quantization=quantization),

            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 2]),

            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 4]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 4]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 4]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 4]),

            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),
            DilatedBlock(256, 256, stride=1, quantization=quantization, dilations=[1, 14]),

            DilatedBlock(256, 320, stride=1, quantization=quantization, dilations=[1, 14]),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        """
        forward function :)
        :param x: input feature maps
        :return: dict of feature maps
        """
        x4 = self.stem(x)

        x8 = self.stage8(x4)

        x16 = self.stage16(x8)

        return x4, x8, x16


class Decoder(nn.Module):
    """
    Decoder of RegSeg
    """

    def __init__(self, quantization: bool = False) -> None:
        """
        :param quantization: use quantization
        """
        super().__init__()
        self.conv4 = ConvBNAct(48, 8, 1)
        self.conv8 = ConvBNAct(128, 128, 1)
        self.conv16 = ConvBNAct(320, 128, 1)

        self.conv_sum = ConvBNAct(128, 64, kernel_size=3, stride=1, padding=1)

        self.cat = CatLayer(dim=1, quantization=quantization)
        self.add = AddLayer(quantization=quantization)

    def forward(self, x4: torch.Tensor, x8: torch.Tensor, x16: torch.Tensor) -> torch.Tensor:
        x4 = self.conv4(x4)
        x8 = self.conv8(x8)
        x16 = self.conv16(x16)

        x4_shape = x4.shape[-2:]
        x8_shape = x8.shape[-2:]

        x = self.add(x8, nn.functional.interpolate(x16, size=x8_shape, mode="bilinear", align_corners=False))
        x = self.conv_sum(x)
        x = self.cat([x4, nn.functional.interpolate(x, size=x4_shape, mode="bilinear", align_corners=False)])

        return x


class Head(nn.Module):
    """
    Head block BiSeNet V2
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        """
        :param channels: number of input channels
        :param num_classes: number of classes
        """
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3,
                               stride=1, padding=1, use_act=True)
        self.conv2 = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = self.conv1(x)
        return self.conv2(x)


class RegSeg(nn.Module):
    """
    RegSeg
    """

    def __init__(self, num_classes: int, quantization: bool = False, grad_cam=False, inference=False) -> None:
        """
        :param num_classes: number of classes
        :param quantization: use quantization
        """
        super().__init__()

        if quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        self.backbone = BackBone(quantization=quantization)
        self.decoder = Decoder(quantization=quantization)

        self.head = Head(64 + 8, num_classes)
        self.grad_cam = grad_cam and (not quantization)
        self.gradients = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps or OrderedDict
        """
        input_shape = x.shape[-2:]

        if self.quant is not None:
            x = self.quant(x)

        x4, x8, x16 = self.backbone(x)
        if self.grad_cam:
            x4.register_hook(self.activations_hook)
            x8.register_hook(self.activations_hook)
            x16.register_hook(self.activations_hook)
        x = self.decoder(x4=x4, x8=x8, x16=x16)
        x = nn.functional.interpolate(self.head(x), size=input_shape, mode="bilinear", align_corners=False)

        if self.dequant is not None:
            x = self.dequant(x)

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

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        """
        fuse model for quantization :)
        :param is_qat: use quantization aware training
        """
        for m in self.modules():
            if type(m) is ConvBNAct:
                m.fuse_model(is_qat)

    def activations_hook(self, grad):
        """
        save gradient
        :param grad: gradient
        """
        self.gradients.append(grad)

    def get_activations_gradient(self):
        """
        get gradients
        :return: list of gradients
        """
        return self.gradients[::-1]

    def get_activations(self, x):
        """
        :param x: get activations
        :return: dict of activations
        """
        x4, x8, x16 = self.backbone(x)
        return {
            "feature_4": x4,
            "feature_8": x8,
            "feature_16": x16,
        }

if __name__ == "__main__":
    # model = DeepLabV3()
    import torchinfo
    dump = torch.randn(8, 3, 768 // 1, 768 // 1).cuda()
    deeplab = RegSeg(19).cuda()
    out_dump = deeplab(dump)
    print(torchinfo.summary(deeplab, (8, 3, 768 // 1, 768 // 1), device="cuda"))
