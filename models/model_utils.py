import math
from typing import Optional, Union, List, Any

import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional


# ============
#  Functions
# =============


def efficientnet_init_weights(m: nn.Module) -> None:
    """
    EfficientNet weight initialization :)
    Reference Paper: https://arxiv.org/pdf/1905.11946
    :param m: module
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def regnet_init_weights(m: nn.Module) -> None:
    """
    RegNet weight initialization :)
    Reference Paper: https://arxiv.org/pdf/2003.13678
    :param m: module
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out = fan_out // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def convnext_init_weights(m: nn.Module) -> None:
    """
    ConvNext weight initialization :)
    Reference Paper: https://arxiv.org/pdf/2201.03545
    :param m: module
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def set_bn_momentum(model: nn.Module, momentum: float = 0.1) -> None:
    """
    change batch norm momentum in a model :)
    :param model: model
    :param momentum: new momentum
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def _fuse_modules(
        model: nn.Module, modules_to_fuse: Union[List[str], List[List[str]]], is_qat: Optional[bool], **kwargs: Any
):
    """
    fuse function for quantization model  :)
    :param model: model
    :param modules_to_fuse: list of modules to fuse
    :param is_qat: use qat
    :param kwargs: fuse_modules kwargs
    :return: fused model
    """
    if is_qat is None:
        is_qat = model.training
    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    return method(model, modules_to_fuse, **kwargs)


# ============
#  Classes
# =============

class ConvBNAct(nn.Module):
    """
    Conv + BN + ReLU :)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, groups: int = 1, dilation: int = 1, use_act: bool = True,
                 ) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size default is 3
        :param padding: number of padding in each side default is 1
        :param stride: kernel stride default is 1
        :param dilation: dilation default is 1
        :param groups: number of groups default is 1
        :param use_act: use activation function default is True
        """
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_act = use_act
        self.module_names = ["conv", "bn"]

        self.conv = nn.Sequential()
        self.conv.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                               stride=stride, dilation=dilation, groups=groups, bias=False))
        self.conv.add_module("bn", nn.BatchNorm2d(out_channels))
        if use_act:
            self.conv.add_module("act", nn.ReLU(inplace=True))
            self.module_names.append("act")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        return self.conv(x)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        """
        fuse conv and bn for inference or quantization :)
        :param is_qat: use quantization aware training or not
        :return:
        """

        _fuse_modules(self.conv, self.module_names, is_qat, inplace=True)


class AddLayer(nn.Module):
    """
    Custom Add Layer for quantization :)
    """

    def __init__(self, quantization: bool = False, use_relu: bool = False):
        """
        :param quantization: use quantization
        :param use_relu: use relu after add or no
        """
        super().__init__()
        self.use_relu = use_relu
        if quantization:
            self.add_layer = FloatFunctional()
        else:
            self.add_layer = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :param y: other feature maps
        :return: output feature maps
        """
        if self.add_layer is not None:
            if self.use_relu:
                x = self.add_layer.add_relu(x, y)
            else:
                x = self.add_layer.add(x, y)
        else:
            x = x + y
            if self.use_relu:
                x = nn.functional.relu(x)
        return x


class CatLayer(nn.Module):
    """
    Custom Cat Layer for quantization :)
    """

    def __init__(self, dim: int = 1, quantization: bool = False):
        """
        :param dim: concatenate dim
        :param quantization: if use quantization
        """
        super().__init__()
        self.dim = dim
        if quantization:
            self.cat_layer = FloatFunctional()
        else:
            self.cat_layer = None

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        if self.cat_layer is not None:
            x = self.cat_layer.cat(x, dim=self.dim)
        else:
            x = torch.cat(x, dim=self.dim)
        return x


class MulLayer(nn.Module):
    """
    Custom Multiple Layer for quantization :)
    """

    def __init__(self, quantization: bool = False):
        """
        :param dim: concatenate dim
        :param quantization: if use quantization
        """
        super().__init__()
        if quantization:
            self.mul_layer = FloatFunctional()
        else:
            self.mul_layer = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :param y: other feature maps
        :return: output feature maps
        """
        if self.mul_layer is not None:
            x = self.mul_layer.mul(x, y)
        else:
            x = x * y
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Networks :)
    Reference Paper: https://arxiv.org/pdf/1910.03151
    """

    def __init__(self, channels: int, scale: float = 0.25, quantization: bool = False) -> None:
        """
        :param channels: number of channels
        :param scale: a scale for calculate mid-channels (mid-channels = channels * scale) default is 0.25
        :param quantization: use QAT or no
        """
        super().__init__()
        self.reduce = nn.Conv2d(in_channels=channels, out_channels=int(channels * scale), kernel_size=1,
                                stride=1, bias=True)
        self.relu = nn.ReLU()
        self.expand = nn.Conv2d(in_channels=int(channels * scale), out_channels=channels, kernel_size=1,
                                stride=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.mul_layer = MulLayer(quantization=quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.avg_pool(x)
        x_ = self.reduce(x_)
        x_ = self.relu(x_)
        x_ = self.expand(x_)
        x_ = torch.sigmoid(x_)
        return self.mul_layer(x, x_.expand_as(x))


if __name__ == "__main__":
    pass
