from collections import OrderedDict
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from models.model_utils import ConvBNAct, AddLayer, CatLayer, MulLayer


class DetailBranch(nn.Module):
    """
    Detail Branch of BiSeNet V2
    """

    def __init__(self, channels: int) -> None:
        """
        :param channels: number of channels
        """
        super().__init__()
        self.stage2 = nn.Sequential(
            ConvBNAct(3, channels, kernel_size=3, stride=2, padding=1),
            ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1),
        )

        self.stage4 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, stride=2, padding=1),
            ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1),
        )

        self.stage8 = nn.Sequential(
            ConvBNAct(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            ConvBNAct(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            ConvBNAct(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = self.stage2(x)
        x = self.stage4(x)
        x = self.stage8(x)
        return x


class Stem(nn.Module):
    """
    Stem of BiSeNet V2
    """

    def __init__(self, channels: int, quantization: bool = False) -> None:
        """
        :param channels: number of channels
        :param quantization: use quantization
        """
        super().__init__()
        self.conv1 = ConvBNAct(3, channels, kernel_size=3, stride=2, padding=1)
        self.branch1 = nn.Sequential(
            ConvBNAct(channels, channels // 2, 1),
            ConvBNAct(channels // 2, channels, kernel_size=3, stride=2, padding=1)
        )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNAct(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.cat = CatLayer(quantization=quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = self.conv1(x)
        x = self.cat([self.branch1(x), self.branch2(x)])
        x = self.conv2(x)
        return x


class GatherAndExpansion(nn.Module):
    """
    Gather & Expansion block BiSeNet V2
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, quantization: bool = False) -> None:
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param quantization: use quantization
        :param stride: conv stride
        """
        super().__init__()
        self.stride = stride

        self.conv1 = ConvBNAct(in_channels, out_channels * 6, kernel_size=3,
                               stride=1, padding=1, use_act=True)
        self.conv2 = ConvBNAct(out_channels * 6, out_channels * 6, kernel_size=3,
                               stride=1, padding=1, use_act=False, groups=out_channels * 6)
        self.conv3 = ConvBNAct(out_channels * 6, out_channels, kernel_size=1,
                               stride=1, padding=0, use_act=False)
        self.add = AddLayer(quantization=quantization, use_relu=True)

        if stride != 1:
            self.conv_stride = ConvBNAct(out_channels * 6, out_channels * 6, kernel_size=3,
                                         stride=stride, padding=1, use_act=False, groups=out_channels * 6)
            self.shortcut = nn.Sequential(
                ConvBNAct(in_channels, in_channels, kernel_size=3,
                          stride=stride, padding=1, groups=in_channels, use_act=False),
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

        if self.conv_stride is not None:
            y = self.conv_stride(y)
        if self.shortcut is not None:
            x = self.shortcut(x)

        y = self.conv2(y)
        y = self.conv3(y)
        return self.add(x, y)


class ContextEmbedding(nn.Module):
    """
    Gather & Expansion block BiSeNet V2
    """

    def __init__(self, channels: int, quantization: bool = False) -> None:
        """
        :param channels: number of input channels
        :param quantization: use quantization
        """
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = ConvBNAct(channels, channels, kernel_size=1,
                               stride=1, padding=0, use_act=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1)
        self.add = AddLayer(quantization=quantization, use_relu=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        y = self.avg(x)
        y = self.conv1(y)
        y = self.add(x, y)
        return self.conv2(y)


class BilateralGuidedAggregation(nn.Module):
    """
    Bilateral Guided Aggregation block BiSeNet V2
    """

    def __init__(self, channels: int, quantization: bool = False) -> None:
        """
        :param channels: number of input channels
        :param quantization: use quantization
        """
        super().__init__()
        self.detail_branch1_1 = ConvBNAct(channels, channels, kernel_size=3,
                                          stride=1, padding=1, use_act=False, groups=channels)
        self.detail_branch1_2 = nn.Conv2d(channels, channels, kernel_size=1)

        self.detail_branch2_1 = ConvBNAct(channels, channels, kernel_size=3,
                                          stride=2, padding=1, use_act=False)
        self.detail_branch2_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.semantic_branch1_1 = ConvBNAct(channels, channels, kernel_size=3,
                                            stride=1, padding=1, use_act=False, groups=channels)
        self.semantic_branch1_2 = nn.Conv2d(channels, channels, kernel_size=1)

        self.semantic_branch2_1 = ConvBNAct(channels, channels, kernel_size=3,
                                            stride=1, padding=1, use_act=False)
        self.semantic_branch2_2 = nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False)

        self.up_sample4 = nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False)
        self.proj = ConvBNAct(channels, channels, kernel_size=3,
                              stride=1, padding=1, use_act=False)

        self.mul1 = MulLayer(quantization=quantization)
        self.mul2 = MulLayer(quantization=quantization)
        self.add = AddLayer(quantization=quantization)

        self.sigmoid = nn.Sigmoid()

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param detail: input feature maps of detail branch
        :param semantic: input feature maps of semantic branch
        :return: output feature maps
        """
        detail_1 = self.detail_branch1_2(self.detail_branch1_1(detail))
        detail_2 = self.detail_branch2_2(self.detail_branch2_1(detail))

        semantic_1 = self.semantic_branch1_2(self.semantic_branch1_1(semantic))
        semantic_2 = self.semantic_branch2_2(self.semantic_branch2_1(semantic))

        semantic_1 = self.sigmoid(semantic_1)
        semantic_2 = self.sigmoid(semantic_2)

        detail = self.mul1(detail_1, semantic_2)
        semantic = self.mul1(detail_2, semantic_1)

        return self.proj(self.add(detail, self.up_sample4(semantic)))


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


class SemanticBranch(nn.Module):
    """
    Semantic Branch of BiSeNet V2
    """

    def __init__(self, channels: int, quantization: bool = False) -> None:
        """
        :param channels: number of input channels
        :param quantization: use quantization
        """
        super().__init__()
        self.stem = Stem(channels, quantization=quantization)

        self.layer8 = nn.Sequential(
            GatherAndExpansion(channels, channels * 2, stride=2, quantization=quantization),
            GatherAndExpansion(channels * 2, channels * 2, stride=1, quantization=quantization),
        )

        self.layer16 = nn.Sequential(
            GatherAndExpansion(channels * 2, channels * 4, stride=2, quantization=quantization),
            GatherAndExpansion(channels * 4, channels * 4, stride=1, quantization=quantization),
        )

        self.layer32 = nn.Sequential(
            GatherAndExpansion(channels * 4, channels * 8, stride=2, quantization=quantization),
            GatherAndExpansion(channels * 8, channels * 8, stride=1, quantization=quantization),
            GatherAndExpansion(channels * 8, channels * 8, stride=1, quantization=quantization),
            GatherAndExpansion(channels * 8, channels * 8, stride=1, quantization=quantization),
        )

        self.proj = ContextEmbedding(channels * 8, quantization=quantization)

    def forward(self, x: torch.Tensor):
        """
        forward function :)
        :param x: input feature maps
        :return: dict of feature maps
        """
        out = OrderedDict()
        x = self.stem(x)
        out["4"] = x

        x = self.layer8(x)
        out["8"] = x

        x = self.layer16(x)
        out["16"] = x

        x = self.layer32(x)
        out["32"] = x

        x = self.proj(x)
        out["se"] = x
        return out


class BiSeNetV2(nn.Module):
    """
    BiSeNet V2
    """

    def __init__(self, num_classes: int, quantization: bool = False, inference: bool = False) -> None:
        """
        :param num_classes: number of classes
        :param quantization: use quantization
        """
        super().__init__()

        self.inference = inference
        if quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

        self.detail_branch = DetailBranch(64)
        self.semantic_branch = SemanticBranch(16, quantization=quantization)
        self.bga = BilateralGuidedAggregation(128, quantization=quantization)

        self.head = Head(128, num_classes)
        self.aux8 = Head(32, num_classes)
        self.aux16 = Head(64, num_classes)
        self.aux32 = Head(128, num_classes)

    def forward(self, x: torch.Tensor):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps or OrderedDict
        """
        output = OrderedDict()
        input_shape = x.shape[-2:]

        if self.quant is not None:
            x = self.quant(x)

        semantic_features = self.semantic_branch(x)
        detail_feature = self.detail_branch(x)
        x = self.bga(detail=detail_feature, semantic=semantic_features["se"])
        x = nn.functional.interpolate(self.head(x), size=input_shape, mode="bilinear", align_corners=False)
        output["out"] = x

        if self.dequant is not None:
            output["out"] = self.dequant(output["out"])

        if not self.inference:
            output["aux8"] = nn.functional.interpolate(self.aux8(semantic_features["8"]), size=input_shape,
                                                       mode="bilinear", align_corners=False)
            output["aux16"] = nn.functional.interpolate(self.aux16(semantic_features["16"]), size=input_shape,
                                                        mode="bilinear", align_corners=False)
            output["aux32"] = nn.functional.interpolate(self.aux32(semantic_features["32"]), size=input_shape,
                                                        mode="bilinear", align_corners=False)

            if self.dequant is not None:
                output["aux8"] = self.dequant(output["aux8"])
                output["aux16"] = self.dequant(output["aux16"])
                output["aux32"] = self.dequant(output["aux32"])

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
    # model = DeepLabV3()
    import torchinfo

    #
    dump = torch.randn(2, 3, 512, 1024).cuda()
    deeplab = BiSeNetV2(19).cuda()
    out_dump = deeplab(dump)
    print(torchinfo.summary(deeplab, (8, 3, 768, 768), device="cuda"))
