import math
from typing import List, Tuple, Union, Any

import torch
import torchvision.transforms.functional as trf
from torch import Tensor
from torchvision.transforms import InterpolationMode


def _apply_op(
        img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode,
        fill: Union[int, List[int], float]
):
    if op_name == "ShearX":
        img = trf.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        img = trf.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = trf.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = trf.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = trf.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = trf.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = trf.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = trf.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = trf.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = trf.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = trf.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = trf.autocontrast(img)
    elif op_name == "Equalize":
        img = trf.equalize(img)
    elif op_name == "Invert":
        img = trf.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def _augmentation_space_rand_aug(num_bins: int, image_size: Tuple[int, int])\
        -> tuple[dict[str | Any, tuple[Tensor, bool] | Any], list[str]]:
    affine_ops = [
        "Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"
    ]
    aug_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
    }
    return aug_space, affine_ops


def _augmentation_space_tri_aug(num_bins: int) -> tuple[dict[str | Any, tuple[Tensor, bool] | Any], list[str]]:
    aug_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
        "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
        "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
        "Color": (torch.linspace(0.0, 0.99, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
    }
    affine_ops = [
        "Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"
    ]
    return aug_space, affine_ops
