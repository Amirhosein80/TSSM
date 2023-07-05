import torch
from typing import Tuple


def unnormalize_image(img: torch.Tensor, mean: Tuple, std: Tuple) -> torch.Tensor:
    """
    return un-normalized tensor :)
    :param img: normalized image
    :param mean: image mean
    :param std: image std
    :return: un-normalize image
    """
    un_img = img.mul_(torch.tensor(std).reshape(-1, 1, 1)).add_(torch.tensor(mean).reshape(-1, 1, 1))
    un_img = un_img.mul_(255).to(torch.uint8)
    return un_img
