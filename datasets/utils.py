from typing import List, Tuple
from imutils import auto_canny

import numpy as np
import torch
import cv2


def unnormalize_image(img: torch.Tensor, mean: List, std: List) -> torch.Tensor:
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


def cat_list(images: List[torch.Tensor], fill_value: int = 0) -> torch.Tensor:
    """
    concatenate image in batch dim and make sure that their sizes are equal :)
    :param images: list of images
    :param fill_value: fill value for padding
    :return: a batch of image
    """
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def extract_edges(labels: torch.Tensor) -> torch.Tensor:
    """
    extract edges from labels for SAB Loss :)
    :param labels: segment labels
    :return: edge labels
    """
    kernel = np.ones((11, 11), np.uint8)
    edges = []
    for label in labels:
        edge = auto_canny(label.detach().cpu().numpy().astype(np.uint8))
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = np.where(edge == 255, label.detach().cpu().numpy(), 255)
        edges.append(torch.tensor(edge, dtype=torch.int64).unsqueeze(0))
    return torch.cat(edges, dim=0)


def collate_fn(batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    collate function for train :)
    :param batch: datas batch
    :return: images_batch, labels_batch
    """
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)

    return batched_imgs, batched_targets

