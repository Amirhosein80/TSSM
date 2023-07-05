import os

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from typing import Tuple
from torch import Tensor

from configs.cityscapes import LABEL_MAPPING, CLASSES, PHASES, MEAN, STD
from .base import BaseDataset
from .utils import unnormalize_image


def convert_labels(label: np.ndarray) -> np.ndarray:
    """
    convert all classes of cityscapes to 19 classes :)
    :param label: old label
    :return: new label
    """
    temp = label.copy()
    for k, v in LABEL_MAPPING.items():
        label[temp == k] = v
    return label


def show_image_city(image: PIL.Image.Image) -> None:
    """
    plot image :)
    :param image: image
    """
    plt.clf()
    plt.imshow(image)
    plt.show()


def show_numpy_mask_city(label: PIL.Image.Image) -> None:
    """
    plot mask + guide :)
    """
    colors = []
    labels = []
    for v in CLASSES.values():
        colors.append(list(v["color"]))
        labels.append(v["name"])
    colors = np.array(colors, dtype=np.uint8)
    handles = [Rectangle((0, 0), 1, 1, color=_c / 255) for _c in colors]
    label.putpalette(colors)
    plt.imshow(label)
    plt.show()

    plt.imshow(np.ones((1, 1, 3)))
    plt.legend(handles, labels, loc="center")
    plt.axis('off')
    plt.show()


class Cityscapes(BaseDataset):
    """
    Cityscapes Dataset Class :)
    """

    def __init__(self, phase, root, transforms=None, debug: bool = False) -> None:
        """
        :param phase: train or validation
        :param root: dataset directory
        :param transforms: data augmentations for change datas
        :param debug: watch image & mask
        """
        super().__init__()
        assert phase in PHASES, f"{phase} not in {PHASES} :)"
        if not os.path.isfile("./train_val_paths.json"):
            self.create_json_paths_cityscapes(root, PHASES)
        self.files = self.read_json_file(phase)
        self.debug = debug
        self.transforms = transforms

    def __getitem__(self, idx) -> Tuple[Tensor | PIL.Image.Image, Tensor | PIL.Image.Image]:
        image, mask = self.files[idx]["Image"], self.files[idx]["Mask"]
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = convert_labels(mask)

        image = PIL.Image.fromarray(image)
        mask = PIL.Image.fromarray(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        if self.debug:
            show_image_city(PIL.Image.fromarray(unnormalize_image(image, mean=MEAN, std=STD)
                                                .detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)))
            show_numpy_mask_city(PIL.Image.fromarray(mask.detach().cpu().numpy().astype(np.uint8)))

        return image, mask

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    pass
