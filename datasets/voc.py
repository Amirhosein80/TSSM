import os
from typing import Tuple, Optional, Callable

import PIL
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from torch import Tensor

from datasets.base import BaseDataset

PHASES = ["train", "val", "trainval"]

VOC_CLASSES = {
    0: {"name": "__background__", "color": (0, 0, 0)},
    1: {"name": "Aeroplane", "color": (128, 0, 0)},
    2: {"name": "Bicycle", "color": (0, 128, 0)},
    3: {"name": "Bird", "color": (128, 128, 0)}, 4: {"name": "Boat", "color": (0, 0, 128)},
    5: {"name": "Bottle", "color": (128, 0, 128)}, 6: {"name": "Bus", "color": (0, 128, 128)},
    7: {"name": "Car", "color": (128, 128, 128)},
    8: {"name": "Cat", "color": (64, 0, 0)},
    9: {"name": "Chair", "color": (192, 0, 0)},
    10: {"name": "Cow", "color": (64, 128, 0)},
    11: {"name": "Diningtable", "color": (192, 128, 0)}, 12: {"name": "Dog", "color": (64, 0, 128)},
    13: {"name": "Horse", "color": (192, 0, 128)}, 14: {"name": "Motorbike", "color": (64, 128, 128)},
    15: {"name": "Person", "color": (192, 128, 128)}, 16: {"name": "Pottedplant", "color": (0, 64, 0)},
    17: {"name": "Sheep", "color": (128, 64, 0)},
    18: {"name": "Sofa", "color": (0, 192, 0)},
    19: {"name": "Train", "color": (128, 192, 0)},
    20: {"name": "Tvmonitor", "color": (0, 64, 128)},
}


def show_image_voc(image: PIL.Image.Image) -> None:
    """
    plot image :)
    :param image: image
    """
    plt.clf()
    plt.imshow(image)
    plt.show()


def show_numpy_mask_voc(label: PIL.Image.Image) -> None:
    """
    plot mask + guide :)
    :param label: label
    """
    colors = []
    labels = []
    for v in VOC_CLASSES.values():
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


class Voc(BaseDataset):
    """
    Voc Dataset Class :)
    Link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#introduction
    """

    def __init__(self, phase, root, transforms: Optional[Callable] = None, version: int = 1) -> None:
        """
        :param phase: train or validation
        :param root: datasets directory
        :param transforms: data augmentations for change datas
        :param version: select dataset version *(version 1 is original version)
        """
        super().__init__()
        assert phase in PHASES, f"{phase} not in {PHASES} :)"
        if version == 1:
            load_fn = self.read_json_file_voc
            create_fn = self.create_json_paths_voc
            json_file = "./train_val_paths_voc.json"
        else:
            raise NotImplemented
        if phase != "test":
            if not os.path.isfile(json_file):
                create_fn(root, PHASES)
            self.files = load_fn(phase)
        self.transforms = transforms

    def __getitem__(self, idx) -> Tuple[Tensor | PIL.Image.Image, Tensor | PIL.Image.Image]:
        image, mask = self.files[idx]["Image"], self.files[idx]["Mask"]

        image = PIL.Image.open(image).convert("RGB")
        mask = PIL.Image.open(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.files)

    def get_test_image(self, image_path):
        image = PIL.Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image, _ = self.transforms(image, None)
        return image


if __name__ == "__main__":
    train_ds = Voc(phase="train", root="../data/Voc")
