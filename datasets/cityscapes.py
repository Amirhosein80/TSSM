import os
from typing import Tuple, Optional, Callable

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from torch import Tensor

from datasets.base import BaseDataset

PHASES = ["train", "val"]
CLASSES = {0: {"name": "road", "color": (128, 64, 128)}, 1: {"name": "sidewalk", "color": (244, 35, 232)},
           2: {"name": "building", "color": (70, 70, 70)}, 3: {"name": "wall", "color": (102, 102, 156)},
           4: {"name": "fence", "color": (190, 153, 153)}, 5: {"name": "pole", "color": (153, 153, 153)},
           6: {"name": "traffic light", "color": (250, 170, 30)},
           7: {"name": "traffic sign", "color": (220, 220, 0)},
           8: {"name": "vegetation", "color": (107, 142, 35)},
           9: {"name": "terrain", "color": (152, 251, 152)},
           10: {"name": "sky", "color": (70, 130, 180)}, 11: {"name": "person", "color": (220, 20, 60)},
           12: {"name": "rider", "color": (255, 0, 0)}, 13: {"name": "car", "color": (0, 0, 142)},
           14: {"name": "truck", "color": (0, 0, 70)}, 15: {"name": "bus", "color": (0, 60, 100)},
           16: {"name": "train", "color": (0, 80, 100)},
           17: {"name": "motorcycle", "color": (0, 0, 230)},
           18: {"name": "bicycle", "color": (119, 11, 32)}, }


def convert_labels(label: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    """
    convert all classes of cityscapes to 19 classes :)
    :param label: old label
    :param ignore_label: ignore variable for unselected labels
    :return: new label
    """
    label_mapping = {-1: ignore_label, 0: ignore_label,
                     1: ignore_label, 2: ignore_label,
                     3: ignore_label, 4: ignore_label,
                     5: ignore_label, 6: ignore_label,
                     7: 0, 8: 1, 9: ignore_label,
                     10: ignore_label, 11: 2, 12: 3,
                     13: 4, 14: ignore_label,
                     15: ignore_label,
                     16: ignore_label, 17: 5,
                     18: ignore_label,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15,
                     29: ignore_label, 30: ignore_label,
                     31: 16, 32: 17, 33: 18}
    temp = label.copy()
    for k, v in label_mapping.items():
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
    :param label: label
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
    Link: "https://www.cityscapes-dataset.com/
    """

    def __init__(self, phase, root, transforms: Optional[Callable] = None) -> None:
        """
        :param phase: train or validation
        :param root: datasets directory
        :param transforms: data augmentations for change datas
        """
        super().__init__()
        assert phase in PHASES, f"{phase} not in {PHASES} :)"
        if not os.path.isfile("./train_val_paths_cityscapes.json"):
            self.create_json_paths_cityscapes(root, PHASES)
        self.files = self.read_json_file(phase)
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

        return image, mask

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    pass
