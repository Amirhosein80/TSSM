import glob
import json
import os
from typing import Dict, List

import tqdm.autonotebook as tqdm
from torch.utils.data import Dataset
import cv2
import numpy as np



class BaseDataset(Dataset):
    """
    Basic Class for all datasets to make a json file for train files & validation files :)
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def convert_labels_cityscapes(self, label: np.ndarray, ignore_label: int = 255) -> np.ndarray:
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

    def create_json_paths_cityscapes(self, root: str, phases: List) -> None:
        """
        if json file doesn't exist create one file for cityscapes datasets:)
        :param root: datasets directory
        :param phases: train or validation list
        """
        json_dict = {phase: [] for phase in phases}
        for key, value in json_dict.items():
            images_path = glob.glob(root + "/leftImg8bit/" + key + "/*/*.png")
            loop = tqdm.tqdm(images_path, total=len(images_path))
            for step, path in enumerate(loop):
                name = path.split("\\")[-1].split("_leftImg8bit")[0]
                try:
                    mask = glob.glob(root + "/gtFine/" + f"*/*/{name}*labelIds.png")[0]
                except:
                    print(f"Can't found mask file for {name}")
                    continue
                value.append({"Name": name,
                              "Image": path,
                              "Mask": mask})
                loop.set_description(f"Creating JSON file for datasets files paths. Phase: {key}, Step: {step}")
        with open("train_val_paths_cityscapes.json", "w") as openfile:
            json.dump(json_dict, openfile)

    def create_json_paths_cityscapes_v2(self, root: str, phases: List) -> None:
        """
        if json file doesn't exist create one file for cityscapes datasets but also removed many background masks:)
        :param root: datasets directory
        :param phases: train or validation list
        """
        json_dict = {phase: [] for phase in phases}
        for key, value in json_dict.items():
            images_path = glob.glob(root + "/leftImg8bit/" + key + "/*/*.png")
            loop = tqdm.tqdm(images_path, total=len(images_path))
            for step, path in enumerate(loop):
                name = path.split("\\")[-1].split("_leftImg8bit")[0]
                try:
                    mask = glob.glob(root + "/gtFine/" + f"*/*/{name}*labelIds.png")[0]
                except:
                    print(f"Can't found mask file for {name}")
                    continue
                m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                m = self.convert_labels_cityscapes(m)
                if (((m == 255).sum() / m.size) * 100) <= 20 or key != "train":
                    value.append({"Name": name,
                                  "Image": path,
                                  "Mask": mask})
                loop.set_description(f"Creating JSON file for datasets files paths. Phase: {key}, Step: {step}")
        with open("train_val_paths_cityscapes_v2.json", "w") as openfile:
            json.dump(json_dict, openfile)

    def read_json_file_cityscapes(self, phase: str) -> Dict:
        """
        read json datas of phase :)
        :param phase: train or validation
        :return dictionary of files paths
        """
        with open('train_val_paths_cityscapes.json', 'r') as openfile:
            json_object = json.load(openfile)
        return json_object[phase]

    def read_json_file_cityscapes_v2(self, phase: str) -> Dict:
        """
        read json datas of phase :)
        :param phase: train or validation
        :return dictionary of files paths
        """
        with open('train_val_paths_cityscapes_v2.json', 'r') as openfile:
            json_object = json.load(openfile)
        return json_object[phase]

    def create_json_paths_voc(self, root: str, phases: List) -> None:
        """
        if json file doesn't exist create one file for voc datasets:)
        :param root: datasets directory
        :param phases: train or validation list
        """
        voc_root = os.path.join(root, 'VOCdevkit/VOC2012')
        splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        image_dir = os.path.join(voc_root, "JPEGImages")
        target_dir = os.path.join(voc_root, "SegmentationClass")
        json_dict = {phase: [] for phase in phases}
        for key, value in json_dict.items():
            split_f = os.path.join(splits_dir, key + ".txt")
            with open(os.path.join(split_f)) as f:
                file_names = [x.strip() for x in f.readlines()]
            loop = tqdm.tqdm(file_names, total=len(file_names))
            for step, name in enumerate(loop):
                value.append({
                    "Name": name,
                    "Image": os.path.join(image_dir, name + ".jpg"),
                    "Mask": os.path.join(target_dir, name + ".png")
                })
                loop.set_description(f"Creating JSON file for datasets files paths. Phase: {key}, Step: {step}")
        with open("train_val_paths_voc.json", "w") as openfile:
            json.dump(json_dict, openfile)

    def read_json_file_voc(self, phase: str) -> Dict:
        """
        read json datas of phase :)
        :param phase: train or validation
        :return dictionary of files paths
        """
        with open('train_val_paths_voc.json', 'r') as openfile:
            json_object = json.load(openfile)
        return json_object[phase]
