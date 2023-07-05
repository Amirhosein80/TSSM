import glob
import json
from typing import Dict

import tqdm.autonotebook as tqdm
from torch.utils.data import Dataset


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

    def create_json_paths_cityscapes(self, root: str, phases: str) -> None:
        """
        if json file doesn't exist create one file for cityscapes dataset:)
        :param root: dataset directory
        :param phases: train or validation
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
                loop.set_description(f"Creating JSON file for dataset files paths. Phase: {key}, Step: {step}")
        with open("train_val_paths.json", "w") as openfile:
            json.dump(json_dict, openfile)

    def read_json_file(self, phase: str) -> Dict:
        """
        read json datas of phase
        :param phase: train or validation
        :return dictionary of files paths
        """
        with open('train_val_paths.json', 'r') as openfile:
            json_object = json.load(openfile)
        return json_object[phase]
