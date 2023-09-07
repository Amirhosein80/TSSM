from datasets.utils import *
from datasets.cityscapes import *
from datasets.voc import *

DATASETS = {
    "cityscapes": (Cityscapes, CITYSCAPES_CLASSES),
    "voc": (Voc, VOC_CLASSES)
}