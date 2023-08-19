from models.model_utils import *
from models.unet import *
from models.teacher_model import *
from models.bisenetv2 import *
from models.regseg import *

MODELS_COLLECTIONS = {
    "unet": Unet,
    "deeplabv3": DeepLabV3,
    "bisenetv2": BiSeNetV2,
    "regseg": RegSeg,
}
