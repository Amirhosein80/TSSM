from models.model_utils import *
from models.unet import *
from models.teacher_model import *

MODELS_COLLECTIONS = {
    "unet": Unet,
    "deeplabv3": DeepLabV3
}
