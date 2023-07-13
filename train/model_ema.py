import torch
import torch.nn as nn
import copy
from typing import Optional


class ModelEma(nn.Module):
    """
    Model Exponential Moving Average :)
    Code from: "https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py"
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional = None) -> None:
        """
        :param: model: your model
        :param decay: decat parameter for teacher and student weights. default is 0.9999
        :param device: device for teacher model. default is None
        """
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model: nn.Module) -> None:
        """
        update function for student weights :)
        :param model: student model
        """
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(self._update_fn(ema_v, model_v))

    def _update_fn(self, e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        calculate teacher weight:)
        :param e: teacher weights
        :param m: student weights
        :return: ema model parameter
        """
        return self.decay * e + (1. - self.decay) * m

    def update(self, model) -> None:
        """
        update model ema
        :param model: model
        """
        self._update(model)

    def set(self, model) -> None:
        """
        update model ema
        :param model: model
        """
        self._update(model)
