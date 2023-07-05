# Pytorch Semantic Segmentation Models (TSSM)
#### This project contain datasets, transforms, models, losses, metrics, and training features 
#### for train semantic segmentation models with pytorch. 



| `Datasets` | `Transforms` | `Models`            | `Losses`      | `Metrics` | `Training Features`         |
|------------|--------------|---------------------|---------------|-----------|-----------------------------|
| cityscapes |              | unet                | cross-entropy | mIOU      | knowledge-distillation      |
|            |              | deeplabv3 (Pytorch) | OHEM          |           | float 16 training           |
|            |              |                     |               |           | quantization aware training |
|            |              |                     |               |           | EMA                         |

