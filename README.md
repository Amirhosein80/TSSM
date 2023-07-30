# Pytorch Semantic Segmentation Models (TSSM)
#### This project contain datasets, transforms, models, losses, metrics, and training features 
#### for train semantic segmentation models with pytorch. 



| `Datasets` | `Transforms` | `Models`                      | `Losses`      | `Metrics` | `Training Features`         | `Training Visualization` |
|------------|--------------|-------------------------------|---------------|-----------|-----------------------------|--------------------------|
| cityscapes | Rand Aug     | UNet                          | cross-entropy | mIOU      | EMA                         | Native                   |
|            | Trivial Aug  | DeepLabV3++<br/>(Torchvision) | OHEM          |           | float 16 training           | TensorBoard              |
|            |              |                               |               |           | quantization aware training | WandB                    |

