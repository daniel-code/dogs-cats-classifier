from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d

from .resnet import ResNet


class ResNext(ResNet):
    def _setup_models_mapping(self) -> dict:
        return {
            'resnext50_32x4d': resnext50_32x4d,
            'resnext101_32x8d': resnext101_32x8d,
        }
