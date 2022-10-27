import torch.nn
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d

from .base import ModelBase


class ResNext(ModelBase):
    def _setup_models_mapping(self) -> dict:
        return {
            'resnext50_32x4d': resnext50_32x4d,
            'resnext101_32x8d': resnext101_32x8d,
        }

    def _setup_model(self, model_type):
        model = self.models_mapping[model_type](weights='DEFAULT')
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, self.num_classes)
        return model
