import torch.nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101

from .base import ModelBase


class ResNet(ModelBase):
    def _setup_models_mapping(self) -> dict:
        return {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
        }

    def _setup_model(self, model_type):
        model = self.models_mapping[model_type](weights='DEFAULT')
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, self.num_classes)
        return model
