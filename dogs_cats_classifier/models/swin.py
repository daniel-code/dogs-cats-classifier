from torch.nn import Linear

from .base import ModelBase
from torchvision.models import swin_b, swin_t, swin_s


class Swin(ModelBase):
    def _setup_models_mapping(self) -> dict:
        return {
            'swin_b': swin_b,
            'swin_t': swin_t,
            'swin_s': swin_s,
        }

    def _setup_model(self, model_type):
        model = self.models_mapping[model_type](weights='DEFAULT' if self.user_pretrained_weight else None)
        in_feature = model.head.in_features
        model.head = Linear(in_feature, self.num_classes)
        return model
