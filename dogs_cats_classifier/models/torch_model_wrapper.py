import types

from .base import ModelBase
from torch.nn import Module, Sequential, Linear


def is_torch_builtin_models(model_type: str):
    model_type = model_type.lower()
    model_module = __import__('torchvision.models', fromlist=(model_type, ), level=0)
    if model_type in dir(model_module):
        model_func = getattr(model_module, model_type)
        return True if isinstance(model_func, types.FunctionType) else False

    return False


class TorchModelWrapper(ModelBase):
    def _setup_model(self, model_type) -> Module:
        # dynamic import module
        model_module = __import__('torchvision.models', fromlist=(model_type, ), level=0)
        model_func = getattr(model_module, model_type)
        model = model_func(weights='DEFAULT' if self.user_pretrained_weight else None)

        # get last layer's name
        layer_name = list(model.named_children())[-1][0]

        # check last layer
        last_layer = model.get_submodule(layer_name)
        if isinstance(last_layer, Sequential):
            last_layer = last_layer[-1]
        in_features = last_layer.in_features

        # replace the last layer
        last_layer = getattr(model, layer_name)
        if isinstance(last_layer, Sequential):
            last_layer[-1] = Linear(in_features=in_features, out_features=self.num_classes)
        else:
            setattr(model, layer_name, Linear(in_features=in_features, out_features=self.num_classes))

        return model
