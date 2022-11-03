import pytorch_lightning as pl
from typing import Any
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.functional import accuracy, precision, recall, auroc


class ModelBase(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float = 3e-4,
                 model_type: str = 'model-type',
                 input_shape: tuple = (256, 256),
                 max_epochs=None,
                 use_lr_scheduler=False,
                 user_pretrained_weight=False,
                 finetune_last_layer=False,
                 *args: Any,
                 **kwargs: Any):
        super(ModelBase, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.use_lr_scheduler = use_lr_scheduler
        self.user_pretrained_weight = user_pretrained_weight
        self.finetune_last_layer = finetune_last_layer

        self.model = self._setup_model(model_type=model_type)

        if finetune_last_layer:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        self.loss_func = torch.nn.BCELoss()

        self.example_input_array = torch.zeros((1, 3, input_shape[0], input_shape[1]), dtype=torch.float32)

    def _setup_model(self, model_type) -> torch.nn.Module:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.max_epochs)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def forward(self, x) -> Any:
        return torch.sigmoid(self.model(x))

    def _step(self, batch, prefix: str):
        x, y = batch
        y = y.unsqueeze(1)

        y_pred = self(x)
        loss = self.loss_func(y_pred, y.float())
        self.log(f'{prefix}_loss', loss)

        self.log_dict(
            {
                f'{prefix}_acc': accuracy(y_pred, y),
                f'{prefix}_precision': precision(y_pred, y),
                f'{prefix}_recall': recall(y_pred, y),
                f'{prefix}_auc': auroc(y_pred, y)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self._step(batch, prefix='train')

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return self._step(batch, prefix='val')

    def test_step(self, batch, batch_idx, *args, **kwargs):
        return self._step(batch, prefix='test')
