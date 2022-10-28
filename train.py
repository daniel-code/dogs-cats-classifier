import os

import click
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import LearningRateMonitor

from dogs_cats_classifier.data import DogsCatsImagesDataModule
from dogs_cats_classifier.models import ResNet, Swin, ResNext
from dogs_cats_classifier.utils import evaluate_model
from datetime import datetime


@click.command()
@click.option('-r', '--dataset-root', type=click.Path(exists=True), required=True, help='The root path to dataset.')
@click.option('--batch-size', type=int, default=16)
@click.option('--max-epochs', type=int, default=3)
@click.option('--num-workers',
              type=int,
              default=0,
              help=f'Number of workers. #CPU of this machine: {os.cpu_count()}. Default: 0')
@click.option('--image-size', type=int, nargs=2, default=(256, 256), help='The size of input image. Default: (256,256)')
@click.option('--fast-dev-run', type=bool, is_flag=True, help='Run fast develop loop of pytorch lightning')
@click.option('--seed', type=int, default=168, help='Random seed of train/test split. Default: 168')
@click.option('--model-type', type=str, default='resnet_50', help='The types of model. Default: resnet_50')
@click.option('--accelerator',
              type=str,
              default='auto',
              help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto") '
              'as well as custom accelerator instances. Default: auto')
@click.option('--devices', type=int, default=None)
@click.option('--output-path',
              type=str,
              default='model_weights',
              help='Path to output model weight. Default: model_weights')
@click.option('--use-lr-scheduler', type=bool, is_flag=True, help='Use OneCycleLR lr scheduler')
@click.option('--use-auto-augment', type=bool, is_flag=True, help='Use AutoAugmentPolicy')
def main(batch_size, max_epochs, num_workers, image_size, dataset_root, fast_dev_run, seed, model_type, accelerator,
         devices, output_path, use_lr_scheduler, use_auto_augment):
    exp_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(exp_time)
    output_path = os.path.join(output_path, f'{model_type}_{exp_time}')
    # check output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # set all random seed
    pl.seed_everything(seed, workers=True)

    # prepare dataset
    train_t = []
    if use_auto_augment:
        train_t.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))

    train_t.append(T.Resize(image_size))
    train_t.append(T.ToTensor())
    train_t = T.Compose(train_t)
    print('Training Data Augmentations')
    print(train_t)

    test_t = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])

    dogs_cats_datamodule = DogsCatsImagesDataModule(
        root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=train_t,
        test_transforms=test_t,
        val_transforms=test_t,
        random_seed=seed,
    )
    print(dogs_cats_datamodule)

    # prepare model
    if 'swin' in model_type:
        model = Swin
    elif 'resnext' in model_type:
        model = ResNext
    elif 'resnet' in model_type:
        model = ResNet
    else:
        raise ValueError(f'{model_type} is not available.')

    model = model(num_classes=1,
                  model_type=model_type,
                  input_shape=image_size,
                  max_epochs=max_epochs,
                  use_lr_scheduler=use_lr_scheduler)

    # use pytorch lightning trainer
    trainer = pl.Trainer(
        default_root_dir=output_path,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        auto_scale_batch_size=True,
        fast_dev_run=fast_dev_run,
        callbacks=[LearningRateMonitor(logging_interval='epoch')],
    )
    # training
    trainer.fit(model, datamodule=dogs_cats_datamodule)
    # testing
    trainer.test(model, datamodule=dogs_cats_datamodule)

    # save for use in production environment
    script_model = model.to_torchscript()
    torch.jit.save(script_model, os.path.join(output_path, 'model.pt'))

    # evaluation
    dogs_cats_datamodule.setup()
    evaluate_model(model=model,
                   dataloader=dogs_cats_datamodule.test_dataloader(),
                   title=f'{model_type}_test',
                   output_path=output_path,
                   verbose=False)


if __name__ == '__main__':
    main()
