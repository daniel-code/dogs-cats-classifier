import os

import click
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms as T

from dogs_cats_classifier.data import DogsCatsImagesDataModule
from dogs_cats_classifier.utils import Evaluator


@click.command()
@click.option('-r', '--dataset-root', type=click.Path(exists=True), required=True, help='The root path to dataset.')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to the model weight')
@click.option('--batch-size', type=int, default=16, help='Batch size. Default: 16')
@click.option('--num-workers',
              type=int,
              default=0,
              help=f'Number of workers. #CPU of this machine: {os.cpu_count()}. Default: 0')
@click.option('--image-size', type=int, nargs=2, default=(256, 256), help='The size of input image. Default: (256,256)')
@click.option('--seed', type=int, default=168, help='Random seed of train/test split. Default: 168')
@click.option('--output-path',
              type=str,
              default='reports/figures',
              help='Path to output model weight. Default: reports/figures')
@click.option('--verbose', type=bool, is_flag=True, help='Display evaluation figures')
def main(dataset_root, batch_size, num_workers, image_size, seed, model_path, output_path, verbose):
    # check output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pl.seed_everything(seed, workers=True)

    test_t = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    datamodule = DogsCatsImagesDataModule(
        root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=test_t,
        test_transforms=test_t,
        val_transforms=test_t,
        random_seed=seed,
    )
    datamodule.setup(stage='test')

    model = torch.jit.load(model_path)

    evaluator = Evaluator(model=model, output_path=output_path)
    evaluator.evaluate(dataloader=datamodule.train_dataloader(), title='train', verbose=verbose)
    evaluator.evaluate(dataloader=datamodule.val_dataloader(), title='val', verbose=verbose)
    evaluator.evaluate(dataloader=datamodule.test_dataloader(), title='test', verbose=verbose)


if __name__ == '__main__':
    main()
