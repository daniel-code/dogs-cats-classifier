import os

import click
import pandas as pd
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
@click.option('--output-path', type=str, default='reports', help='Path to output model prediction. Default: reports')
def main(dataset_root, batch_size, num_workers, image_size, seed, model_path, output_path):
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

    ground_truth, predictions, class_predictions = evaluator.evaluate(datamodule.test_dataloader(), title='test')
    df = zip(datamodule.test_image_filenames, ground_truth, predictions, class_predictions)
    df = pd.DataFrame(df, columns=['filename', 'ground_truth', 'prediction', 'class_predictions'])
    print(df)
    df.to_csv(os.path.join(output_path, 'output.csv'), index=False)

    evaluator.plot_images(df[df['ground_truth'] != df['class_predictions']],
                          title='test',
                          output_path=output_path,
                          verbose=True)


if __name__ == '__main__':
    main()
