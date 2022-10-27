import os.path
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.jit._script import ScriptModule
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from dogs_cats_classifier.data.dogs_cats_images import DogsCatsImages

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_single_image(model: ScriptModule, image_path: str, image_size: tuple, output_path: str):
    """
    predict single image and save the result
    Args:
        model: script jit model
        image_path: path to the image
        image_size: image size
        output_path: path to output result

    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    image_tensor = pil_to_tensor(image)
    image_tensor = image_tensor.div(255)
    image_tensor = image_tensor.unsqueeze(0)

    y = model(image_tensor)
    y = y.detach().numpy().flatten()[0]

    if y > 0.5:
        y_class = 'dog'
    else:
        y_class = 'cat'

    plt.imshow(image)
    plt.title(f'{y_class} ({y:.3f})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'result.png'))
    plt.show()


def predict_images(model: ScriptModule, image_folder: str, image_size: tuple, output_path: str, batch_size: int):
    """
    predict images in the folder
    Args:
        model: script jit model
        image_folder: the folder contains jpg images
        image_size: the input size of model
        output_path: path to output result csv file
        batch_size: inference batch size

    """
    image_filenames = list(Path(image_folder).glob('**/*.jpg'))
    image_filenames = list(map(lambda x: str(x), image_filenames))

    test_t = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])

    dataset = DogsCatsImages(root=image_folder, image_filenames=image_filenames, transform=test_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.to(device)

    predictions = []

    with tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x = x.to(device)

            y_pred = model(x)

            predictions.extend(y_pred.detach().to('cpu').numpy().tolist())

            pbar.update()

    predictions = np.array(predictions).flatten()
    class_predictions = (predictions > 0.5).astype(int)

    output = list(zip(image_filenames, class_predictions, predictions))
    output_df = pd.DataFrame(output, columns=['filenames', 'class', 'model_pred'])
    output_df['filenames'] = output_df['filenames'].apply(lambda x: os.path.basename(x))
    output_df['class'] = output_df['class'].apply(lambda x: 'cat' if x == 0 else 'dog')
    print(output_df)
    output_df.to_csv(os.path.join(output_path, 'results.csv'), index=False)


@click.command()
@click.option('--image-path', type=click.Path(exists=True), default=None, help='Path to the single image.')
@click.option('--image-folder', type=click.Path(exists=True), default=None, help='Path to the images folder')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to the model weight')
@click.option('--image-size', type=int, nargs=2, default=(256, 256), help='The size of input image. Default: (256,256)')
@click.option('--output-path', type=str, default='reports', help='Path to output model prediction. Default: reports')
@click.option('--batch-size', type=int, default=32)
def main(image_path, image_folder, model_path, image_size, output_path, batch_size):
    assert image_path is not None or image_folder is not None, '`image_path` or `image_folder` should not be None.'

    model = torch.jit.load(model_path)

    if image_path:
        predict_single_image(model=model, image_path=image_path, image_size=image_size, output_path=output_path)

    if image_folder:
        predict_images(model=model,
                       image_folder=image_folder,
                       image_size=image_size,
                       output_path=output_path,
                       batch_size=batch_size)


if __name__ == '__main__':
    main()
