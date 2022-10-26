from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import pil_to_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DogsCatsImages(VisionDataset):
    def __init__(self, root: str, image_filenames: List[str], *args, **kwargs):
        super(DogsCatsImages, self).__init__(root, *args, **kwargs)
        self.root = root
        self.image_filenames = image_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = str(self.image_filenames[index])

        if image_path.split('.')[0].lower() == 'cat':
            target = 0
        else:
            target = 1

        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = pil_to_tensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DogsCatsImagesDataModule(LightningDataModule):
    def __init__(self,
                 root: str = "path/to/dir",
                 batch_size: int = 32,
                 num_workers=0,
                 random_seed=None,
                 split_rate=(0.8, 0.1, 0.1),
                 train_transforms=None,
                 test_transforms=None,
                 val_transforms=None,
                 **kwargs):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_transforms = train_transforms
        self._test_transforms = test_transforms
        self._val_transforms = val_transforms

        self.image_filenames = list(Path(self.root).glob('**/*.jpg'))
        self.image_filenames = list(map(lambda x: str(x), self.image_filenames))

        np.random.seed(random_seed)
        np.random.shuffle(self.image_filenames)

        assert len(split_rate) == 3, 'Length of `split_rate` needs to be 3. (train_split, val_split, test_split)'
        train_split, val_split, test_split = split_rate
        samples_size = len(self.image_filenames)
        self.train_image_filenames = self.image_filenames[:int(samples_size * train_split)]
        self.val_image_filenames = self.image_filenames[int(samples_size * train_split):int(samples_size *
                                                                                            (train_split + val_split))]
        self.test_image_filenames = self.image_filenames[-int(samples_size * test_split):]

    def setup(self, stage: Optional[str] = None) -> None:
        self.stage = stage
        self.train_data = DogsCatsImages(root=self.root,
                                         image_filenames=self.train_image_filenames,
                                         transform=self._train_transforms)
        self.val_data = DogsCatsImages(root=self.root,
                                       image_filenames=self.train_image_filenames,
                                       transform=self._val_transforms)
        self.test_data = DogsCatsImages(root=self.root,
                                        image_filenames=self.train_image_filenames,
                                        transform=self._test_transforms)

    def train_dataloader(self):
        if self.stage == 'test':
            return DataLoader(self.train_data,
                              batch_size=self.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=self.num_workers)
        else:
            return DataLoader(self.train_data,
                              batch_size=self.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)

    def __repr__(self):
        return f'Embryo Phases Classification\n' + \
               f'#Labels: 1 = dog, 0 = cat\n' + \
               f'#Train: {len(self.train_image_filenames)}\n' + \
               f'#Val: {len(self.val_image_filenames)}\n' + \
               f'#Test: {len(self.test_image_filenames)}'
