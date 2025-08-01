"""`datasets` module provides convenient preprocessing and loading of data."""

import os

import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np

from sklearn.model_selection import train_test_split

from PIL import Image

from data_research import DataFolder, ImageType

BATCH_SIZE = 64
IMAGENET_MEAN_AND_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class Dataset(data.Dataset):
    """`Dataset` performs transforms to images beforehand."""

    PATH = ""
    FILES = []

    def __init__(self, transform, label_maker):
        super().__init__()
        self.transform = transform
        self.label_maker = label_maker

    def __len__(self):
        return len(self.FILES)

    def __getitem__(self, index):
        file = self.FILES[index]
        with Image.open(os.path.join(self.PATH, file)) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, self.label_maker(file)


FIT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),

    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=(0.8, 1.0), contrast=1, saturation=0, hue=0.2),

    transforms.ToTensor(),

    transforms.Normalize(*IMAGENET_MEAN_AND_STD)
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(*IMAGENET_MEAN_AND_STD)
])


class TrainingDataset(Dataset):
    """`TrainingDataset` performs transforms to training images beforehand."""

    PATH = str(DataFolder.TRAIN)
    FILES = DataFolder.TRAIN.files()

    def __init__(self, transform):
        super().__init__(transform, lambda file: ImageType.from_file(file).label())

    @staticmethod
    def split(fit_transform, val_transform, val_size: float) -> (Dataset, Dataset):
        """Splits fitting data to obtain data specifically for fitting and data for validation."""

        dataset = TrainingDataset(lambda x: x)

        fit_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=val_size,
            stratify=[ImageType.from_file(file) for file in dataset.FILES],
            random_state=42
        )

        fit_dataset = data.Subset(TrainingDataset(fit_transform), fit_indices)
        val_dataset = data.Subset(TrainingDataset(val_transform), val_indices)
        return fit_dataset, val_dataset


VAL_SIZE = 0.2
FIT_DATASET, VAL_DATASET = TrainingDataset.split(FIT_TRANSFORM, VAL_TRANSFORM, VAL_SIZE)

FIT_DATALOADER = data.DataLoader(
    FIT_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)
VAL_DATALOADER = data.DataLoader(
    VAL_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(*IMAGENET_MEAN_AND_STD)
])


class TestingDataset(Dataset):
    """`TestingDataset` performs transforms to testing images beforehand."""

    PATH = str(DataFolder.TEST)
    FILES = sorted(DataFolder.TEST.files())

    def __init__(self):
        super().__init__(TEST_TRANSFORM, lambda x: x)


TEST_DATASET = TestingDataset()
TEST_DATALOADER = data.DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

if __name__ == "__main__":
    def show_images(images, denormalize=False):
        """Shows several images from their tensor form."""

        import matplotlib.pyplot as plt

        if denormalize:
            mean, std = IMAGENET_MEAN_AND_STD
            images = np.clip((std * images.numpy().transpose((0, 2, 3, 1)) + mean), 0, 1)
        else:
            images = images.numpy().transpose((0, 2, 3, 1))

        _, axes = plt.subplots(int(BATCH_SIZE ** 0.5), int(BATCH_SIZE ** 0.5), figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.axis("off")
        plt.show()


    fit_data = iter(FIT_DATALOADER)
    show_images(next(fit_data)[0], denormalize=True)

    test_data = iter(TEST_DATALOADER)
    show_images(next(test_data)[0], denormalize=True)
  
