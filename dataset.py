import os

from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms

import numpy as np

from sklearn.model_selection import train_test_split

from PIL import Image

from data_research import DataFolder, ImageType


class TrainingDataset(Dataset):
    """`TrainingDataset` allows performing transforms to images beforehand."""

    FILES = DataFolder.TRAIN.files()

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __len__(self):
        return len(TrainingDataset.FILES)

    def __getitem__(self, index):
        file = TrainingDataset.FILES[index]
        with Image.open(os.path.join(DataFolder.TRAIN, file)) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, ImageType.of_file(file).label()

    @staticmethod
    def split(fit_transform, val_transform, val_size: float) -> (Dataset, Dataset):
        """Splits fitting data to obtain data specifically for fitting and data for validation."""

        dataset = TrainingDataset(None)

        fit_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=val_size,
            stratify=[ImageType.of_file(file) for file in dataset.FILES],
            random_state=42
        )

        fit_dataset = Subset(TrainingDataset(fit_transform), fit_indices)
        val_dataset = Subset(TrainingDataset(val_transform), val_indices)
        return fit_dataset, val_dataset


IMAGENET_MEAN_AND_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
TRAIN_TRANSFORM = transforms.Compose([
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

VAL_SIZE = 0.2
FIT_DATASET, VAL_DATASET = TrainingDataset.split(TRAIN_TRANSFORM, VAL_TRANSFORM, VAL_SIZE)

BATCH_SIZE = 64
FIT_DATALOADER = DataLoader(
    FIT_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
VAL_DATALOADER = DataLoader(
    VAL_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
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


    data = iter(FIT_DATALOADER)
    show_images(next(data)[0], denormalize=True)
    show_images(next(data)[0], denormalize=True)
