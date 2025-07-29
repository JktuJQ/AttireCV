"""`data` module provides basic research on the dataset."""

import os

import typing as t
from enum import StrEnum
from collections import Counter

from PIL import Image, UnidentifiedImageError


class ImageType(StrEnum):
    """`ImageType` class lists all classifiable types of images."""

    BLOUSE = "bluzy"
    TROUSERS = "bryuki"

    def __repr__(self):
        match self:
            case ImageType.BLOUSE:
                return "blouse"
            case ImageType.TROUSERS:
                return "trousers"


DATA_FOLDER_PATH: str = os.path.join("lamoda-images-classification", "images")


class DataFolder(StrEnum):
    """`DataFolder` class provides operations on the dataset folders."""

    TRAIN = os.path.join(DATA_FOLDER_PATH, "train")
    TEST = os.path.join(DATA_FOLDER_PATH, "test")

    def __repr__(self):
        match self:
            case DataFolder.TRAIN:
                return "train"
            case DataFolder.TEST:
                return "test"

    def __len__(self):
        return len(self.files())

    def files(self) -> list[str]:
        """Returns names of files in the folder."""

        return os.listdir(self)

    def corrupt_files(self) -> list[str]:
        """Finds corrupt files in the folder and returns their names."""

        corrupt_files = list()
        for file in self.files():
            file = os.path.join(self, file)
            try:
                with Image.open(file) as image:
                    image.verify()
            except (UnidentifiedImageError, IOError):
                corrupt_files.append(file)
        return corrupt_files

    def image_types(self) -> t.Optional[Counter[ImageType]]:
        """Returns `Counter` of types of all images if called for `DataFolder.TRAIN`, otherwise returns `None`."""

        if self == DataFolder.TEST:
            return None
        return Counter([ImageType(os.path.splitext(file)[0].split("_")[1].lower()) for file in self.files()])

    def image_sizes(self) -> Counter[(int, int)]:
        """Returns `Counter` of sizes of all images. This function does not handle cases where file is not an image."""

        sizes = list()
        for file in self.files():
            file = os.path.join(self, file)
            with Image.open(file) as image:
                sizes.append(image.size)
        return Counter(sizes)

    def extensions(self) -> Counter[str]:
        """Returns `Counter` of extensions of all files."""

        return Counter([os.path.splitext(file)[1].lower() for file in self.files()])


if __name__ == "__main__":
    def research(folder: DataFolder):
        """Conducts research on the supplied folder - that includes information about files, their extensions, etc."""

        print(f"There is {len(folder)} images in '{folder}', {len(folder.corrupt_files())} of them are corrupt.")
        # for file in folder.corrupt_files(): os.remove(file)

        print(f"Extensions of the '{folder}' images: {folder.extensions()}")
        print(f"Types of the '{folder}' images: {folder.image_types()} (for `DataFolder.TEST` it should be `None`)")
        print(f"Sizes of the '{folder}' images: {folder.image_sizes()}")
        # The dataset is quite good, so we trust it won't have any outliers.


    print("Research 'train':")
    research(DataFolder.TRAIN)
    print("\n\n\n")
    print("Research 'test':")
    research(DataFolder.TEST)
    print("\n\n\n")
