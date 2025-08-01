import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.models as models

import pandas as pd

from data_research import ImageType
import datasets
import models


def submission(model, device):
    """Saves classification results of test dataset in the file."""

    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(datasets.TEST_DATALOADER):
            images = images.to(device)

            outputs = model.forward(images)
            _, prediction = torch.max(outputs, 1)

            predictions.extend(map(lambda label: str(ImageType.of_label(label)), prediction.cpu().numpy()))
    test_df = pd.DataFrame({
        "index": datasets.TEST_DATASET.FILES,
        "label": predictions
    })
    print(test_df)
    test_df.to_csv("submission/submission.csv", index=False)


class ApplicationDataset(datasets.TestingDataset):
    """`ApplicationDataset` performs transforms to images that user wants to classify beforehand."""

    PATH = ""
    FILES = []

    def __init__(self, path: str):
        self.PATH = path
        self.FILES = sorted(os.listdir(path))
        super().__init__()


def main():
    """Entry point of an application."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        print()
        model_id = int(input(
            """`GarmentCV` offers 3 models for blouses vs trousers detection:
        `EfficientNetB0` ('1')
        `ResNet18` ('2')
        `ResNet34` ('3')\n"""))
        if model_id == 1:
            model = models.EfficientNet()
        elif model_id == 2:
            model = models.ResNet18()
        elif model_id == 3:
            model = models.ResNet34()
        else:
            print("Wrong input - there is no model with code " + str(model_id))
            continue

        choice = int(input("Do you want to use pre-trained model ('1') or to train the model manually ('2')?\n"))
        if choice == 1:
            model = model.load()
        elif choice == 2:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=0.01)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
            loss_fn = nn.CrossEntropyLoss()

            epochs = int(input("How many epochs do you want to train for?\n"))

            metrics = model.fit_and_val(epochs, optimizer, lr_scheduler, loss_fn, datasets.FIT_DATALOADER,
                                        datasets.VAL_DATALOADER, device)
            print(f"Metrics: ")
            print(metrics)
        else:
            print("Wrong input - there is no option with code " + str(choice))
            continue

        images_path = input("Enter a path to the folder where your images for classification are located:\n")
        application_dataset = ApplicationDataset(images_path)
        application_dataloader = data.DataLoader(
            application_dataset,
            batch_size=datasets.BATCH_SIZE,
            shuffle=False,
        )

        predictions = []
        with torch.no_grad():
            for images, _ in tqdm(application_dataloader):
                images = images.to(device)

                outputs = model.forward(images)
                _, prediction = torch.max(outputs, 1)

            predictions.extend(map(lambda label: repr(ImageType.of_label(label)), prediction.cpu().numpy()))
        df = pd.DataFrame({
            "index": application_dataset.FILES,
            "label": predictions
        })
        print(df)

        save_to_file = int(
            input("Enter whether you want to save results to a 'results.csv' file ('1') or not ('2')?\n"))
        if save_to_file == 1:
            df.to_csv("results.csv", index=False)
        elif save_to_file == 2:
            continue
        else:
            print("Wrong input - there is no option with code " + str(save_to_file))
            continue


if __name__ == "__main__":
    main()
