from tqdm import tqdm

import torch

import pandas as pd

from data_research import ImageType
import datasets
import models


def submission(model, device):
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


def main():
    """Entry point of an application."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.EfficientNet().load("best")
    model.model.to(device)


if __name__ == "__main__":
    main()
