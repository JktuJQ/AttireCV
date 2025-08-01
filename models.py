"""`models` module provides several pre-trained CV models that could be used for the task."""

import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class CVModel(nn.Module):
    """`CVModel` is an abstract class that provides fitting and validation of models."""

    MODELS_PATH = "models"

    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model

    def forward(self, x):
        return self.model(x)

    @classmethod
    def load(cls, file: str = None):
        """Loads model from the file."""

        model = cls()
        model.model.load_state_dict(
            torch.load(os.path.join(model.MODELS_PATH, model.name if file is None else file) + ".pth",
                       map_location=torch.device("cpu"))
        )
        return model

    def save(self, file: str = None):
        """Saves model to the file."""

        torch.save(self.model.state_dict(),
                   os.path.join(self.MODELS_PATH, self.name if file is None else file) + ".pth")

    def fit_one_epoch(self, optimizer, lr_scheduler, loss_fn, dataloader, device):
        """Fits model using given data, optimizer, loss function for one epoch. Returns loss and accuracy on epoch."""

        self.model.train()

        full_loss = 0.0
        total = 0
        correct = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = self.model.forward(images)

            loss = loss_fn(outputs, labels)
            full_loss += loss.item()

            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()

        lr_scheduler.step()

        epoch_loss = full_loss / len(dataloader)
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy

    def fit_and_val(self, epochs, optimizer, lr_scheduler, loss_fn, fit_dataloader, val_dataloader, device):
        """Fits and validates model using given data, optimizer, loss function for N epochs and returns metrics."""

        metrics = {
            "fit_loss": [],
            "fit_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        for epoch in range(1, epochs + 1):
            fit_loss, fit_accuracy = self.fit_one_epoch(optimizer, lr_scheduler, loss_fn, fit_dataloader, device)
            metrics["fit_loss"].append(fit_loss)
            metrics["fit_accuracy"].append(fit_accuracy)

            val_loss, val_accuracy = self.validate_one_epoch(loss_fn, val_dataloader, device)
            metrics["val_loss"].append(val_loss)
            metrics["val_accuracy"].append(val_accuracy)

            print(f"Эпоха №{epoch} из {epochs}")
            print(f"Потери тренировки: {fit_loss:.5f}, точность тренировки: {fit_accuracy:.2f}")
            print(f"Потери валидации: {val_loss:.5f}, точность валидации: {val_accuracy:.2f}")
        return metrics

    @torch.no_grad()
    def validate_one_epoch(self, loss_fn, dataloader, device):
        """Validates model using given data, loss function once. Returns loss and accuracy on that validation."""

        self.model.eval()

        full_loss = 0.0
        total = 0
        correct = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = self.model.forward(images)

            loss = loss_fn(outputs, labels)
            full_loss += loss.item()

            total += labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()

        epoch_loss = full_loss / len(dataloader)
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy


class EfficientNet(CVModel):
    """EfficientNet B0 model."""

    def __init__(self):
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        super().__init__("efficientnetb0", model)


class ResNet18(CVModel):
    """ResNet18 model."""

    def __init__(self):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        super().__init__("resnet18", model)


class ResNet34(CVModel):
    """ResNet34 model."""

    def __init__(self):
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        super().__init__("resnet34", model)


if __name__ == "__main__":
    from datasets import FIT_DATALOADER, VAL_DATALOADER

    model = ResNet34()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5

    print(model.fit_and_val(epochs, optimizer, lr_scheduler, loss_fn, FIT_DATALOADER, VAL_DATALOADER, device))
    model.save()
