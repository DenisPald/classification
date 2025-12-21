import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import os
import yaml

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def build_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return correct / total, running_loss / total


def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total, running_loss / total

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("DEVICE:", device)

    data_dir = Path("data/images/processed")

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(data_dir / "val", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print("CLASSES:", class_names)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cfg = params["train"]

    model = build_model(cfg["model"], len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    epochs = 5
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_loss     = validate_one_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")
        print(f"Val   Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/best_{cfg['model']}.pt")
            print("Saved new best model")

    print(f"\nTraining finished. Best Val Accuracy: {best_acc:.4f}")
