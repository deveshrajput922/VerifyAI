import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/classification")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="models/document_classifier.pth")
    args = parser.parse_args()

    seed_everything()
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir.resolve()}")

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Discover class names and ensure enough samples exist.
    full_for_names = datasets.ImageFolder(data_dir)
    if len(full_for_names.classes) < 2:
        raise ValueError("Need at least two class folders with images.")
    if len(full_for_names.samples) < 20:
        raise ValueError("Need at least 20 images to start; more is strongly recommended.")

    # Split by random indices. For serious evaluation, split by source document/person
    # before placing files in folders to avoid near-duplicate leakage.
    n = len(full_for_names)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        datasets.ImageFolder(data_dir, transform=train_tf),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_base = datasets.ImageFolder(data_dir, transform=eval_tf)
    val_ds.dataset = val_base

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(full_for_names.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}: train_loss={running_loss/max(n_train,1):.4f}, val_accuracy={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": full_for_names.classes,
                "image_size": args.image_size,
                "architecture": "efficientnet_b0",
            }, args.output)

    print("\nBest validation accuracy:", round(best_acc, 4))
    print("Classes:", full_for_names.classes)
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=full_for_names.classes, zero_division=0))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
