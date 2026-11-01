import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class RGBPlusELA:
    """Create a 6-channel tensor: normalized RGB image + normalized ELA map."""
    def __call__(self, image):
        from PIL import Image, ImageChops, ImageEnhance
        import io

        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        diff = ImageChops.difference(image, recompressed)
        diff = ImageEnhance.Contrast(diff).enhance(5)
        return image, diff


class SixChannelTransform:
    def __init__(self, size=224, train=False):
        self.size = size
        self.train = train
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406] * 2,
            [0.229, 0.224, 0.225] * 2
        )

    def __call__(self, image):
        from PIL import Image
        import io
        image = image.convert("RGB")
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((self.size, self.size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        from PIL import ImageChops, ImageEnhance
        ela = ImageEnhance.Contrast(ImageChops.difference(image, recompressed)).enhance(5)
        rgb_tensor = self.to_tensor(image)
        ela_tensor = self.to_tensor(ela)
        return self.normalize(torch.cat([rgb_tensor, ela_tensor], dim=0))


class SixChannelImageFolder(datasets.ImageFolder):
    def __init__(self, root, train=False, size=224):
        super().__init__(root)
        self.custom_transform = SixChannelTransform(size=size, train=train)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return self.custom_transform(sample), target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tampering")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="models/tampering_detector.pth")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = Path(args.data)
    ds_train_full = SixChannelImageFolder(data_dir, train=True, size=args.image_size)
    ds_val_full = SixChannelImageFolder(data_dir, train=False, size=args.image_size)
    if ds_train_full.classes != ["genuine", "tampered"]:
        raise ValueError("Create exactly these class folders: genuine/ and tampered/")
    if len(ds_train_full.samples) < 40:
        raise ValueError("Need at least 40 images total; use substantially more for meaningful results.")

    n_val = max(1, int(0.2 * len(ds_train_full)))
    n_train = len(ds_train_full) - n_val
    train_ds, _ = random_split(ds_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    _, val_ds = random_split(ds_val_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight * 0.5
        new_conv.weight[:, 3:] = old_conv.weight * 0.5
    model.features[0][0] = new_conv
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
        print(f"Epoch {epoch+1}: train_loss={total_loss/max(n_train,1):.4f}, val_accuracy={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": ds_train_full.classes,
                "image_size": args.image_size,
                "architecture": "efficientnet_b0_rgb_ela_6ch",
            }, args.output)

    print("\nBest validation accuracy:", round(best_acc, 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["genuine", "tampered"], zero_division=0))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
