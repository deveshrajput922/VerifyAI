import argparse
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint["classes"]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    size = checkpoint.get("image_size", 224)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf


def load_tamper_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = models.efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride, padding=old_conv.padding, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint.get("image_size", 224)


def make_rgb_ela_tensor(image, size):
    import io
    from PIL import ImageChops, ImageEnhance
    image = image.convert("RGB").resize((size, size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    ela = ImageEnhance.Contrast(ImageChops.difference(image, recompressed)).enhance(5)
    to_tensor = transforms.ToTensor()
    tensor = torch.cat([to_tensor(image), to_tensor(ela)], dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406] * 2).view(6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2).view(6, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--classifier", default="models/document_classifier.pth")
    parser.add_argument("--tamper-model", default="models/tampering_detector.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")

    if Path(args.classifier).exists():
        model, classes, tf = load_classifier(args.classifier, device)
        with torch.no_grad():
            probs = torch.softmax(model(tf(image).unsqueeze(0).to(device)), dim=1)[0]
        idx = int(probs.argmax())
        print(f"Document class: {classes[idx]} ({probs[idx].item():.1%})")
    else:
        print("Document classifier checkpoint not found; skipping.")

    if Path(args.tamper_model).exists():
        model, size = load_tamper_model(args.tamper_model, device)
        x = make_rgb_ela_tensor(image, size).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        print(f"Tampering class: {['genuine', 'tampered'][int(probs.argmax())]} "
              f"({probs.max().item():.1%} model confidence)")
    else:
        print("Tampering detector checkpoint not found; skipping.")


if __name__ == "__main__":
    main()
