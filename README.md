# Accurate-Classification-of-Alzheimer-s-Disease
Alzheimer's disease (AD) is a progressive neurodegenerative disorder that predominantly impacts the elderly population, characterised by the gradual deterioration of cognitive functions. It makes up between 60 to 70 per cent of all dementia cases, making it the most prevalent kind of dementia. 
"""
Deep Learning Models for Accurate Classification of Alzheimer's Disease:
Insights from Lightweight (LWDCNN), ResNet-18, and SqueezeNet

- Ready for Kaggle Notebook or local run
- Single-file training script you can commit to GitHub
- Trains three models (ResNet-18, SqueezeNet1.0, LWDCNN)
- Uses 80/20 Train/Validation split (from the provided training set)
- Evaluates on the provided test set
- Applies reflection (flip), translation, and scaling data augmentation
- Runs 20 epochs by default

Author: <your-name>
License: MIT
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------
# LWDCNN Definition (5 Conv Blocks)
# Each block: Conv -> BatchNorm -> LeakyReLU -> MaxPool
# -------------------------------
class LWDCNN(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        # Keep it lightweight: modest channel sizes
        channels = [16, 32, 64, 128, 256]
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# -------------------------------
# Data Pipeline
# -------------------------------

def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    # ImageNet normalization for ResNet/SqueezeNet; works fine for LWDCNN too
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Augmentations: reflection (flip), translation, scaling
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # reflection
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # translation up to 10%
            scale=(0.9, 1.1),      # scaling 0.9x to 1.1x
        ),
        transforms.ToTensor(),
        normalize,
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    test_tfms = val_tfms
    return train_tfms, val_tfms, test_tfms


def load_datasets(data_root: Path, image_size: int = 224, val_split: float = 0.2, seed: int = 42):
    train_tfms, val_tfms, test_tfms = build_transforms(image_size)

    # Expect Kaggle-like structure:
    # data_root/train/<class folders>
    # data_root/test/<class folders>
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'

    # If the dataset is single folder, allow fallback to ImageFolder and split from there
    if train_dir.exists():
        full_train = datasets.ImageFolder(train_dir, transform=train_tfms)
        class_names = full_train.classes
        n_total = len(full_train)
        n_val = int(val_split * n_total)
        n_train = n_total - n_val
        train_set, val_set = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        # Override transforms for val subset
        val_set.dataset = datasets.ImageFolder(train_dir, transform=val_tfms)
    else:
        # Single-folder fallback
        full = datasets.ImageFolder(data_root, transform=train_tfms)
        class_names = full.classes
        n_total = len(full)
        n_val = int(val_split * n_total)
        n_train = n_total - n_val
        train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        val_set.dataset = datasets.ImageFolder(data_root, transform=val_tfms)
        test_dir = None

    if test_dir and test_dir.exists():
        test_set = datasets.ImageFolder(test_dir, transform=test_tfms)
    else:
        # If no explicit test set, use validation as test (not ideal, but keeps script robust)
        test_set = val_set

    return train_set, val_set, test_set, class_names


# -------------------------------
# Model Factory
# -------------------------------

def build_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == 'resnet18' or name == 'resnet-18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Adjust first conv for single-channel inputs handled by ToTensor (we keep 3 channels)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif name == 'squeezenet' or name == 'squeezenet1_0' or name == 'squeezenet1.0':
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif name == 'lwdcnn' or name == 'lightweight' or name == 'lw':
        return LWDCNN(num_classes=num_classes, in_channels=3)
    else:
        raise ValueError(f"Unknown model name: {name}")


# -------------------------------
# Train / Evaluate
# -------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    return epoch_loss, epoch_acc, y_true, y_pred


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alzheimer's MRI Classification: ResNet-18, SqueezeNet, LWDCNN")
    parser.add_argument('--data_root', type=str, default='/kaggle/input/alzheimers-dataset-4-class',
                        help='Path to dataset root. Expect train/ and test/ subfolders.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', action='store_true', help='Disable ImageNet pretraining for backbones')
    parser.add_argument('--models', type=str, default='resnet18,squeezenet,lwdcnn',
                        help='Comma-separated: resnet18,squeezenet,lwdcnn')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & Dataloaders
    train_set, val_set, test_set, class_names = load_datasets(data_root, image_size=args.image_size,
                                                              val_split=args.val_split, seed=args.seed)
    print(f"Classes: {class_names}")
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Prepare log file
    log_path = out_dir / 'training_log.csv'
    with open(log_path, 'w') as f:
        f.write('model,epoch,train_loss,train_acc,val_loss,val_acc,params\n')

    results_txt = out_dir / 'results.txt'
    with open(results_txt, 'w') as f:
        f.write('')

    for model_name in [m.strip() for m in args.models.split(',') if m.strip()]:
        print(f"\n==== Training {model_name} ====")
        model = build_model(model_name, num_classes=len(class_names), pretrained=not args.no_pretrained)
        model = model.to(device)
        n_params = count_params(model)
        print(f"Trainable parameters: {n_params/1e6:.2f}M")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        best_val_acc = -1.0
        best_weights_path = out_dir / f"best_{model_name}.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            with open(log_path, 'a') as f:
                f.write(f"{model_name},{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{n_params}\n")

            print(f"Epoch {epoch:02d}/{args.epochs} | Train: loss {train_loss:.4f}, acc {train_acc:.4f} | Val: loss {val_loss:.4f}, acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state_dict': model.state_dict(),
                            'class_names': class_names,
                            'params': n_params}, best_weights_path)

        # Load best and evaluate on test set
        print(f"\nEvaluating best {model_name} on the test set...")
        ckpt = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

        with open(out_dir / f"report_{model_name}.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Params: {n_params}\n")
            f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Acc: {test_acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n\n")
            f.write("Confusion Matrix (rows=true, cols=pred):\n")
            f.write(np.array2string(cm, separator=', ') + "\n")

        with open(results_txt, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Params: {n_params}\n")
            f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Acc: {test_acc:.4f}\n\n")

        print(f"Done. Test Acc: {test_acc:.4f}. Reports saved to {out_dir}/")

    print("\nTraining complete for all models.")
    print(f"Logs: {log_path}")
    print(f"Per-model reports in: {out_dir}")


if __name__ == '__main__':
    main()
