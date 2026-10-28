"""
Train ResNet-50 on the HAM10000 Dataset.

Features:
- Leakage-free lesion-level dataset loading
- WeightedRandomSampler to handle extreme 58:1 class imbalance
- Differential learning rates (backbone 1e-4, classifier 1e-3) with Cosine Annealing
- Checkpoint selection based on Validation Macro F1
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

LABEL_MAP = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}
CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SkinDataset(Dataset):
    def __init__(self, csv_path, project_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform
        unknown = set(self.df["dx"]) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"Unknown dx values in {csv_path}: {unknown}")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path_str):
        path = Path(raw_path_str)
        if path.exists():
            return path
        p1 = self.project_root / "HAM10000_images_part_1" / path.name
        if p1.exists():
            return p1
        p2 = self.project_root / "HAM10000_images_part_2" / path.name
        if p2.exists():
            return p2
        return path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["path"])
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[row["dx"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"Project root: {root}")
    print(f"Data directory: {data_dir}")

    train_dataset = SkinDataset(data_dir / "split_train.csv", project_root=root, transform=train_transform)
    val_dataset = SkinDataset(data_dir / "split_val.csv", project_root=root, transform=eval_transform)
    test_dataset = SkinDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)

    # Class balancing
    train_counts = train_dataset.df["dx"].value_counts()
    n_train = len(train_dataset)
    class_weights = {
        name: n_train / (len(CLASS_ORDER) * int(train_counts[name]))
        for name in CLASS_ORDER
    }

    sample_weights = train_dataset.df["dx"].map(class_weights).to_numpy()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Model definition
    model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model_resnet.fc = torch.nn.Linear(2048, 7)
    model_resnet = model_resnet.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    backbone_params = [
        param for name, param in model_resnet.named_parameters() if not name.startswith("fc.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-4},
            {"params": model_resnet.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )

    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_macro_f1 = -1.0
    best_epoch = 0
    checkpoint_path = root / "best_resnet_sampler.pth"

    train_losses, val_losses, val_macro_f1s = [], [], []
    epochs_without_improvement = 0
    patience = 5

    print("\nStarting ResNet-50 training...")
    for epoch in range(1, epochs + 1):
        model_resnet.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model_resnet.eval()
        val_loss_sum = 0.0
        pred_list, label_list = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_resnet(images)
                val_loss_sum += criterion(outputs, labels).item()
                pred_list.extend(outputs.argmax(dim=1).cpu().tolist())
                label_list.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(label_list, pred_list)
        val_bal = balanced_accuracy_score(label_list, pred_list)
        val_f1 = f1_score(label_list, pred_list, average="macro", zero_division=0)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_macro_f1s.append(val_f1)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Bal Acc: {val_bal:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model_resnet.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Macro F1: {best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining completed. Best Epoch: {best_epoch} with Val Macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
