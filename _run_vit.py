from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_MAP = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}

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
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        unknown = set(self.df["dx"]) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"Unknown dx values in {csv_path}: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        if "preprocessed_images" in path.parts:
            raise ValueError(f"Refusing preprocessed image: {path}")
        image = Image.open(path).convert("RGB")
        label = LABEL_MAP[row["dx"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


here = Path.cwd()
train_dataset = SkinDataset(here / "split_train.csv", transform=train_transform)
val_dataset = SkinDataset(here / "split_val.csv", transform=eval_transform)
test_dataset = SkinDataset(here / "split_test.csv", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))
print("Batch shape:", tuple(images.shape))
print("Label set:", sorted(set(labels.tolist())))
channel_mean = images.mean(dim=(0, 2, 3))
channel_std = images.std(dim=(0, 2, 3))
print("Per-channel mean:", [round(v, 4) for v in channel_mean.tolist()])
print("Per-channel std:", [round(v, 4) for v in channel_std.tolist()])


import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
train_counts = train_dataset.df["dx"].value_counts()
n_train = len(train_dataset)
class_weights = {
    name: n_train / (len(CLASS_ORDER) * int(train_counts[name]))
    for name in CLASS_ORDER
}

print(f"{'class':<8} {'count':>8} {'weight':>10}")
for name in CLASS_ORDER:
    print(f"{name:<8} {int(train_counts[name]):>8} {class_weights[name]:>10.4f}")

weight_tensor = torch.tensor(
    [class_weights[name] for name in CLASS_ORDER], dtype=torch.float32
)
sample_weights = train_dataset.df["dx"].map(class_weights).to_numpy()
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
train_loader = DataLoader(
    train_dataset, batch_size=32, sampler=sampler, shuffle=False
)

sampled_idx = list(sampler)[:800]
sampled_dx = train_dataset.df.iloc[sampled_idx]["dx"]
print("\nSampled class histogram (800 draws):")
for name in CLASS_ORDER:
    print(f"{name:<8} {int((sampled_dx == name).sum()):>5}")

criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
print("\nCriterion:", criterion)
print("Loss weights:", [round(v, 4) for v in criterion.weight.tolist()])
print("Label smoothing:", criterion.label_smoothing)


import math
import matplotlib.pyplot as plt
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
print("Loss: unweighted cross-entropy, label smoothing", criterion.label_smoothing)

model_vit = timm.create_model(
    "vit_base_patch16_224", pretrained=True, num_classes=7, drop_path_rate=0.1
)
model_vit = model_vit.to(device)

optimizer = torch.optim.AdamW(model_vit.parameters(), lr=2e-5, weight_decay=0.05)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1 / 3, end_factor=1.0, total_iters=3
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[3]
)

epochs = 25
patience = 5
best_macro_f1 = -1.0
best_val_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0
checkpoint_path = here / "best_vit_fair.pth"
train_losses, val_losses = [], []
val_accuracies, val_balanced, val_macro_f1s = [], [], []

train_resized = [Image.open(path).convert("RGB").resize((224, 224)) for path in train_dataset.df["path"]]
val_resized = [Image.open(path).convert("RGB").resize((224, 224)) for path in val_dataset.df["path"]]
train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CachedSkin(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.labels[idx])

train_labels = train_dataset.df["dx"].map(LABEL_MAP).tolist()
val_labels = val_dataset.df["dx"].map(LABEL_MAP).tolist()
train_loader = DataLoader(
    CachedSkin(train_resized, train_labels, train_aug),
    batch_size=32,
    sampler=sampler,
    shuffle=False,
)
val_loader = DataLoader(
    CachedSkin(val_resized, val_labels, eval_tensor),
    batch_size=32,
    shuffle=False,
)
print(f"Cached images: train {len(train_resized)}, val {len(val_resized)}", flush=True)

stop_for_review = False
for epoch in range(1, epochs + 1):
    model_vit.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_vit(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model_vit.eval()
    val_loss_sum = 0.0
    pred_list, label_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_vit(images)
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
    val_accuracies.append(val_acc)
    val_balanced.append(val_bal)
    val_macro_f1s.append(val_f1)

    print(
        f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Bal Acc: {val_bal:.4f} | Val Macro F1: {val_f1:.4f}",
        flush=True,
    )

    if val_f1 > best_macro_f1:
        best_macro_f1 = val_f1
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model_vit.state_dict(), checkpoint_path)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if epoch == 5 and train_loss < 0.15 and val_f1 < 0.55:
        stop_for_review = True
        print(
            "Stopping: train loss collapsed by epoch 5 while val macro F1 stayed under 0.55. "
            "Learning rate was not changed.",
            flush=True,
        )
        break

if not stop_for_review:
    epoch_axis = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epoch_axis, train_losses, label="Train loss")
    axes[0].plot(epoch_axis, val_losses, label="Val loss")
    axes[0].set_title("ViT-B/16 — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epoch_axis, val_macro_f1s, label="Val macro F1", color="green")
    axes[1].set_title("ViT-B/16 — Validation macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(here / "vit_fair_curves.png", dpi=150)
    plt.show()

    model_vit.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best val macro F1: {best_macro_f1:.4f}")
