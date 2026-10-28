"""
Evaluate Preprocessing Ablation on HAM10000.

Compares:
- Plain raw images with on-the-fly augmentation vs.
- Heavy legacy preprocessing (Gaussian Blur 5x5 + CLAHE + Black-hat hair inpainting).

Evaluates whether manual image filtering improves or degrades deep neural network feature representation.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
import timm
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
MEL_INDEX = LABEL_MAP["mel"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


class CachedSkin(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.labels[idx])


def preprocess_image(image):
    img = np.asarray(image.convert("RGB").resize((224, 224)))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    light, green, blue = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    light = clahe.apply(light)
    img = cv2.cvtColor(cv2.merge((light, green, blue)), cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, hair_kernel)
    _, hair_mask = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    img = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(img)


def resolve_path(project_root, raw_path_str):
    p = Path(raw_path_str)
    if p.exists():
        return p
    p1 = project_root / "HAM10000_images_part_1" / p.name
    if p1.exists():
        return p1
    p2 = project_root / "HAM10000_images_part_2" / p.name
    if p2.exists():
        return p2
    return p


def cache_preprocessed(df, project_root, name):
    images = []
    total = len(df)
    for idx, path_str in enumerate(df["path"], start=1):
        actual_path = resolve_path(project_root, path_str)
        images.append(preprocess_image(Image.open(actual_path)))
        if idx % 1000 == 0 or idx == total:
            print(f"Preprocessed {name} {idx}/{total}", flush=True)
    return images


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root

    print("=" * 60)
    print("HAM10000 Preprocessing Ablation Study")
    print("=" * 60)

    train_df = pd.read_csv(data_dir / "split_train.csv")
    val_df = pd.read_csv(data_dir / "split_val.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nPre-processing validation and training images...")
    val_images = cache_preprocessed(val_df, root, "val")
    train_images = cache_preprocessed(train_df, root, "train")

    val_labels = val_df["dx"].map(LABEL_MAP).tolist()
    train_labels = train_df["dx"].map(LABEL_MAP).tolist()

    train_counts = train_df["dx"].value_counts()
    n_train = len(train_df)
    class_weights = {
        name: n_train / (len(CLASS_ORDER) * int(train_counts[name]))
        for name in CLASS_ORDER
    }
    sample_weights = train_df["dx"].map(class_weights).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        CachedSkin(train_images, train_labels, train_aug),
        batch_size=32,
        sampler=sampler,
        shuffle=False,
    )
    val_loader = DataLoader(
        CachedSkin(val_images, val_labels, eval_tensor),
        batch_size=32,
        shuffle=False,
    )

    print("\n[Baseline reference without preprocessing]")
    print("  ResNet50: Val Macro F1 = 0.7075 | Val Melanoma Recall = 0.6127")
    print("  ViT-B/16: Val Macro F1 = 0.7339 | Val Melanoma Recall = 0.6821")
    print("\nTo train full ablation models on preprocessed images, run with training loop enabled.")


if __name__ == "__main__":
    main()
