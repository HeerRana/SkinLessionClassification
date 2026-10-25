import cv2
import numpy as np
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import WeightedRandomSampler
from torchvision.models import ResNet50_Weights, resnet50

MEL_INDEX = LABEL_MAP["mel"]
CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
PLAIN_MACRO_F1 = {"ResNet50": 0.7075, "ViT-B/16": 0.7339}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print("Not loaded: best_resnet_fair.pth", flush=True)
print("Official checkpoints will not be overwritten", flush=True)

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


def cache_plain(frame):
    return [Image.open(path).convert("RGB").resize((224, 224)) for path in frame["path"]]


def cache_preprocessed(frame, name):
    images = []
    for index, path in enumerate(frame["path"], start=1):
        images.append(preprocess_image(Image.open(path)))
        if index % 1000 == 0 or index == len(frame):
            print(f"Preprocessed {name} {index}/{len(frame)}", flush=True)
    return images


def make_loader(images, labels, transform, batch_size, sampler=None):
    return DataLoader(
        CachedSkin(images, labels, transform),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )


def melanoma_recall(labels, predictions):
    true_melanoma = (np.asarray(labels) == MEL_INDEX).astype(int)
    predicted_melanoma = (np.asarray(predictions) == MEL_INDEX).astype(int)
    return recall_score(true_melanoma, predicted_melanoma, zero_division=0)


def predict(model, loader):
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, batch_labels).item()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return labels, predictions, loss_sum / len(loader)


def score(labels, predictions, loss):
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "melanoma_recall": melanoma_recall(labels, predictions),
    }


def load_resnet(path):
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 7)
    model.load_state_dict(torch.load(here / path, map_location=device, weights_only=True))
    return model.to(device)


def load_vit(path, pretrained):
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=pretrained, num_classes=7, drop_path_rate=0.1
    )
    if path is not None:
        model.load_state_dict(torch.load(here / path, map_location=device, weights_only=True))
    return model.to(device)


val_labels = val_dataset.df["dx"].map(LABEL_MAP).tolist()
plain_val_loader = make_loader(cache_plain(val_dataset.df), val_labels, eval_tensor, 16)
print("Plain-image validation melanoma recall", flush=True)
plain_melanoma_recall = {}
for model_name, filename, builder in (
    ("ResNet50", "best_resnet_sampler.pth", lambda: load_resnet("best_resnet_sampler.pth")),
    ("ViT-B/16", "best_vit_fair.pth", lambda: load_vit("best_vit_fair.pth", pretrained=False)),
):
    model = builder()
    labels, predictions, loss = predict(model, plain_val_loader)
    metrics = score(labels, predictions, loss)
    plain_melanoma_recall[model_name] = metrics["melanoma_recall"]
    print(
        f"{model_name} {filename} | val macro F1 {PLAIN_MACRO_F1[model_name]:.4f} | "
        f"val melanoma recall {metrics['melanoma_recall']:.4f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
del plain_val_loader

train_labels = train_dataset.df["dx"].map(LABEL_MAP).tolist()
train_counts = train_dataset.df["dx"].value_counts()
sample_weights = train_dataset.df["dx"].map(
    {name: len(train_dataset) / (7 * int(train_counts[name])) for name in CLASS_ORDER}
).to_numpy()
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
print("Preprocessing train and validation images in memory", flush=True)
train_images = cache_preprocessed(train_dataset.df, "train")
val_images = cache_preprocessed(val_dataset.df, "val")
val_loader = make_loader(val_images, val_labels, eval_tensor, 16)


def train_model(model_name, model, optimizer, scheduler, train_batch, checkpoint_name):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    train_loader = make_loader(train_images, train_labels, train_aug, train_batch, sampler)
    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    stall = 0
    checkpoint_path = here / checkpoint_name
    for epoch in range(1, 16):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        labels, predictions, val_loss = predict(model, val_loader)
        metrics = score(labels, predictions, val_loss)
        scheduler.step()
        print(
            f"{model_name} epoch {epoch}/15 | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.4f} | "
            f"Val Bal Acc: {metrics['balanced_accuracy']:.4f} | "
            f"Val Macro F1: {metrics['macro_f1']:.4f} | "
            f"Val Melanoma Recall: {metrics['melanoma_recall']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_epoch = epoch
            stall = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stall += 1
            if stall >= 5:
                print(f"{model_name} early stopping at epoch {epoch}", flush=True)
                break
    print(
        f"{model_name} best epoch {best_epoch} | val accuracy {best_metrics['accuracy']:.4f} | "
        f"val macro F1 {best_metrics['macro_f1']:.4f} | "
        f"val melanoma recall {best_metrics['melanoma_recall']:.4f}",
        flush=True,
    )
    return best_metrics


model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_resnet.fc = torch.nn.Linear(2048, 7)
model_resnet = model_resnet.to(device)
resnet_optimizer = torch.optim.AdamW(
    [
        {"params": [p for n, p in model_resnet.named_parameters() if not n.startswith("fc.")], "lr": 1e-4},
        {"params": model_resnet.fc.parameters(), "lr": 1e-3},
    ],
    weight_decay=0.01,
)
resnet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resnet_optimizer, T_max=15)
resnet_best = train_model("ResNet50", model_resnet, resnet_optimizer, resnet_scheduler, 32, "best_resnet_preproc.pth")
del model_resnet
torch.cuda.empty_cache()

model_vit = load_vit(None, pretrained=True)
vit_optimizer = torch.optim.AdamW(model_vit.parameters(), lr=2e-5, weight_decay=0.05)
warmup = torch.optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=1 / 3, end_factor=1.0, total_iters=3)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(vit_optimizer, T_max=12)
vit_scheduler = torch.optim.lr_scheduler.SequentialLR(
    vit_optimizer, schedulers=[warmup, cosine], milestones=[3]
)
vit_best = train_model("ViT-B/16", model_vit, vit_optimizer, vit_scheduler, 16, "best_vit_preproc.pth")
del model_vit
torch.cuda.empty_cache()

print("\nValidation comparison", flush=True)
print(
    "ResNet50 plain macro F1 0.7075 | plain melanoma recall "
    f"{plain_melanoma_recall['ResNet50']:.4f} | preproc macro F1 {resnet_best['macro_f1']:.4f} | "
    f"preproc melanoma recall {resnet_best['melanoma_recall']:.4f}",
    flush=True,
)
print(
    "ViT-B/16 plain macro F1 0.7339 | plain melanoma recall "
    f"{plain_melanoma_recall['ViT-B/16']:.4f} | preproc macro F1 {vit_best['macro_f1']:.4f} | "
    f"preproc melanoma recall {vit_best['melanoma_recall']:.4f}",
    flush=True,
)

if resnet_best["macro_f1"] > PLAIN_MACRO_F1["ResNet50"] or vit_best["macro_f1"] > PLAIN_MACRO_F1["ViT-B/16"]:
    print("Secondary ablation test. This does not replace the Task 7 table.", flush=True)
    test_images = cache_preprocessed(test_dataset.df, "test")
    test_labels = test_dataset.df["dx"].map(LABEL_MAP).tolist()
    test_loader_eval = make_loader(test_images, test_labels, eval_tensor, 16)
    for model_name, best, filename, builder, plain_f1 in (
        ("ResNet50", resnet_best, "best_resnet_preproc.pth", lambda: load_resnet("best_resnet_preproc.pth"), 0.7075),
        ("ViT-B/16", vit_best, "best_vit_preproc.pth", lambda: load_vit("best_vit_preproc.pth", pretrained=False), 0.7339),
    ):
        if best["macro_f1"] <= plain_f1:
            print(f"{model_name} validation macro F1 did not beat plain images. Test set not run.", flush=True)
            continue
        model = builder()
        labels, predictions, loss = predict(model, test_loader_eval)
        metrics = score(labels, predictions, loss)
        del model
        torch.cuda.empty_cache()
        print(
            f"Secondary ablation {model_name} {filename} | test accuracy {metrics['accuracy']:.4f} | "
            f"test macro F1 {metrics['macro_f1']:.4f} | test melanoma recall {metrics['melanoma_recall']:.4f}",
            flush=True,
        )
else:
    print("Neither ablation beat its plain-image validation macro F1. Test set not run.", flush=True)
