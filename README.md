# 🔬 Skin Lesion Classification: ResNet-50 vs. Vision Transformer (ViT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TIMM](https://img.shields.io/badge/timm-0.9+-blue.svg)](https://github.com/huggingface/pytorch-image-models)
[![Dataset: HAM10000](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

An empirical deep-learning study evaluating **Convolutional Neural Networks (ResNet-50)** against **Vision Transformers (ViT-B/16)** on the 7-class **HAM10000** dermoscopic skin cancer dataset (10,015 images).

> **Medical Disclaimer:** This project is intended strictly for research and educational purposes. It is not a certified medical device, diagnostic system, or substitute for professional clinical dermatological evaluation.

---

## 📌 Executive Summary & Key Results

When benchmarked under an identical, clinically valid protocol (zero patient/lesion leakage, class-balanced sampling, and fair optimization schedules), the **Vision Transformer (ViT-B/16) decisively outperforms ResNet-50** across every diagnostic metric on the official test set (1,502 images):

| Architecture | Checkpoint | Test Accuracy | Balanced Accuracy | Macro F1 | Melanoma Recall | Melanoma AP | Model Size | GPU Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-50** | `best_resnet_sampler.pth` | 80.69% | 70.99% | 0.7095 | 46.71% | 0.5836 | **89.7 MB** | 31.85 ms |
| **ViT-B/16** | `best_vit_fair.pth` | **85.55%** | **74.52%** | **0.7632** | **60.48%** | **0.6348** | 327.3 MB | **24.92 ms** |

---

## 🎯 Supported Diagnostic Classes

The dataset consists of 10,015 images spanning 7 lesion types with severe class imbalance (**58.3 : 1** ratio):

| Label | Diagnosis | Description | Images | Dataset Share |
|:---:|---|---|:---:|:---:|
| `nv` | Melanocytic nevi | Common benign mole | 6,705 | 66.95% |
| `mel` | **Melanoma** | **Malignant, life-threatening skin cancer** | 1,113 | 11.11% |
| `bkl` | Benign keratosis | Seborrheic keratoses, solar lentigines | 1,099 | 10.97% |
| `bcc` | **Basal cell carcinoma** | Malignant epithelial cancer | 514 | 5.13% |
| `akiec` | Actinic keratoses | Pre-cancerous / Bowen's disease | 327 | 3.26% |
| `vasc` | Vascular lesions | Cherry angiomas, angiokeratomas | 142 | 1.42% |
| `df` | Dermatofibroma | Benign fibrous skin lesion | 115 | 1.15% |

---

## 🔄 Project Evolution: Version 1 vs. Version 2

This project underwent a major methodological overhaul from an initial baseline exploration (`skincancer.ipynb`) to a rigorous, leak-free benchmark (`skincancer_v2.ipynb`):

```
Version 1 (Initial Prototype)              Version 2 (Rigorous Benchmark)
├── Random Image-level Split               ├── Lesion-level Stratified Split (0% Leakage)
├── Heavy Blur / CLAHE / Hair Inpainting   ├── Clean Images + Real-time Color Jitter
├── Plain Adam / No LR Warmup for ViT      ├── AdamW + 3-epoch Warmup + Cosine Annealing
├── Ignored Class Skew in Evaluation       ├── WeightedRandomSampler + Label Smoothing (0.1)
└── Relied on Deceptive Overall Accuracy   └── Evaluated on Macro F1, Melanoma Recall & AP
```

### 🚨 Critical Discoveries in Version 1:
1. **Lesion-Level Data Leakage:** The HAM10000 dataset contains 10,015 images originating from **7,470 unique physical lesions** (many lesions were photographed multiple times). Random image splitting allowed images of the *same lesion* to appear in both train and test splits, artificially inflating scores.
2. **The "Accuracy Paradox":** Because 67% of the images are benign moles (`nv`), a dummy model predicting "mole" every time would score 67% accuracy. In Version 1, high accuracy (~84%) masked the reality that **over 50% of lethal melanomas were missed**!
3. **Over-Processing Degradation:** Heavy classical image processing (Gaussian blur and morphological hair inpainting) unintentionally smudged the delicate honeycomb pigment networks and irregular borders that dermatologists and deep networks need to spot malignancy.

---

## 🛠️ Key Technical Innovations (Version 2)

### 1. Leakage-Free Stratified Lesion Splitting
* Grouped strictly by `lesion_id` before partitioning: 70% Train (5,229 lesions / 6,981 images), 15% Validation (1,120 lesions / 1,532 images), and 15% Test (1,121 lesions / 1,502 images).
* Guaranteed **0 lesion overlap** across train, validation, and test sets.

### 2. Handling 58:1 Class Imbalance
* Implemented a training-only `WeightedRandomSampler` using inverse class frequencies ($w_c = \frac{N}{7 \cdot N_c}$), ensuring minority classes like Dermatofibroma (`df`) and Vascular lesions (`vasc`) are sampled regularly.
* Utilized **Label Smoothing (0.1)** to soften overconfident predictions on majority classes.

### 3. Fair Optimization & Modern Training Recipes
* **ViT-B/16 (`timm`):** Trained with AdamW ($lr = 2 \times 10^{-5}$, weight decay = $0.05$), stochastic depth (`drop_path_rate=0.1`), 3-epoch linear warmup, and cosine annealing over 25 epochs.
* **ResNet-50:** Trained with AdamW ($lr = 10^{-4}$ on backbone, $10^{-3}$ on classification head, weight decay = $0.01$) and cosine annealing.
* **Early Stopping:** Monitored on **Validation Macro F1-score**, preventing validation loss bias.

### 4. Clinical Safety Calibration (Melanoma Threshold Tuning)
In oncology, a False Negative (missing a melanoma) is catastrophic. By sweeping the decision threshold on validation probabilities, setting a **0.05 threshold** dramatically raised Melanoma Recall:
* **ResNet-50 Melanoma Recall:** Jumped from **46.71% $\rightarrow$ 76.65%**
* **ViT-B/16 Melanoma Recall:** Jumped from **60.48% $\rightarrow$ 74.85%**

---

## 📊 Visualizations & Explainability (XAI)

### Confusion Matrices
Both models evaluated on the official test set (1,502 images):
<p align="center">
  <img src="results/figures/test_confusion_matrices.png" width="85%" alt="Confusion Matrices" />
</p>

### Precision-Recall Curves (Melanoma)
ViT demonstrates superior area under the Precision-Recall curve (AP = 0.6348 vs. ResNet's 0.5836):
<p align="center">
  <img src="results/figures/test_melanoma_pr_curves.png" width="70%" alt="Melanoma PR Curves" />
</p>

### Explainable AI on Critical Misses (Melanoma predicted as Nevus)
Inspecting 8 shared false-negative cases using **Grad-CAM** (`layer4[-1]`) on ResNet-50 and **Multi-Head Self-Attention rollout** on ViT-B/16:
<p align="center">
  <img src="results/figures/task9_mel_to_nv.png" width="85%" alt="Explainability Grad-CAM and ViT Attention" />
</p>
*Observation: ResNet concentrates tightly on pigmented core centers or borders, whereas ViT exhibits distributed patch attention that takes broader lesion context and skin transition zones into account.*

---

## 🧪 Preprocessing Ablation Study

Does heavy classical preprocessing (Gaussian blur, CLAHE, hair removal) help deep neural networks?

| Architecture | Plain Validation Macro F1 | Plain Melanoma Recall | Preprocessed Validation Macro F1 | Preprocessed Melanoma Recall |
|---|:---:|:---:|:---:|:---:|
| **ResNet-50** | **0.7075** | 61.27% | 0.6466 | **71.10%** |
| **ViT-B/16** | **0.7339** | **68.21%** | 0.7015 | 58.38% |

**Conclusion:** Deep networks perform best on **raw, uncorrupted images with spatial/color data augmentations**. Handcrafted filters degrade feature extraction by eroding microscopic textural cues.

---

## 📁 Repository Structure

```text
├── data/                               # Dataset splits and metadata
│   ├── HAM10000_metadata.csv           # Original metadata
│   ├── split_train.csv                 # 70% lesion-stratified train split
│   ├── split_val.csv                   # 15% lesion-stratified validation split
│   └── split_test.csv                  # 15% lesion-stratified test split
├── results/
│   └── figures/                        # Evaluation plots and attention maps
│       ├── test_confusion_matrices.png
│       ├── test_melanoma_pr_curves.png
│       ├── task9_mel_to_nv.png
│       ├── vit_fair_curves.png
│       └── resnet_sampler_curves.png
├── scripts/                            # Standalone Python CLI scripts
│   ├── train_vit.py                    # Train ViT-B/16 with warmup and sampler
│   ├── train_resnet.py                 # Train ResNet-50 with differential LR
│   └── evaluate_preprocessing_ablation.py  # Ablation study pipeline
├── skincancer.ipynb                    # Version 1 baseline exploration notebook
├── skincancer_v2.ipynb                 # Version 2 complete 11-step experimental pipeline
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/HeerRana/SkinLessionClassification.git
cd SkinLessionClassification
pip install torch torchvision timm scikit-learn pandas numpy matplotlib opencv-python pillow
```

### 2. Dataset Setup
Download the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) from Kaggle or Harvard Dataverse. Unpack the image folders into the repository root:
```text
HAM10000_images_part_1/
HAM10000_images_part_2/
```

### 3. Running Experiments
* **Jupyter Notebook:** Open `skincancer_v2.ipynb` to step through the entire 11-task workflow with interactive charts and reports.
* **CLI Training (ViT-B/16):**
  ```bash
  python scripts/train_vit.py
  ```
* **CLI Training (ResNet-50):**
  ```bash
  python scripts/train_resnet.py
  ```
* **Run Preprocessing Ablation Study:**
  ```bash
  python scripts/evaluate_preprocessing_ablation.py
  ```

---

## 📜 Citation & References

* **HAM10000 Dataset:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
* **Vision Transformers:** Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
* **ResNet:** He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
