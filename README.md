# Skin Lesion Classification

A deep-learning project for classifying dermoscopic skin-lesion images into seven diagnostic categories using the HAM10000 dataset.

The project compares transfer-learning approaches based on **ResNet50** and **ViT-B/16**, while investigating class imbalance, lesion-level data splitting, image preprocessing, melanoma recall, and macro F1 score.

> **Medical disclaimer:** This project is intended for research and educational purposes only. It is not a medical device, diagnostic system, or substitute for evaluation by a qualified healthcare professional.

## Features

- Seven-class skin-lesion classification.
- Lesion-level train, validation, and test splitting.
- ResNet50 and ViT-B/16 model comparison.
- Image augmentation and ImageNet normalization.
- Class balancing using `WeightedRandomSampler`.
- Optional CLAHE enhancement and hair-artifact removal.
- Evaluation using accuracy, balanced accuracy, macro F1, and melanoma recall.
- Model checkpoints selected using validation macro F1.

## Supported Classes

| Label | Description | Class ID |
|---|---|---:|
| `nv` | Melanocytic nevi | 0 |
| `mel` | Melanoma | 1 |
| `bkl` | Benign keratosis-like lesions | 2 |
| `bcc` | Basal cell carcinoma | 3 |
| `akiec` | Actinic keratoses / intraepithelial carcinoma | 4 |
| `vasc` | Vascular lesions | 5 |
| `df` | Dermatofibroma | 6 |

## Repository Structure

```text
.
├── _run_task10.py       # Preprocessing ablation and model comparison
├── _run_vit.py          # ViT data pipeline and training script
├── _task10_cell.py      # Preprocessing and training experiment code
├── skincancer.ipynb     # Original notebook experiments
├── skincancer_v2.ipynb  # Revised end-to-end experiment notebook
├── data/                # Local data directory
└── results/             # Local experiment results
```

Large datasets, model checkpoints, and experiment logs are excluded from version control. See `.gitignore` for the current exclusions.

## Dataset

This project uses the [HAM10000 dataset](https://www.nature.com/articles/sdata2018161).

Download the dataset and place the following files and directories in the working directory:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

The HAM10000 dataset contains 10,015 dermoscopic images.

The revised workflow generates the following split files:

```text
split_train.csv
split_val.csv
split_test.csv
```

Each split contains:

- `lesion_id`
- `image_id`
- `dx`
- `path`

The dataset is split by `lesion_id` rather than by individual image. This prevents images from the same lesion from appearing in multiple splits.

## Installation

Python 3.9 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio pandas numpy pillow scikit-learn matplotlib opencv-python timm jupyter
```

For GPU training, install the PyTorch version appropriate for your CUDA version from the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Usage

### 1. Prepare the Dataset

Place the HAM10000 metadata file and image folders in the expected location:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

### 2. Generate the Data Splits

Open `skincancer_v2.ipynb` and run the lesion-level split section.

The split process verifies:

- All 10,015 metadata rows are present.
- All image paths are valid.
- No image paths are missing.
- No lesion appears in more than one split.
- All diagnosis labels are recognized.

The default split uses:

- 70% of lesions for training.
- 15% of lesions for validation.
- 15% of lesions for testing.
- `random_state=42`.

### 3. Run the Notebook

Start Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Open one of the following notebooks:

```text
skincancer.ipynb
skincancer_v2.ipynb
```

The revised notebook documents:

1. Baseline experiments.
2. Lesion-level splitting.
3. Image preprocessing.
4. Class balancing.
5. ResNet50 training.
6. ViT-B/16 training.
7. Model evaluation.
8. Preprocessing ablation experiments.

### 4. Run the Scripts

After generating the split CSV files, the scripts can be run directly:

```bash
python _run_vit.py
python _run_task10.py
```

The scripts automatically use CUDA when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If GPU memory is insufficient, reduce the batch size in the relevant script.

## Methodology

### Data Splitting

The revised pipeline groups images by `lesion_id` before splitting. This helps prevent data leakage caused by images from the same lesion appearing in multiple partitions.

The split is stratified by diagnosis where possible.

### Image Transformations

Training images are resized to `224 × 224` and augmented using:

- Random horizontal flipping.
- Random rotation up to 20 degrees.
- Color jitter.
- Conversion to tensors.
- ImageNet normalization.

Validation and test images use resizing and ImageNet normalization without random augmentation.

### Class Balancing

HAM10000 is highly imbalanced, with the `nv` class representing a large portion of the dataset.

To reduce the effect of class imbalance, the training pipeline can use:

- `WeightedRandomSampler`.
- Label smoothing.
- Balanced accuracy.
- Macro F1.
- Melanoma-specific recall.

### Optional Image Preprocessing

The Task 10 experiment includes an optional preprocessing pipeline:

1. Resize the image to `224 × 224`.
2. Apply Gaussian smoothing.
3. Enhance contrast using CLAHE in the LAB color space.
4. Detect hair artifacts using morphological black-hat processing.
5. Remove detected hair artifacts using Telea inpainting.

The preprocessing experiment compares models trained with plain images against models trained with preprocessed images.

## Models

### ResNet50

The ResNet50 pipeline uses:

- ImageNet-pretrained weights.
- A new seven-class fully connected output layer.
- AdamW optimization.
- Separate learning rates for the backbone and classifier head.
- Cosine annealing learning-rate scheduling.
- Early stopping based on validation macro F1.

### ViT-B/16

The ViT-B/16 pipeline uses:

- The `vit_base_patch16_224` model from `timm`.
- ImageNet-pretrained weights.
- Seven output classes.
- AdamW optimization.
- Linear warm-up.
- Cosine annealing learning-rate scheduling.
- Drop-path regularization.
- Early stopping based on validation macro F1.

## Evaluation Metrics

The project reports the following metrics:

- Accuracy.
- Balanced accuracy.
- Macro F1.
- Weighted F1.
- Per-class precision.
- Per-class recall.
- Per-class F1 score.
- Melanoma recall.
- Melanoma average precision where available.
- Confusion matrices and common confusion pairs.

Accuracy should not be interpreted in isolation because the dataset is strongly imbalanced. Macro F1 and melanoma recall provide a more informative view of performance across classes.

## Reference Results

The revised notebook records the following plain-image validation results:

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| ResNet50 | 0.8303 | 0.7075 |
| ViT-B/16 | 0.8492 | 0.7339 |

These values are experiment records from the notebook. Results may vary depending on:

- Hardware.
- CUDA and PyTorch versions.
- Random seeds.
- Dataset paths.
- Batch size.
- Available GPU memory.
- Library versions.

The preprocessing ablation results are evaluated separately and should not be treated as identical to the plain-image baseline.

## Generated Checkpoints and Outputs

Training may generate files such as:

```text
best_resnet_fair.pth
best_resnet_sampler.pth
best_resnet_preproc.pth
best_vit_fair.pth
best_vit_preproc.pth
resnet_fair_curves.png
resnet_sampler_curves.png
vit_fair_curves.png
```

Model checkpoints and generated outputs are ignored by Git because they may be large. Store them locally or in an external artifact repository if they need to be preserved.

## Reproducibility

For more reproducible results:

- Use the lesion-level split from `skincancer_v2.ipynb`.
- Keep the label order fixed.
- Use the same `random_state`.
- Record Python, PyTorch, torchvision, timm, and CUDA versions.
- Record the hardware used for training.
- Set random seeds where appropriate.
- Avoid mixing the original image-level split with the revised lesion-level split.
- Do not evaluate the test set repeatedly during model development.

## Citation

If you use the HAM10000 dataset, cite the original publication:

> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018).  
> https://doi.org/10.1038/sdata.2018.161

## License

No license is currently specified for this repository.

Add a `LICENSE` file before distributing the code or granting permissions for reuse.

## Contributing

Issues and pull requests are welcome.

When proposing changes, please include:

- The experiment or preprocessing configuration.
- The dataset split used.
- The random seed.
- Validation metrics, especially macro F1 and melanoma recall.
- Hardware and software versions where relevant.
- Any changes to preprocessing or class balancing.
