# Retinal Image Classifier for Diabetic Retinopathy

Classifies diabetic retinopathy severity (0–4) from fundus retinal images using transfer learning and explainable AI.

**Best validation Kappa: 0.868**

## Motivation

Diabetic retinopathy is the leading cause of blindness among working-age adults. Early detection is critical but screening is limited in rural and low-resource settings. This project explores automating severity grading from fundus photographs to assist clinical decision-making.

## Dataset

- **Source:** APTOS 2019 Blindness Detection (public, Aravind Eye Hospital, India)
- **Size:** 3,662 labeled fundus images across 5 severity grades
- **Challenge:** Significant class imbalance (Grade 0: 1,805 vs Grade 3: 193 images)
- **Collection environment:** Multiple rural clinics with varying camera equipment - causing inconsistent lighting and contrast across images

## Approach

**Preprocessing**

- CLAHE (Contrast Limited Adaptive Histogram Equalization) applied in LAB color space to normalize contrast across images from different equipment
- Resized to 224×224 to match ResNet50 pretraining dimensions

**Model**

- ResNet50 pretrained on ImageNet (transfer learning)
- Custom classification head with dropout regularization
- Two-phase training: frozen backbone (10 epochs) → full fine-tuning (5 epochs, lr=1e-5)

**Training**

- Class-weighted CrossEntropyLoss to handle imbalance
- Adam optimizer
- Data augmentation: horizontal/vertical flips, rotation, color jitter

## Results

| Phase           | Epochs | Best Kappa |
| --------------- | ------ | ---------- |
| Frozen backbone | 10     | 0.812      |
| Fine-tuning     | 5      | 0.868      |

**Confusion matrix highlights:**

- No DR detection: 96% accuracy - no healthy eyes misclassified as severe
- Main errors occur between adjacent grades (Mild<->Moderate, Severe<->Proliferative)
- No cross-extreme confusions observed

## Explainability

Grad-CAM visualizations show which retinal regions influenced each prediction. The model attends to areas consistent with known DR markers (hemorrhages, exudates). Coarse spatial resolution is a known limitation of Grad-CAM with ResNet-ViT-based attention maps would produce finer localization.

## Limitations & Next Steps

- Grad-CAM heatmaps are coarse (7×7 feature maps upsampled to 224×224)
- Grade 3 confusion with Grade 4 due to limited samples (154 training images)
- Next: Vision Transformer (ViT) for sharper attention maps, Optuna for hyperparameter search, ordinal regression loss for better grade boundary discrimination

## Setup

```bash
conda create -n retina python=3.10
conda activate retina
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pandas numpy matplotlib scikit-learn tqdm grad-cam jupyter ipykernel
```

Download the dataset from [Academic Torrents](https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd) and place it in `data/`.
