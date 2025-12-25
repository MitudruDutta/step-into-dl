# 🔄 Data Augmentation

## What is Data Augmentation?

**Data augmentation** is a technique to artificially expand your training dataset by applying various transformations to existing images. This helps the model learn invariances and reduces overfitting without collecting more data.

```
Original Image → [Transformations] → Multiple Training Samples

One cat image becomes:
  - Flipped cat
  - Rotated cat
  - Zoomed cat
  - Brighter cat
  - Cropped cat
  ... all still labeled as "cat"
```

---

## Why Data Augmentation Works

### The Core Idea

A cat is still a cat whether it's:

- On the left or right side of the image
- Slightly rotated
- In bright or dim lighting
- Partially visible

By showing the model these variations, it learns to recognize the **essential features** rather than memorizing specific pixel patterns.

### Benefits

| Benefit                              | Explanation                                 |
| ------------------------------------ | ------------------------------------------- |
| **Reduces overfitting**              | Model sees more variety, generalizes better |
| **Increases effective dataset size** | 1000 images → effectively 10,000+           |
| **Improves robustness**              | Model handles real-world variations         |
| **Free data**                        | No additional collection cost               |

---

## Common Augmentation Techniques

### Geometric Transformations

**Horizontal Flip**

```
┌─────────┐      ┌─────────┐
│  🐱    │  →   │    🐱  │
│   →    │      │    ←   │
└─────────┘      └─────────┘
Most common, works for most objects
Don't use for: text, directional objects
```

**Vertical Flip**

```
┌─────────┐      ┌─────────┐
│  🐱    │  →   │  🐱    │
│   ↑    │      │   ↓    │
└─────────┘      └─────────┘
Use for: aerial/satellite images, microscopy
Don't use for: natural scenes (sky should be up)
```

**Rotation**

```
┌─────────┐      ┌─────────┐
│  🐱    │  →   │    🐱  │
│         │      │   ↗    │
└─────────┘      └─────────┘
Typical range: ±10° to ±30°
Large rotations may not preserve label validity
```

**Random Crop**

```
┌─────────────┐      ┌───────┐
│  ┌─────┐   │  →   │ 🐱   │
│  │ 🐱 │   │      │       │
│  └─────┘   │      └───────┘
└─────────────┘
Forces model to recognize partial objects
Common: RandomResizedCrop (crop + resize)
```

**Scaling/Zoom**

```
┌─────────┐      ┌─────────┐
│   🐱   │  →   │  🐱🐱  │
│         │      │  🐱🐱  │
└─────────┘      └─────────┘
Typical range: 0.8× to 1.2×
Helps with scale invariance
```

**Translation/Shift**

```
┌─────────┐      ┌─────────┐
│ 🐱     │  →   │     🐱 │
│         │      │         │
└─────────┘      └─────────┘
Shift image horizontally/vertically
Typical: ±10% of image size
```

### Color/Photometric Transformations

**Brightness Adjustment**

```
Original → Brighter (+30%) or Darker (-30%)
Simulates different lighting conditions
```

**Contrast Adjustment**

```
Original → Higher contrast or Lower contrast
Helps with varying image quality
```

**Saturation Adjustment**

```
Original → More vivid or More muted colors
Simulates different camera settings
```

**Hue Shift**

```
Original → Slight color shift
Use carefully — may change object identity
```

**Grayscale Conversion**

```
RGB → Grayscale (randomly)
Forces model to use shape, not just color
```

### Advanced Techniques

**Cutout / Random Erasing**

```
┌─────────┐      ┌─────────┐
│  🐱    │  →   │  🐱    │
│   👀   │      │   ██   │
└─────────┘      └─────────┘
Randomly mask rectangular regions
Forces model to use multiple features
```

**Mixup**

```
Image A (cat) + Image B (dog) → Blended image
Label: 0.6 cat + 0.4 dog

Creates soft labels, improves calibration
```

**CutMix**

```
┌─────────┐      ┌─────────┐
│  🐱    │      │  🐱 🐕 │
│         │  +   │         │
└─────────┘      └─────────┘
Paste patch from one image onto another
Label proportional to area
```

---

## Augmentation in PyTorch

### Using torchvision.transforms

```python
from torchvision import transforms

# Basic training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test transform (NO augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Using torchvision.transforms.v2 (Modern API)

```python
from torchvision.transforms import v2

# Modern augmentation pipeline with v2
train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Random Erasing (Cutout)

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])
```

### AutoAugment (Learned Policies)

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Use policies learned from ImageNet, CIFAR10, or SVHN
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# RandAugment - simpler alternative
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandAugment(num_ops=2, magnitude=9),  # Apply 2 random transforms
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Applying Augmentation to Datasets

```python
from torchvision import datasets
from torch.utils.data import DataLoader

# Load dataset with augmentation
train_dataset = datasets.ImageFolder(
    root='data/train',
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root='data/val',
    transform=val_transform  # No augmentation!
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

---

## Using Albumentations Library

[Albumentations](https://albumentations.ai/) is a fast augmentation library with more options.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Albumentations pipeline
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom dataset to use Albumentations
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
```

---

## Best Practices

### 1. Match Augmentation to Your Domain

| Domain               | Recommended Augmentations                       |
| -------------------- | ----------------------------------------------- |
| **Natural images**   | Flip, rotation, crop, color jitter              |
| **Medical imaging**  | Rotation, elastic deformation, intensity shifts |
| **Satellite/Aerial** | All flips, 90° rotations, scale                 |
| **Documents/Text**   | Slight rotation, perspective, noise             |
| **Faces**            | Horizontal flip, small rotations only           |

### 2. Don't Over-Augment

```
Too Little Augmentation:
  - Model overfits to training data
  - Poor generalization

Too Much Augmentation:
  - Model can't learn meaningful features
  - Slow convergence
  - May introduce invalid samples

Just Right:
  - Model generalizes well
  - Learns robust features
```

### 3. Never Augment Validation/Test Data

```python
# ✅ CORRECT
train_dataset = Dataset(transform=train_transform)  # With augmentation
val_dataset = Dataset(transform=val_transform)      # Only resize + normalize

# ❌ WRONG
train_dataset = Dataset(transform=train_transform)
val_dataset = Dataset(transform=train_transform)    # Don't augment validation!
```

### 4. Augmentation During Training Only

```
Training:
  Original → [Random Augmentation] → Model

Inference:
  Original → [Deterministic Preprocessing] → Model
```

### 5. Test Time Augmentation (TTA)

For improved inference accuracy, apply augmentations at test time and average predictions:

```python
def predict_with_tta(model, image, n_augments=5):
    """Apply test-time augmentation and average predictions."""
    model.eval()
    predictions = []

    # Original prediction
    with torch.no_grad():
        pred = model(image)
        predictions.append(pred)

    # Augmented predictions
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        # Add more transforms as needed
    ]

    for transform in tta_transforms:
        augmented = transform(image)
        with torch.no_grad():
            pred = model(augmented)
            predictions.append(pred)

    # Average all predictions
    return torch.stack(predictions).mean(dim=0)
```

---

## Quick Reference: Transform Parameters

| Transform              | Key Parameters                          | Typical Values            |
| ---------------------- | --------------------------------------- | ------------------------- |
| `RandomHorizontalFlip` | `p`                                     | 0.5                       |
| `RandomRotation`       | `degrees`                               | 10-30                     |
| `RandomResizedCrop`    | `scale`                                 | (0.8, 1.0)                |
| `ColorJitter`          | `brightness, contrast, saturation, hue` | 0.1-0.3                   |
| `RandomErasing`        | `p, scale`                              | p=0.5, scale=(0.02, 0.33) |
| `GaussianBlur`         | `kernel_size`                           | 3-7                       |
| `RandomAffine`         | `degrees, translate, scale`             | varies                    |

---

## Summary

| Concept           | Key Point                                       |
| ----------------- | ----------------------------------------------- |
| **Purpose**       | Expand dataset artificially, reduce overfitting |
| **Geometric**     | Flips, rotations, crops, translations           |
| **Photometric**   | Brightness, contrast, saturation, hue           |
| **Advanced**      | Cutout, Mixup, CutMix, AutoAugment              |
| **Best Practice** | Only augment training data, match domain        |
| **Libraries**     | torchvision.transforms, Albumentations          |

---

## Next Steps

Continue to [07-Transfer-Learning.md](07-Transfer-Learning.md) to learn how to leverage pre-trained models and fine-tune them for your specific tasks.

---

_Data augmentation is one of the most effective techniques for improving model performance without collecting more data. Master it to build robust computer vision models!_
