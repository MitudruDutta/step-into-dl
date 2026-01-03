# 🔄 Data Augmentation

Data augmentation artificially expands the training set by creating modified versions of existing samples. It's one of the most effective regularization techniques, especially for image data, and is essential when training data is limited.

---

## Why Data Augmentation Works

### The Core Insight

Neural networks learn from examples. More diverse examples lead to better generalization:

```
Without augmentation:
  Model sees: 1000 cat images
  Learns: Cats look exactly like these 1000 images

With augmentation:
  Model sees: 1000 × 10 = 10,000 variations
  Learns: Cats can be rotated, scaled, different colors, etc.
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Increased data** | Effectively multiplies dataset size |
| **Invariance learning** | Model learns to ignore irrelevant variations |
| **Reduced overfitting** | Harder to memorize augmented data |
| **Better generalization** | Model handles real-world variations |

---

## Image Augmentation

### Common Transformations

```python
from torchvision import transforms

# Geometric transformations
transforms.RandomHorizontalFlip(p=0.5)      # Flip left-right
transforms.RandomVerticalFlip(p=0.5)        # Flip up-down
transforms.RandomRotation(degrees=15)        # Rotate ±15°
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # Random crop and resize
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Shift
transforms.RandomPerspective(distortion_scale=0.2)  # Perspective change

# Color transformations
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.RandomGrayscale(p=0.1)           # Convert to grayscale
transforms.GaussianBlur(kernel_size=3)      # Blur

# Normalization (always apply)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```


### Complete Training Pipeline

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (NO augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Apply to datasets
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Important: No Augmentation at Test Time

```python
# WRONG: Augmenting test data
test_transform = train_transform  # Don't do this!

# CORRECT: Only normalize test data
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Task-Specific Augmentation

### Image Classification

```python
classification_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Object Detection

For detection, you must transform both image AND bounding boxes:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

detection_transform = A.Compose([
    A.RandomResizedCrop(height=416, width=416, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
```

### Semantic Segmentation

For segmentation, transform image AND mask identically:

```python
segmentation_transform = A.Compose([
    A.RandomResizedCrop(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# Apply to both image and mask
transformed = segmentation_transform(image=image, mask=mask)
aug_image = transformed['image']
aug_mask = transformed['mask']
```

---

## Advanced Augmentation Techniques

### Cutout / Random Erasing

Randomly masks out rectangular regions:

```python
transforms.RandomErasing(
    p=0.5,           # Probability
    scale=(0.02, 0.33),  # Area range
    ratio=(0.3, 3.3),    # Aspect ratio range
    value=0          # Fill value (0 = black)
)
```

### Mixup

Blends two images and their labels:

```python
def mixup(images, labels, alpha=0.2):
    """
    Mixup: x = λ*x_i + (1-λ)*x_j
           y = λ*y_i + (1-λ)*y_j
    """
    batch_size = images.size(0)
    
    # Sample mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation for pairing
    index = torch.randperm(batch_size)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index]
    
    # Return mixed images and both label sets with lambda
    return mixed_images, labels, labels[index], lam

# In training loop
mixed_images, labels_a, labels_b, lam = mixup(images, labels)
outputs = model(mixed_images)
loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
```

### CutMix

Cuts and pastes patches between images:

```python
def cutmix(images, labels, alpha=1.0):
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    index = torch.randperm(batch_size)
    
    # Get random box
    W, H = images.size(3), images.size(2)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    return images, labels, labels[index], lam
```

### AutoAugment

Learned augmentation policies:

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform = transforms.Compose([
    transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### RandAugment

Simplified automatic augmentation:

```python
from torchvision.transforms import RandAugment

transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Augmentation for Other Domains

### Text Augmentation

```python
# Synonym replacement
def synonym_replacement(text, n=1):
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        words[idx] = get_synonym(words[idx])  # Use WordNet or similar
    return ' '.join(words)

# Random deletion
def random_deletion(text, p=0.1):
    words = text.split()
    return ' '.join([w for w in words if random.random() > p])

# Back-translation
def back_translate(text, src='en', pivot='de'):
    # Translate to pivot language and back
    translated = translate(text, src, pivot)
    return translate(translated, pivot, src)
```

### Audio Augmentation

```python
import torchaudio.transforms as T

# Time stretching
time_stretch = T.TimeStretch(hop_length=512, n_freq=201)

# Pitch shifting
pitch_shift = T.PitchShift(sample_rate=16000, n_steps=4)

# Add noise
def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

# Time masking (SpecAugment)
time_mask = T.TimeMasking(time_mask_param=80)
freq_mask = T.FrequencyMasking(freq_mask_param=27)
```

### Tabular Data Augmentation

```python
# SMOTE for imbalanced data
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature noise
def add_feature_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Mixup for tabular
def tabular_mixup(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    X_mixed = lam * X + (1 - lam) * X[idx]
    y_mixed = lam * y + (1 - lam) * y[idx]
    return X_mixed, y_mixed
```

---

## Augmentation Libraries

### Albumentations (Recommended for Images)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.Normalize(),
    ToTensorV2()
])

# Usage
augmented = transform(image=image)
aug_image = augmented['image']
```

### imgaug

```python
import imgaug.augmenters as iaa

transform = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])

aug_image = transform(image=image)
```

---

## Best Practices

### 1. Match Augmentation to Task

```python
# Medical imaging: Be careful with flips
# Chest X-rays are NOT symmetric - don't flip horizontally

# Satellite imagery: Rotation is fine
# Can rotate 90°, 180°, 270° freely

# Text: Preserve meaning
# Don't augment in ways that change the label
```

### 2. Start Simple, Add Complexity

```python
# Level 1: Basic
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(...)
])

# Level 2: Moderate
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(...)
])

# Level 3: Aggressive
transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomErasing(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### 3. Visualize Augmentations

```python
import matplotlib.pyplot as plt

def visualize_augmentations(image, transform, n=8):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i == 0:
            ax.imshow(image)
            ax.set_title('Original')
        else:
            aug_image = transform(image)
            ax.imshow(aug_image.permute(1, 2, 0))
            ax.set_title(f'Aug {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## Key Takeaways

1. **Data augmentation expands training data** by creating variations
2. **Only augment training data**, never validation/test
3. **Match augmentations to your task** and domain
4. **Start simple** and add complexity as needed
5. **Visualize augmentations** to ensure they make sense
6. **Use libraries** like Albumentations for efficiency
7. **Advanced techniques** (Mixup, CutMix) can further improve results

---

*Data augmentation is often the single most effective technique for improving model performance, especially with limited data. It's free extra data!*
