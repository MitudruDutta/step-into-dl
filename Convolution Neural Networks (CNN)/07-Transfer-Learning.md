# 🔄 Transfer Learning

## What is Transfer Learning?

**Transfer learning** is a technique where a model trained on one task is reused as the starting point for a model on a different task. Instead of training from scratch, you leverage knowledge learned from large datasets (like ImageNet) and adapt it to your specific problem.

```
Traditional Approach:
  Your Data → Train from Scratch → Model
  ❌ Needs lots of data
  ❌ Long training time
  ❌ May not converge well

Transfer Learning Approach:
  ImageNet Data → Pre-trained Model → Fine-tune on Your Data → Your Model
  ✅ Works with limited data
  ✅ Faster training
  ✅ Better performance
```

---

## Why Transfer Learning Works

### Feature Hierarchy in CNNs

CNNs learn features in a hierarchical manner:

```
Layer 1-2: Generic low-level features
┌─────────────────────────────────────┐
│  Edges, colors, textures            │
│  ─── ║║║ ░░░ ╱╲╱                    │
│  Useful for ANY image task          │
└─────────────────────────────────────┘
           ↓
Layer 3-4: Mid-level features
┌─────────────────────────────────────┐
│  Shapes, patterns, parts            │
│  ○ □ ◇ 🔺 eyes, wheels, fur        │
│  Somewhat task-specific             │
└─────────────────────────────────────┘
           ↓
Layer 5+: High-level features
┌─────────────────────────────────────┐
│  Object parts, semantic concepts    │
│  faces, cars, animals               │
│  Task-specific                      │
└─────────────────────────────────────┘
           ↓
Final Layer: Task-specific
┌─────────────────────────────────────┐
│  Classification head                │
│  "cat", "dog", "car"                │
│  Must be replaced for new task      │
└─────────────────────────────────────┘
```

### The Key Insight

Early layers learn **universal features** that transfer well across tasks. Only the later layers need to be adapted for your specific problem.

---

## Transfer Learning Strategies

### Strategy 1: Feature Extraction (Freeze All)

Use the pre-trained model as a fixed feature extractor.

```
Pre-trained CNN (FROZEN)     New Classifier
┌─────────────────────┐     ┌──────────────┐
│  Conv1 → Conv2 →    │ →   │  FC → Output │
│  Conv3 → Conv4 →    │     │  (trainable) │
│  Conv5 (frozen)     │     └──────────────┘
└─────────────────────┘

When to use:
  ✅ Very small dataset (< 1000 images)
  ✅ Similar domain to pre-trained model
  ✅ Limited compute resources
```

### Strategy 2: Fine-tuning (Unfreeze Some)

Freeze early layers, train later layers + classifier.

```
Pre-trained CNN               New Classifier
┌─────────────────────┐     ┌──────────────┐
│  Conv1 → Conv2 →    │ →   │  FC → Output │
│  (frozen)           │     │  (trainable) │
│  Conv3 → Conv4 →    │     └──────────────┘
│  Conv5 (trainable)  │
└─────────────────────┘

When to use:
  ✅ Medium dataset (1000-10000 images)
  ✅ Somewhat similar domain
  ✅ Want better performance than feature extraction
```

### Strategy 3: Full Fine-tuning (Unfreeze All)

Train the entire network with a small learning rate.

```
Pre-trained CNN (all trainable)   New Classifier
┌─────────────────────────────┐  ┌──────────────┐
│  Conv1 → Conv2 → Conv3 →    │→ │  FC → Output │
│  Conv4 → Conv5              │  │  (trainable) │
│  (trainable, small LR)      │  └──────────────┘
└─────────────────────────────┘

When to use:
  ✅ Large dataset (10000+ images)
  ✅ Different domain from pre-trained model
  ✅ Have compute resources and time
```

### Strategy Comparison

| Strategy            | Data Size | Training Speed | Performance | Compute |
| ------------------- | --------- | -------------- | ----------- | ------- |
| Feature Extraction  | Small     | Fast           | Good        | Low     |
| Partial Fine-tuning | Medium    | Medium         | Better      | Medium  |
| Full Fine-tuning    | Large     | Slow           | Best        | High    |

---

## Pre-trained Models in PyTorch

### Available Models

```python
import torchvision.models as models

# Classification models (ImageNet pre-trained)
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
efficientnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1')
mobilenet_v3 = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
convnext = models.convnext_tiny(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')  # Vision Transformer

# Check available weights
from torchvision.models import ResNet50_Weights
print(ResNet50_Weights.IMAGENET1K_V1)
print(ResNet50_Weights.IMAGENET1K_V2)  # Better accuracy
print(ResNet50_Weights.DEFAULT)        # Best available
```

### Model Comparison

| Model           | Parameters | Top-1 Acc | Speed     | Use Case       |
| --------------- | ---------- | --------- | --------- | -------------- |
| MobileNetV3     | 2.5M       | 67.7%     | Very Fast | Mobile/Edge    |
| EfficientNet-B0 | 5.3M       | 77.7%     | Fast      | Balanced       |
| ResNet50        | 25.6M      | 80.9%     | Medium    | General        |
| ConvNeXt-Tiny   | 28.6M      | 82.1%     | Medium    | Modern CNN     |
| ViT-B/16        | 86.6M      | 81.1%     | Slow      | Large datasets |

---

## Implementation Guide

### Step 1: Load Pre-trained Model

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet50
model = models.resnet50(weights='IMAGENET1K_V2')

# Check the final layer
print(model.fc)  # Linear(in_features=2048, out_features=1000)
```

### Step 2: Modify for Your Task

```python
# Option A: Replace the final layer
num_classes = 10  # Your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Option B: Replace with more complex head
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

### Step 3: Freeze Layers (Optional)

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final layer (always trainable)
for param in model.fc.parameters():
    param.requires_grad = True

# Or freeze specific layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### Step 4: Set Up Different Learning Rates

```python
# Higher LR for new layers, lower LR for pre-trained layers
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=1e-5)  # Default for other layers
```

---

## Complete Transfer Learning Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)

# Load pre-trained model
model = models.resnet50(weights='IMAGENET1K_V2')

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and replace fc
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Train the model
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f'Best Validation Accuracy: {best_acc:.2f}%')
```

---

## Gradual Unfreezing

A technique to progressively unfreeze layers during training:

```python
def gradual_unfreeze(model, epoch, unfreeze_schedule):
    """
    Gradually unfreeze layers based on epoch.

    unfreeze_schedule = {
        0: ['fc'],           # Epoch 0: only fc
        3: ['layer4', 'fc'], # Epoch 3: layer4 + fc
        6: ['layer3', 'layer4', 'fc'],  # Epoch 6: more layers
    }
    """
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Find the appropriate schedule for current epoch
    current_layers = []
    for e, layers in sorted(unfreeze_schedule.items()):
        if epoch >= e:
            current_layers = layers

    # Unfreeze specified layers
    for name, param in model.named_parameters():
        for layer_name in current_layers:
            if layer_name in name:
                param.requires_grad = True
                break

# Usage
unfreeze_schedule = {
    0: ['fc'],
    3: ['layer4', 'fc'],
    6: ['layer3', 'layer4', 'fc'],
    9: None  # Unfreeze all (set manually)
}

for epoch in range(num_epochs):
    gradual_unfreeze(model, epoch, unfreeze_schedule)
    # ... training code
```

---

## Domain Adaptation Guidelines

### Similar Domain (e.g., ImageNet → Dogs/Cats)

```python
# Use feature extraction or light fine-tuning
# Pre-trained features work well

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False

# Only train classifier
model.fc = nn.Linear(2048, num_classes)
```

### Different Domain (e.g., ImageNet → Medical Images)

```python
# Full fine-tuning with small learning rate
# Pre-trained weights provide good initialization

for param in model.parameters():
    param.requires_grad = True

# Use smaller learning rate for pre-trained layers
optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},  # Small LR
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # Normal LR
])
```

### Very Different Domain (e.g., ImageNet → Spectrograms)

```python
# May benefit more from training from scratch
# Or try pre-trained, then decide based on results

# Option 1: Pre-trained initialization
model = models.resnet50(weights='IMAGENET1K_V2')

# Option 2: Random initialization
model = models.resnet50(weights=None)

# Compare both approaches on your data
```

---

## Using timm Library

[timm](https://github.com/huggingface/pytorch-image-models) provides many more pre-trained models:

```python
import timm

# List available models
print(timm.list_models(pretrained=True)[:10])

# List models matching a pattern
print(timm.list_models('*efficientnet*'))

# Load a pre-trained model
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=10)

# Get model configuration
model_cfg = timm.get_pretrained_cfg('efficientnet_b3')
print(f"Input size: {model_cfg.input_size}")
print(f"Mean: {model_cfg.mean}")
print(f"Std: {model_cfg.std}")

# Feature extraction mode
model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # No classifier
features = model(images)  # Returns feature vector
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Wrong Input Size

```python
# ❌ Wrong: Using wrong input size
transform = transforms.Resize((128, 128))  # Model expects 224x224

# ✅ Correct: Match model's expected input
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)  # Or use RandomResizedCrop(224) for training
])
```

### Pitfall 2: Forgetting to Normalize

```python
# ❌ Wrong: No normalization
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# ✅ Correct: Use ImageNet normalization for ImageNet pre-trained models
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Pitfall 3: Too High Learning Rate

```python
# ❌ Wrong: Same high LR for all layers
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ✅ Correct: Lower LR for pre-trained layers
optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### Pitfall 4: Not Setting Model to Eval Mode

```python
# ❌ Wrong: Forgetting eval mode during inference
predictions = model(test_images)

# ✅ Correct: Set eval mode (affects BatchNorm, Dropout)
model.eval()
with torch.no_grad():
    predictions = model(test_images)
```

---

## Summary

| Concept                | Key Point                                         |
| ---------------------- | ------------------------------------------------- |
| **What**               | Reuse pre-trained models for new tasks            |
| **Why**                | Faster training, better results, less data needed |
| **Feature Extraction** | Freeze all, train new classifier only             |
| **Fine-tuning**        | Unfreeze some/all layers, use small LR            |
| **Learning Rate**      | Lower for pre-trained, higher for new layers      |
| **Normalization**      | Use same stats as pre-training (ImageNet)         |
| **Libraries**          | torchvision.models, timm                          |

---

## Best Practices Checklist

- [ ] Choose appropriate strategy based on dataset size
- [ ] Use correct input size and normalization
- [ ] Start with frozen layers, then gradually unfreeze
- [ ] Use different learning rates for different layers
- [ ] Apply data augmentation to training data
- [ ] Monitor for overfitting (validation loss)
- [ ] Save the best model based on validation accuracy
- [ ] Set model to eval mode during inference

---

## Next Steps

You've completed the CNN module! Here's what to explore next:

- Apply transfer learning to the CIFAR10 or CALTECH101 notebooks
- Experiment with different pre-trained models
- Try object detection with pre-trained backbones
- Explore Vision Transformers (ViT) for image classification

---

_Transfer learning is one of the most powerful techniques in deep learning. It democratizes AI by allowing anyone to build state-of-the-art models without massive datasets or compute resources!_
