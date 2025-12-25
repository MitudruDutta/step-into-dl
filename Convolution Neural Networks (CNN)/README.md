# 🖼️ Convolutional Neural Networks (CNNs)

This module provides an in-depth exploration of **Convolutional Neural Networks (CNNs)**—the specialized architecture that revolutionized computer vision. From basic convolution operations to transfer learning, we cover everything you need to build powerful image recognition systems.

---

## 📚 Topics

| File                                                   | Topic               | Description                                                  |
| ------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| [01-CNN-Fundamentals.md](01-CNN-Fundamentals.md)       | CNN Fundamentals    | Why CNNs work, convolution operation, feature hierarchies    |
| [02-Kernels-and-Filters.md](02-Kernels-and-Filters.md) | Kernels and Filters | How filters detect features, learned vs hand-crafted kernels |
| [03-Padding-and-Strides.md](03-Padding-and-Strides.md) | Padding and Strides | Controlling output dimensions, spatial management            |
| [04-Pooling-Layers.md](04-Pooling-Layers.md)           | Pooling Layers      | Downsampling, max pooling, average pooling                   |
| [05-CNN-Architectures.md](05-CNN-Architectures.md)     | CNN Architectures   | LeNet, AlexNet, VGG, ResNet, and modern designs              |
| [06-Data-Augmentation.md](06-Data-Augmentation.md)     | Data Augmentation   | Expanding training data with transformations                 |
| [07-Transfer-Learning.md](07-Transfer-Learning.md)     | Transfer Learning   | Using pre-trained models, fine-tuning strategies             |

---

## 🎯 Learning Path

1. **Understand the basics** → [01-CNN-Fundamentals.md](01-CNN-Fundamentals.md)
2. **Learn about filters** → [02-Kernels-and-Filters.md](02-Kernels-and-Filters.md)
3. **Master spatial control** → [03-Padding-and-Strides.md](03-Padding-and-Strides.md)
4. **Explore pooling** → [04-Pooling-Layers.md](04-Pooling-Layers.md)
5. **Study architectures** → [05-CNN-Architectures.md](05-CNN-Architectures.md)
6. **Augment your data** → [06-Data-Augmentation.md](06-Data-Augmentation.md)
7. **Leverage pre-trained models** → [07-Transfer-Learning.md](07-Transfer-Learning.md)

---

## 🔑 Key Concepts

### Why CNNs for Images?

Traditional neural networks treat images as flat vectors, losing spatial information. CNNs preserve spatial relationships through:

```
Fully Connected (Bad for images):
  Image (28×28) → Flatten → 784 neurons
  - Loses spatial structure
  - Too many parameters
  - No translation invariance

CNN (Designed for images):
  Image (28×28) → Conv layers → Preserve structure
  - Maintains spatial relationships
  - Parameter sharing (fewer weights)
  - Translation invariant
```

### The CNN Pipeline

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  FEATURE EXTRACTION                 │
│  ┌─────────┐   ┌─────────┐         │
│  │  Conv   │ → │  Pool   │ → ...   │
│  │ + ReLU  │   │         │         │
│  └─────────┘   └─────────┘         │
│  Detect edges   Reduce size        │
│  and textures   Keep important     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  CLASSIFICATION                     │
│  ┌─────────┐   ┌─────────┐         │
│  │ Flatten │ → │   FC    │ → Output│
│  │         │   │ Layers  │         │
│  └─────────┘   └─────────┘         │
│  Convert to    Make prediction     │
│  vector                            │
└─────────────────────────────────────┘
```

### Core Components at a Glance

| Component       | Purpose           | Key Parameters       |
| --------------- | ----------------- | -------------------- |
| **Convolution** | Extract features  | Kernel size, filters |
| **Activation**  | Add non-linearity | ReLU most common     |
| **Pooling**     | Reduce dimensions | Pool size, stride    |
| **Flatten**     | Convert to 1D     | -                    |
| **Dense/FC**    | Classification    | Neurons              |

---

## 📊 Quick Reference

### PyTorch CNN Building Blocks

```python
import torch.nn as nn

# Convolutional layer
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

# Pooling layers
nn.MaxPool2d(kernel_size, stride=None, padding=0)
nn.AvgPool2d(kernel_size, stride=None, padding=0)

# Batch normalization (common in CNNs)
nn.BatchNorm2d(num_features)

# Flatten for FC layers
nn.Flatten()

# Fully connected
nn.Linear(in_features, out_features)
```

### Simple CNN Example

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Output Size Formula

```
Output Size = (Input Size - Kernel Size + 2×Padding) / Stride + 1

Example:
  Input: 32×32
  Kernel: 3×3
  Padding: 1
  Stride: 1

  Output = (32 - 3 + 2×1) / 1 + 1 = 32×32 (same size)
```

---

## 🎓 Prerequisites

Before diving into CNNs, you should understand:

- Basic neural network concepts (neurons, layers, activation functions)
- PyTorch fundamentals (tensors, nn.Module)
- Training loops and backpropagation

---

## 📓 Notebooks

| Notebook                                                                 | Description                                                                       |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| [CIFAR10_image_classification.ipynb](CIFAR10_image_classification.ipynb) | Build a CNN from scratch and compare with ResNet-18 transfer learning on CIFAR-10 |
| [CALTECH101_classification.ipynb](CALTECH101_classification.ipynb)       | Multi-class classification with custom CNN, ResNet-18, and EfficientNet-B0        |

---

_CNNs are the foundation of modern computer vision. Master these concepts to build image classifiers, object detectors, and more!_
