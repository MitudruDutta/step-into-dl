# 🛡️ Model Optimization: Regularization Techniques

This module provides an in-depth exploration of **Regularization**—a critical set of techniques used to ensure that neural networks generalize well to unseen data rather than simply memorizing the training set.

---

## 📚 Topics

| File | Topic | Description |
|------|-------|-------------|
| [01-Understanding-Regularization.md](01-Understanding-Regularization.md) | Understanding Regularization | What is regularization, overfitting, and the bias-variance tradeoff |
| [02-Dropout.md](02-Dropout.md) | Dropout Regularization | Randomly deactivating neurons to prevent co-adaptation |
| [03-L1-L2-Regularization.md](03-L1-L2-Regularization.md) | L1 and L2 Regularization | Weight penalties, sparsity, and weight decay |
| [04-Batch-Normalization.md](04-Batch-Normalization.md) | Batch Normalization | Normalizing layer inputs for stable training |
| [05-Early-Stopping.md](05-Early-Stopping.md) | Early Stopping | Stopping training at the optimal point |
| [06-Data-Augmentation.md](06-Data-Augmentation.md) | Data Augmentation | Artificially expanding training data |

---

## 🎯 Learning Path

1. **Start with fundamentals** → [01-Understanding-Regularization.md](01-Understanding-Regularization.md)
2. **Learn Dropout** → [02-Dropout.md](02-Dropout.md)
3. **Understand weight penalties** → [03-L1-L2-Regularization.md](03-L1-L2-Regularization.md)
4. **Explore BatchNorm** → [04-Batch-Normalization.md](04-Batch-Normalization.md)
5. **Master early stopping** → [05-Early-Stopping.md](05-Early-Stopping.md)
6. **Expand your data** → [06-Data-Augmentation.md](06-Data-Augmentation.md)

---

## 🔑 Key Concepts

### Why Regularization Matters

Regularization prevents **overfitting**—when a model memorizes training data instead of learning generalizable patterns.

```
Without Regularization:
  Training Accuracy: 99%
  Test Accuracy: 72%  ← Overfitting!

With Regularization:
  Training Accuracy: 92%
  Test Accuracy: 90%  ← Good generalization!
```

### Regularization Techniques at a Glance

| Technique | How It Works | When to Use |
|-----------|--------------|-------------|
| **Dropout** | Randomly drops neurons | Deep networks, FC layers |
| **L2 (Weight Decay)** | Penalizes large weights | Default choice, always |
| **L1** | Produces sparse weights | Feature selection |
| **Batch Normalization** | Normalizes layer inputs | Deep networks |
| **Early Stopping** | Stops at optimal point | Always |
| **Data Augmentation** | Creates data variations | Limited data, images |

### Combining Techniques

```python
# A well-regularized network
class RegularizedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),     # BatchNorm
            nn.ReLU(),
            nn.Dropout(p=0.3),       # Dropout
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, 10)
        )

# L2 via optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Plus: Early stopping + Data augmentation in training loop
```

---

## 📊 Quick Reference

### Regularization Strategy by Overfitting Severity

| Severity | Recommended Approach |
|----------|---------------------|
| **Mild** | L2 regularization (weight_decay=1e-4) |
| **Moderate** | L2 + Dropout (p=0.3) + BatchNorm |
| **Severe** | Above + Data Augmentation |
| **Extreme** | All above + Early Stopping + Smaller model |

### PyTorch Cheat Sheet

```python
# Dropout
nn.Dropout(p=0.5)           # FC layers
nn.Dropout2d(p=0.25)        # Conv layers

# L2 Regularization
optimizer = optim.AdamW(params, lr=0.001, weight_decay=1e-4)

# Batch Normalization
nn.BatchNorm1d(features)    # FC layers
nn.BatchNorm2d(channels)    # Conv layers

# Data Augmentation
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(0.2, 0.2, 0.2)
```

---

---

## 📓 Notebooks

| Notebook | Topic | Description |
|----------|-------|-------------|
| [dropout_regularization.ipynb](dropout_regularization.ipynb) | Dropout | Compare models with/without dropout on Sonar dataset |
| [l2_regularization.ipynb](l2_regularization.ipynb) | L2 / Weight Decay | Effect of weight_decay on training dynamics |
| [batch_norm.ipynb](batch_norm.ipynb) | Batch Normalization | BatchNorm impact on MNIST training |
| [early_stopping.ipynb](early_stopping.ipynb) | Early Stopping | Implementing patience-based stopping |

---

## 🎓 Prerequisites

Before diving into this module, you should understand:
- Basic neural network architecture
- Training loops and loss functions
- Gradient descent and backpropagation

---

*Regularization is essential for building models that work in the real world. Start simple, monitor validation performance, and add regularization as needed.*
