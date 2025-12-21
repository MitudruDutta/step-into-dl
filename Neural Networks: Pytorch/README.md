# 🚀 Neural Networks in PyTorch: Implementation & Classification

This module covers the practical transition from theoretical neural networks to building them in **PyTorch**. We explore the core architecture classes, efficient data handling with DataLoaders, and the critical math behind selecting the right loss functions for classification.

---

## 📚 Documentation

| File | Topic | Description |
|------|-------|-------------|
| [01-nn-Module.md](01-nn-Module.md) | nn.Module | Building models, subclassing, forward method |
| [02-Datasets-DataLoaders.md](02-Datasets-DataLoaders.md) | Data Pipelines | Datasets, DataLoaders, batching, shuffling |
| [03-Binary-Cross-Entropy.md](03-Binary-Cross-Entropy.md) | BCE Loss | Binary classification, why MSE fails |
| [04-Categorical-Cross-Entropy.md](04-Categorical-Cross-Entropy.md) | Cross Entropy | Multi-class classification, softmax |
| [05-Training-Loop.md](05-Training-Loop.md) | Training Loop | Complete workflow, common mistakes |

---

## 💻 Notebooks

| Notebook | Description |
|----------|-------------|
| [log_loss.ipynb](log_loss.ipynb) | MSE vs Binary Cross Entropy comparison |
| [cross_entropy_loss.ipynb](cross_entropy_loss.ipynb) | Cross Entropy for multi-class classification |
| [dataset_dataloader.ipynb](dataset_dataloader.ipynb) | FashionMNIST dataset and DataLoader usage |
| [handwritten_digits.ipynb](handwritten_digits.ipynb) | Complete MNIST digit classifier with evaluation |

---

## 🎯 Learning Path

1. **nn.Module** → Learn to build custom models
2. **Datasets & DataLoaders** → Efficient data handling
3. **Binary Cross Entropy** → Binary classification loss
4. **Categorical Cross Entropy** → Multi-class classification loss
5. **Training Loop** → Put it all together

---

## 🔑 Quick Reference

### Model Definition Pattern

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = ...
    
    def forward(self, x):
        return self.layers(x)
```

### Loss Function Selection

| Problem Type | Loss Function | Final Activation |
|--------------|---------------|------------------|
| Regression | MSE, MAE | None (linear) |
| Binary Classification | BCEWithLogitsLoss | None |
| Multi-Class | CrossEntropyLoss | None |
| Multi-Label | BCEWithLogitsLoss | None |

### Training Loop Essentials

```python
model.train()
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
```

### DataLoader Setup

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```


---

## ⚠️ Common Mistakes

1. **Using softmax with CrossEntropyLoss** — softmax is already included
2. **Forgetting `optimizer.zero_grad()`** — gradients accumulate by default
3. **Wrong mode** — use `model.train()` for training, `model.eval()` for inference
4. **Forgetting `torch.no_grad()`** — wastes memory during inference
5. **Data on wrong device** — move data to same device as model

---

## 📖 Prerequisites

Before this module, you should understand:
- PyTorch tensor basics (see Pytorch/ module)
- Autograd and gradient computation
- Neural network fundamentals (see Neural Networks: Basics/)
- Training concepts (see Neural Network: Training/)

---

*This module bridges theory and practice. Understanding these PyTorch fundamentals prepares you for building real-world classification models.*
