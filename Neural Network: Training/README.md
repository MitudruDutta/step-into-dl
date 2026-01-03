# 📉 Neural Networks: Training & Optimization

This module covers how neural networks learn—from the mathematics of backpropagation to practical training strategies and debugging techniques.

---

## 📚 Documentation

| File                                                        | Topic               | Description                                                      |
| ----------------------------------------------------------- | ------------------- | ---------------------------------------------------------------- |
| [01-Backpropagation.md](docs/01-Backpropagation.md)         | Backpropagation     | How networks learn from errors, chain rule, gradient computation |
| [02-Gradient-Descent.md](docs/02-Gradient-Descent.md)       | Gradient Descent    | Optimization fundamentals, learning rate, convergence            |
| [03-GD-Variants.md](docs/03-GD-Variants.md)                 | GD Variants         | Batch vs Mini-Batch vs SGD, batch size selection                 |
| [04-Optimizers.md](docs/04-Optimizers.md)                   | Advanced Optimizers | Adam, SGD+Momentum, RMSprop, AdamW                               |
| [05-Monitoring-Training.md](docs/05-Monitoring-Training.md) | Monitoring          | Metrics, debugging, early stopping, checkpointing                |

---

## 💻 Notebooks

| Notebook                                                           | Description                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------------------ |
| [data_generation.ipynb](notebooks/data_generation.ipynb)           | Generate synthetic employee bonus dataset with known weights |
| [gradient_descent.ipynb](notebooks/gradient_descent.ipynb)         | Implement gradient descent from scratch in PyTorch           |
| [gd_vs_mini_gd_vs_sgd.ipynb](notebooks/gd_vs_mini_gd_vs_sgd.ipynb) | Compare Batch GD, Mini-Batch GD, and SGD                     |

---

## 🗃️ Dataset Notes

This module is designed to be runnable **without publishing or bundling any real-world dataset**.

- The exercises use a **synthetic “employee bonus” dataset** generated inside
  [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb).
- The generated file(s) are written under this module’s [data/](data/) folder (if you choose to save them).
- Because the target relationship is known (the notebook constructs it), it’s ideal for learning/debugging gradient descent and optimizer behavior.

If you don’t see any files under [data/](data/), just run the data generation notebook first.

## 🎯 Learning Path

1. **Start with Backpropagation** → Understand how gradients flow through networks
2. **Learn Gradient Descent** → Master the core optimization algorithm
3. **Compare GD Variants** → Choose the right approach for your data
4. **Explore Optimizers** → Use modern optimizers for faster convergence
5. **Monitor Training** → Debug issues and know when to stop

---

## 🔑 Key Concepts

### The Training Loop

```
1. Forward Pass    → Compute predictions
2. Calculate Loss  → Measure error
3. Backward Pass   → Compute gradients
4. Update Weights  → Adjust parameters
5. Repeat          → Until convergence
```

### Critical Hyperparameters

| Parameter     | Typical Values | Impact             |
| ------------- | -------------- | ------------------ |
| Learning Rate | 0.001 - 0.1    | Speed vs stability |
| Batch Size    | 32 - 256       | Memory vs noise    |
| Epochs        | 10 - 1000      | Training duration  |
| Momentum      | 0.9            | Smooths updates    |

### Optimizer Quick Reference

| Task               | Recommended Optimizer |
| ------------------ | --------------------- |
| Default choice     | Adam                  |
| Transformers/NLP   | AdamW                 |
| CNNs (with tuning) | SGD + Momentum        |
| RNNs               | Adam or RMSprop       |

---

## ⚠️ Common Issues

| Problem             | Symptom                  | Solution                    |
| ------------------- | ------------------------ | --------------------------- |
| Overfitting         | Val loss increases       | Early stopping, dropout     |
| Underfitting        | Both losses high         | Bigger model, train longer  |
| Vanishing gradients | Early layers don't learn | ReLU, batch norm            |
| Exploding gradients | Loss = NaN               | Gradient clipping, lower LR |

---

## 📖 Prerequisites

Before this module, you should understand:

- Neural network architecture (layers, neurons)
- Activation functions
- Basic calculus (derivatives)
- PyTorch tensors and autograd

---

_Training is where theory meets practice. Master these concepts to build models that actually learn._
