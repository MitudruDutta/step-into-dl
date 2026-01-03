# 🎛️ Model Optimization: Hyperparameter Tuning

This module provides an in-depth exploration of **Hyperparameter Tuning**—the process of finding the optimal settings for a neural network to maximize performance and efficiency.

---

## 📚 Topics

| #   | Topic                                                                     | Description                                                       |
| --- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | [What is Hyperparameter Tuning](docs/01-What-is-Hyperparameter-Tuning.md) | Hyperparameters vs parameters, categories, why tuning matters     |
| 2   | [Manual Tuning](docs/02-Manual-Tuning.md)                                 | Learning rate finder, batch size guidelines, systematic approach  |
| 3   | [Grid Search](docs/03-Grid-Search.md)                                     | Exhaustive search, GridSearchCV, when to use                      |
| 4   | [Random Search](docs/04-Random-Search.md)                                 | Efficient sampling, log-uniform distributions, RandomizedSearchCV |
| 5   | [Bayesian Optimization](docs/05-Bayesian-Optimization.md)                 | Optuna, pruning, intelligent search strategies                    |
| 6   | [Cross-Validation](docs/06-Cross-Validation.md)                           | K-fold CV, stratified splits, time series validation              |
| 7   | [Practical Guidelines](docs/07-Practical-Guidelines.md)                   | Tuning order, starting points, common mistakes                    |

---

## 🎯 Learning Objectives

After completing this module, you will be able to:

- Distinguish between hyperparameters and model parameters
- Use the learning rate finder to identify optimal learning rates
- Implement Grid Search and Random Search for hyperparameter optimization
- Apply Bayesian optimization with Optuna for efficient tuning
- Use cross-validation for robust performance estimation
- Avoid common hyperparameter tuning mistakes

---

## 🔑 Key Concepts

### Hyperparameters vs Model Parameters

| Aspect             | Hyperparameters           | Model Parameters |
| ------------------ | ------------------------- | ---------------- |
| **When set**       | Before training           | During training  |
| **How determined** | Manual or search          | Gradient descent |
| **Examples**       | Learning rate, batch size | Weights, biases  |

### Search Methods Comparison

| Method            | Efficiency | Best For                                 |
| ----------------- | ---------- | ---------------------------------------- |
| Manual            | Low        | Quick experiments                        |
| Grid Search       | Low        | Small spaces (< 100 combinations)        |
| Random Search     | Medium     | Large spaces, initial exploration        |
| Bayesian (Optuna) | High       | Production tuning, expensive evaluations |

### Tuning Priority Order

```
1. Learning Rate      ← Most important
2. Architecture       ← Model capacity
3. Batch Size         ← Training dynamics
4. Regularization     ← Only if overfitting
5. Other              ← Usually defaults work
```

---

## 📓 Notebooks

| Notebook                                             | Description                                       |
| ---------------------------------------------------- | ------------------------------------------------- |
| [optuna_tuning.ipynb](notebooks/optuna_tuning.ipynb) | Bayesian optimization with Optuna on real dataset |

---

## 🚀 Quick Start

### 1. Start with Good Defaults

```python
# Training
learning_rate = 0.001  # For Adam
batch_size = 32
epochs = 100  # With early stopping

# Architecture
hidden_layers = 2
neurons = 128
activation = 'relu'

# Regularization (add if overfitting)
dropout = 0.0
weight_decay = 0.0
```

### 2. Use Optuna for Efficient Tuning

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden = trial.suggest_int('hidden', 32, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    model = build_model(hidden, dropout)
    train(model, lr)
    return evaluate(model)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### 3. Always Use Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## ⚠️ Common Mistakes to Avoid

1. **Tuning on test data** — Use validation set or cross-validation
2. **Too few trials** — Need 50+ for random search, 100+ for Optuna
3. **Ignoring interactions** — LR and batch size work together
4. **Not using early stopping** — Wastes compute on bad configurations
5. **Uniform scale for LR** — Use log scale for learning rate

---

## 📖 Prerequisites

Before starting this module, you should be familiar with:

- Neural network basics (layers, activations)
- PyTorch fundamentals (tensors, training loops)
- Basic training concepts (loss functions, optimizers)

---

## 🔗 Related Modules

- [Model Optimization: Training Algorithms](../Model%20Optimization:%20Training%20Algorithms/) — Optimizers (Adam, SGD, etc.)
- [Model Optimization: Regularization Techniques](../Model%20Optimization:%20Regularization%20Techniques/) — Dropout, L2, BatchNorm

---

_Good hyperparameters can make the difference between a model that barely works and one that achieves state-of-the-art performance. Invest time in tuning!_
