# 🔄 Cross-Validation

Cross-validation is a technique for robustly evaluating model performance during hyperparameter tuning. It helps ensure that your chosen hyperparameters generalize well, not just perform well on a single validation split.

---

## Why Cross-Validation Matters

### The Problem with Single Splits

```
Single train/validation split:

All Data: [████████████████████████████████]
          [    Training (80%)    ][Val 20%]

Problems:
  - Validation set might be "easy" or "hard" by chance
  - Hyperparameters may overfit to this specific split
  - Results vary significantly with different random splits
```

### The Cross-Validation Solution

```
K-Fold Cross-Validation (K=5):

Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]

Each data point is used for validation exactly once.
Final score = average of all fold scores.
```

---

## K-Fold Cross-Validation

### How It Works

```
1. Split data into K equal parts (folds)
   ↓
2. For each fold i = 1 to K:
   a. Use fold i as validation set
   b. Use remaining K-1 folds as training set
   c. Train model and evaluate on fold i
   d. Record score
   ↓
3. Calculate mean and std of K scores
   ↓
4. Report: score = mean ± std
```

### Implementation

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate(model_fn, X, y, n_folds=5):
    """
    Perform K-fold cross-validation.
    
    Args:
        model_fn: Function that creates and trains a model
        X: Features
        y: Labels
        n_folds: Number of folds
    
    Returns:
        mean_score, std_score, all_scores
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = model_fn()
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"Fold {fold+1}: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nCV Score: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score, scores


# Usage
from sklearn.ensemble import RandomForestClassifier

mean, std, scores = cross_validate(
    lambda: RandomForestClassifier(n_estimators=100),
    X, y,
    n_folds=5
)
```

### Scikit-learn cross_val_score

```python
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)

# Perform 5-fold CV
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='accuracy'
)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

---

## Choosing K (Number of Folds)

### Common Choices

| K | Name | Pros | Cons |
|---|------|------|------|
| **5** | 5-Fold | Good balance, common default | - |
| **10** | 10-Fold | Lower bias, more robust | Slower |
| **N** | Leave-One-Out (LOO) | Lowest bias | Very slow, high variance |
| **3** | 3-Fold | Fast | Higher variance |

### Guidelines

```
Small dataset (< 1000 samples):
  → Use K=10 or LOO for more reliable estimates

Medium dataset (1000-10000):
  → Use K=5 (good balance)

Large dataset (> 10000):
  → Use K=3-5 (more data per fold anyway)

Very expensive training:
  → Use K=3 to reduce computation
```

---

## Stratified K-Fold

For classification problems, stratified K-fold ensures each fold has the same class distribution as the full dataset.

### Why Stratification Matters

```
Imbalanced dataset: 90% class A, 10% class B

Regular K-Fold (bad):
  Fold 1: 95% A, 5% B   ← Unrepresentative
  Fold 2: 85% A, 15% B  ← Unrepresentative
  ...

Stratified K-Fold (good):
  Fold 1: 90% A, 10% B  ← Same as original
  Fold 2: 90% A, 10% B  ← Same as original
  ...
```

### Implementation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# With cross_val_score (automatically stratified for classifiers)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

# Manual iteration
for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Verify stratification
    print(f"Fold {fold+1} - Train class dist: {np.bincount(y_train) / len(y_train)}")
    print(f"Fold {fold+1} - Val class dist: {np.bincount(y_val) / len(y_val)}")
```

---

## Cross-Validation with Hyperparameter Tuning

### Nested Cross-Validation

For unbiased evaluation of the tuning process itself:

```
Outer CV: Evaluates the entire tuning process
  │
  └── Inner CV: Tunes hyperparameters
  
Outer Fold 1:
  ├── Inner CV: Find best hyperparameters
  └── Evaluate best model on outer test fold

Outer Fold 2:
  ├── Inner CV: Find best hyperparameters
  └── Evaluate best model on outer test fold
  
...
```

### Implementation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner CV for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32)],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001]
}

inner_cv = GridSearchCV(
    MLPClassifier(max_iter=500),
    param_grid,
    cv=3,  # Inner 3-fold
    scoring='accuracy'
)

# Outer CV for unbiased evaluation
outer_scores = cross_val_score(
    inner_cv,
    X, y,
    cv=5,  # Outer 5-fold
    scoring='accuracy'
)

print(f"Nested CV Score: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")
```

---

## Cross-Validation for Time Series

Standard K-fold doesn't work for time series because it would use future data to predict the past.

### Time Series Split

```
Standard K-Fold (wrong for time series):
  Fold 1: [Val][Train][Train][Train][Train]
          Using future to predict past!

Time Series Split (correct):
  Fold 1: [Train][Val]
  Fold 2: [Train][Train][Val]
  Fold 3: [Train][Train][Train][Val]
  Fold 4: [Train][Train][Train][Train][Val]
  
  Always predicting future from past.
```

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}:")
    print(f"  Train: indices {train_idx[0]} to {train_idx[-1]}")
    print(f"  Val: indices {val_idx[0]} to {val_idx[-1]}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

---

## Cross-Validation with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import numpy as np

def pytorch_cross_validate(model_class, dataset, n_folds=5, epochs=50, lr=0.001):
    """Cross-validation for PyTorch models."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    indices = np.arange(len(dataset))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32)
        
        # Create fresh model for each fold
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        fold_scores.append(accuracy)
        print(f"  Accuracy: {accuracy:.4f}")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\nCV Score: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score, fold_scores
```

---

## Interpreting CV Results

### Understanding the Output

```
CV Scores: [0.92, 0.89, 0.91, 0.88, 0.93]
Mean: 0.906
Std: 0.019

Interpretation:
  - Expected performance: ~90.6%
  - Typical variation: ±1.9%
  - 95% confidence: roughly 90.6% ± 3.8%
```

### Comparing Models

```python
# Model A
scores_a = cross_val_score(model_a, X, y, cv=5)
print(f"Model A: {scores_a.mean():.4f} ± {scores_a.std():.4f}")

# Model B
scores_b = cross_val_score(model_b, X, y, cv=5)
print(f"Model B: {scores_b.mean():.4f} ± {scores_b.std():.4f}")

# Statistical comparison
from scipy import stats
t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference")
```

### Red Flags

```
High variance across folds:
  Scores: [0.95, 0.72, 0.88, 0.91, 0.65]
  Std: 0.12 (very high!)
  
  Possible causes:
    - Small dataset
    - Imbalanced classes (use stratified)
    - Data leakage in some folds
    - Model is unstable
```

---

## Best Practices

### 1. Always Use Stratification for Classification

```python
# Good
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Bad for imbalanced data
cv = KFold(n_splits=5)
```

### 2. Shuffle Before Splitting

```python
# Good
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Bad (data might be ordered)
kfold = KFold(n_splits=5, shuffle=False)
```

### 3. Use Consistent Random State

```python
# Reproducible results
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

### 4. Report Both Mean and Std

```python
# Good
print(f"Accuracy: {mean:.4f} ± {std:.4f}")

# Incomplete
print(f"Accuracy: {mean:.4f}")  # Missing variance info
```

---

## Key Takeaways

1. **Cross-validation provides robust estimates** by using all data for validation
2. **K=5 is a good default** for most problems
3. **Use stratified K-fold** for classification to preserve class distribution
4. **Time series needs special handling** — use TimeSeriesSplit
5. **Report mean ± std** to show reliability of estimates
6. **Nested CV** gives unbiased evaluation of the tuning process
7. **High variance across folds** indicates potential problems

---

*Cross-validation is essential for reliable hyperparameter tuning. Without it, you risk selecting hyperparameters that only work well on a lucky validation split.*
