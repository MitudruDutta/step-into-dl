# 🔳 Grid Search

Grid search is the most straightforward hyperparameter tuning method—it exhaustively evaluates every combination of hyperparameters in a predefined search space.

---

## How Grid Search Works

### The Concept

Grid search creates a "grid" of all possible hyperparameter combinations and evaluates each one systematically.

```
Example Grid:
  learning_rate: [0.001, 0.01]
  batch_size: [32, 64]

All Combinations (2 × 2 = 4):
  1. lr=0.001, batch=32
  2. lr=0.001, batch=64
  3. lr=0.01,  batch=32
  4. lr=0.01,  batch=64

Each combination is trained and evaluated.
Best combination is selected based on validation performance.
```

### Visual Representation

```
                    batch_size
                   32        64
                ┌─────────┬─────────┐
learning   0.001│  ●      │    ●    │
rate            ├─────────┼─────────┤
           0.01 │  ●      │    ●    │
                └─────────┴─────────┘

● = One complete training run
Grid search evaluates ALL points
```

### The Algorithm

```
1. Define parameter grid
   ↓
2. Generate all combinations
   ↓
3. For each combination:
   a. Train model with these hyperparameters
   b. Evaluate on validation set (or cross-validation)
   c. Record performance
   ↓
4. Select combination with best performance
   ↓
5. Retrain on full training data with best hyperparameters
```

---

## Implementing Grid Search

### Manual Implementation

```python
import itertools
from typing import Dict, List, Any

def grid_search(param_grid: Dict[str, List], 
                train_fn, 
                eval_fn,
                X_train, y_train, 
                X_val, y_val):
    """
    Simple grid search implementation.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        train_fn: Function that trains model given hyperparameters
        eval_fn: Function that evaluates model
    """
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    results = []
    best_score = float('-inf')
    best_params = None
    
    print(f"Total combinations to evaluate: {len(combinations)}")
    
    for i, combo in enumerate(combinations):
        # Create parameter dictionary
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Train model
        model = train_fn(X_train, y_train, **params)
        
        # Evaluate
        score = eval_fn(model, X_val, y_val)
        print(f"  Score: {score:.4f}")
        
        results.append({'params': params, 'score': score})
        
        # Track best
        if score > best_score:
            best_score = score
            best_params = params
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params, results

# Usage example
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [64, 128, 256],
    'dropout': [0.0, 0.3, 0.5]
}

best_params, all_results = grid_search(
    param_grid, 
    train_model, 
    evaluate_model,
    X_train, y_train, 
    X_val, y_val
)
```

### Scikit-learn GridSearchCV

For scikit-learn compatible models:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'activation': ['relu', 'tanh']
}

# Create base model
model = MLPClassifier(max_iter=500, early_stopping=True)

# Create grid search with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',      # Metric to optimize
    verbose=2,               # Print progress
    n_jobs=-1,               # Use all CPU cores
    return_train_score=True  # Also record training scores
)

# Run the search
grid_search.fit(X_train, y_train)

# Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Access the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Analyzing Grid Search Results

```python
import pandas as pd

# Convert results to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# View top configurations
top_results = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'mean_train_score']
]
print(top_results)

# Check for overfitting (large gap between train and test)
results_df['overfit_gap'] = (
    results_df['mean_train_score'] - results_df['mean_test_score']
)

# Visualize results
import matplotlib.pyplot as plt

# Plot scores for different learning rates
for lr in [0.001, 0.01]:
    mask = results_df['param_learning_rate_init'] == lr
    subset = results_df[mask]
    plt.plot(subset['param_alpha'], subset['mean_test_score'], 
             label=f'lr={lr}', marker='o')

plt.xlabel('Alpha (L2 regularization)')
plt.ylabel('Mean CV Score')
plt.legend()
plt.title('Grid Search Results')
plt.show()
```

---

## Grid Search with PyTorch

### PyTorch Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import itertools

def create_model(hidden_size, dropout):
    """Create model with given hyperparameters."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, num_classes)
    )

def train_and_evaluate(params, X, y, n_folds=5):
    """Train with cross-validation and return mean score."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_fold),
            torch.LongTensor(y_train_fold)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # Create model
        model = create_model(params['hidden_size'], params['dropout'])
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train()
        for epoch in range(params['epochs']):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val_fold))
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted.numpy() == y_val_fold).mean()
            fold_scores.append(accuracy)
    
    return np.mean(fold_scores)

# Define grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'hidden_size': [64, 128],
    'dropout': [0.0, 0.3],
    'batch_size': [32, 64],
    'epochs': [50]  # Fixed
}

# Generate combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = list(itertools.product(*values))

# Run grid search
results = []
for combo in combinations:
    params = dict(zip(keys, combo))
    print(f"Testing: {params}")
    
    score = train_and_evaluate(params, X, y)
    results.append({'params': params, 'score': score})
    print(f"  CV Score: {score:.4f}")

# Find best
best_result = max(results, key=lambda x: x['score'])
print(f"\nBest: {best_result}")
```

---

## Computational Complexity

### Understanding the Cost

```
Grid Search Complexity:

Number of evaluations = ∏(number of values per parameter)

Example:
  learning_rate: 4 values
  batch_size: 4 values
  hidden_layers: 3 values
  neurons: 4 values
  dropout: 4 values

Total = 4 × 4 × 3 × 4 × 4 = 768 evaluations

If each evaluation takes 10 minutes:
  Total time = 768 × 10 = 7,680 minutes ≈ 5.3 days!
```

### Scaling Problem

```
Parameters    Values Each    Total Combinations
    2              5                25
    3              5               125
    4              5               625
    5              5             3,125
    6              5            15,625
    7              5            78,125

Exponential growth makes grid search impractical
for many hyperparameters!
```

### Strategies to Reduce Cost

1. **Coarse-to-Fine Search**
   ```python
   # First: coarse grid
   coarse_grid = {
       'lr': [0.0001, 0.001, 0.01, 0.1],
       'hidden': [32, 128, 512]
   }
   # Find best region: lr ≈ 0.001, hidden ≈ 128
   
   # Then: fine grid around best
   fine_grid = {
       'lr': [0.0005, 0.001, 0.002, 0.005],
       'hidden': [96, 128, 160, 192]
   }
   ```

2. **Reduce Cross-Validation Folds**
   ```python
   # Use 3-fold instead of 5-fold for initial search
   grid_search = GridSearchCV(model, param_grid, cv=3)
   ```

3. **Early Stopping**
   ```python
   # Stop training early if not improving
   model = MLPClassifier(early_stopping=True, n_iter_no_change=10)
   ```

4. **Parallel Execution**
   ```python
   # Use all CPU cores
   grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
   ```

---

## Advantages and Disadvantages

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Exhaustive** | Guaranteed to find best in the grid |
| **Simple** | Easy to understand and implement |
| **Reproducible** | Same grid always gives same results |
| **Parallelizable** | Each evaluation is independent |
| **Complete coverage** | No region of grid is missed |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Expensive** | Evaluates many poor configurations |
| **Scales poorly** | Exponential growth with parameters |
| **Discrete** | May miss optimal values between grid points |
| **Wasteful** | Spends equal time on important and unimportant parameters |
| **Requires domain knowledge** | Need to define good grid ranges |

---

## When to Use Grid Search

### Good Use Cases

- **Small search spaces** (< 100 combinations)
- **Few hyperparameters** (2-3 parameters)
- **Fine-tuning** around a known good region
- **When reproducibility is critical**
- **When you have compute resources to spare**

### Poor Use Cases

- **Large search spaces** (> 1000 combinations)
- **Many hyperparameters** (> 4 parameters)
- **Continuous parameters** with wide ranges
- **Limited compute budget**
- **Initial exploration** of unknown space

### Decision Guide

```
Should you use Grid Search?

Number of combinations < 100?
  ├── Yes → Grid Search is fine
  └── No → Consider Random Search or Bayesian

Do you need guaranteed best in grid?
  ├── Yes → Grid Search
  └── No → Random Search is more efficient

Is this fine-tuning around known good values?
  ├── Yes → Grid Search with small grid
  └── No → Start with Random Search
```

---

## Key Takeaways

1. **Grid search evaluates all combinations** in a predefined grid
2. **Guaranteed to find the best** configuration within the grid
3. **Computationally expensive** — scales exponentially with parameters
4. **Best for small search spaces** or fine-tuning
5. **Use coarse-to-fine** strategy to reduce cost
6. **Parallelize** when possible to speed up search
7. **Consider alternatives** (Random Search, Bayesian) for large spaces

---

*Grid search is the "brute force" approach to hyperparameter tuning. It's reliable but expensive. For larger search spaces, more efficient methods like Random Search or Bayesian Optimization are often better choices.*
