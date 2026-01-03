# 🎲 Random Search

Random search samples random combinations from the hyperparameter search space. Despite its simplicity, it's often more efficient than grid search, especially for high-dimensional spaces.

---

## Why Random Search Works

### The Key Insight

Not all hyperparameters are equally important. Random search explores more unique values of important parameters than grid search with the same budget.

```
Grid Search (9 points):          Random Search (9 points):
                                 
    ●───●───●                        ●       ●
    │   │   │                            ●
    ●───●───●                      ●         ●
    │   │   │                          ●
    ●───●───●                        ●     ●
                                           ●
Only 3 unique values              9 unique values
per dimension                     per dimension
```

### When One Parameter Dominates

```
Scenario: Learning rate matters a lot, batch size doesn't

Grid Search:
  lr: [0.001, 0.01, 0.1]  → Only 3 LR values tested
  batch: [16, 32, 64]     → Wastes evaluations

Random Search:
  lr: uniform(0.001, 0.1) → Many LR values tested
  batch: choice([16,32,64]) → Doesn't waste much

Random search explores the important dimension more thoroughly!
```

---

## How Random Search Works

### The Algorithm

```
1. Define parameter distributions (not fixed values)
   ↓
2. For n_iterations:
   a. Sample random values from each distribution
   b. Train model with sampled hyperparameters
   c. Evaluate on validation set
   d. Record performance
   ↓
3. Select configuration with best performance
```

### Sampling Strategies

```python
# Different ways to sample hyperparameters

# Uniform sampling (continuous)
learning_rate = np.random.uniform(0.0001, 0.1)

# Log-uniform sampling (for parameters spanning orders of magnitude)
learning_rate = 10 ** np.random.uniform(-4, -1)  # 0.0001 to 0.1

# Integer sampling
hidden_size = np.random.randint(32, 257)  # 32 to 256

# Categorical sampling
optimizer = np.random.choice(['adam', 'sgd', 'rmsprop'])

# Conditional sampling
if optimizer == 'sgd':
    momentum = np.random.uniform(0.0, 0.99)
```

---

## Implementing Random Search

### Manual Implementation

```python
import numpy as np
from typing import Dict, Callable, Any

def random_search(param_distributions: Dict,
                  train_fn: Callable,
                  eval_fn: Callable,
                  X_train, y_train,
                  X_val, y_val,
                  n_iterations: int = 50,
                  random_state: int = 42):
    """
    Random search for hyperparameter optimization.
    """
    np.random.seed(random_state)
    
    results = []
    best_score = float('-inf')
    best_params = None
    
    for i in range(n_iterations):
        # Sample hyperparameters
        params = {}
        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                # Categorical
                params[name] = np.random.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                # Continuous uniform
                params[name] = np.random.uniform(dist[0], dist[1])
            elif callable(dist):
                # Custom distribution
                params[name] = dist()
        
        print(f"[{i+1}/{n_iterations}] Testing: {params}")
        
        # Train and evaluate
        model = train_fn(X_train, y_train, **params)
        score = eval_fn(model, X_val, y_val)
        
        print(f"  Score: {score:.4f}")
        
        results.append({'params': params, 'score': score})
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params, results


# Usage with custom distributions
param_distributions = {
    'learning_rate': lambda: 10 ** np.random.uniform(-4, -1),  # Log-uniform
    'hidden_size': lambda: np.random.randint(32, 257),
    'dropout': (0.0, 0.5),  # Uniform
    'optimizer': ['adam', 'sgd', 'rmsprop'],  # Categorical
    'batch_size': [16, 32, 64, 128]  # Categorical
}

best_params, results = random_search(
    param_distributions,
    train_model,
    evaluate_model,
    X_train, y_train,
    X_val, y_val,
    n_iterations=100
)
```

### Scikit-learn RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint, loguniform

# Define parameter distributions
param_distributions = {
    'hidden_layer_sizes': [(64,), (128,), (256,), (64, 32), (128, 64), (256, 128)],
    'learning_rate_init': loguniform(1e-4, 1e-1),  # Log-uniform distribution
    'alpha': loguniform(1e-5, 1e-2),               # L2 regularization
    'batch_size': randint(16, 129),                # Integer uniform
    'activation': ['relu', 'tanh', 'logistic']
}

# Create model
model = MLPClassifier(max_iter=500, early_stopping=True)

# Create random search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,              # Number of random samples
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,               # Parallel execution
    random_state=42,         # Reproducibility
    return_train_score=True
)

# Run search
random_search.fit(X_train, y_train)

# Results
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Test set evaluation
test_score = random_search.best_estimator_.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

---

## Log-Uniform Sampling

### Why Log-Uniform Matters

For parameters that span multiple orders of magnitude (like learning rate), uniform sampling is inefficient.

```
Learning rate range: 0.0001 to 0.1

Uniform sampling:
  Most samples fall in [0.05, 0.1] range
  Few samples in [0.0001, 0.001] range
  
  0.0001 ─────────────────────────────── 0.1
         [sparse]              [dense]

Log-uniform sampling:
  Equal probability for each order of magnitude
  
  0.0001 ── 0.001 ── 0.01 ── 0.1
  [equal]  [equal]  [equal]
```

### Implementation

```python
import numpy as np
from scipy.stats import loguniform

# Method 1: Using scipy
lr_samples = loguniform(1e-4, 1e-1).rvs(1000)

# Method 2: Manual implementation
def log_uniform(low, high, size=None):
    """Sample from log-uniform distribution."""
    log_low = np.log10(low)
    log_high = np.log10(high)
    return 10 ** np.random.uniform(log_low, log_high, size)

lr_samples = log_uniform(0.0001, 0.1, size=1000)

# Verify distribution
import matplotlib.pyplot as plt
plt.hist(lr_samples, bins=50)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.title('Log-Uniform Distribution')
plt.show()
```

---

## Random Search with PyTorch

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold

def sample_hyperparameters():
    """Sample a random hyperparameter configuration."""
    return {
        'learning_rate': 10 ** np.random.uniform(-4, -1),
        'hidden_size': np.random.choice([64, 128, 256, 512]),
        'num_layers': np.random.randint(1, 5),
        'dropout': np.random.uniform(0.0, 0.5),
        'batch_size': np.random.choice([16, 32, 64, 128]),
        'weight_decay': 10 ** np.random.uniform(-6, -2),
        'optimizer': np.random.choice(['adam', 'sgd', 'adamw'])
    }

def build_model(params, input_size, output_size):
    """Build model from hyperparameters."""
    layers = []
    current_size = input_size
    
    for i in range(params['num_layers']):
        layers.append(nn.Linear(current_size, params['hidden_size']))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(params['dropout']))
        current_size = params['hidden_size']
    
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)

def train_and_evaluate(params, X, y, n_folds=3):
    """Train with CV and return mean score."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        model = build_model(params, X.shape[1], len(np.unique(y)))
        
        # Setup optimizer
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        
        # Training loop (simplified)
        # ... train for some epochs ...
        
        # Evaluate
        accuracy = evaluate(model, X[val_idx], y[val_idx])
        scores.append(accuracy)
    
    return np.mean(scores)

# Run random search
n_iterations = 50
results = []

for i in range(n_iterations):
    params = sample_hyperparameters()
    print(f"[{i+1}/{n_iterations}] {params}")
    
    score = train_and_evaluate(params, X, y)
    results.append({'params': params, 'score': score})
    print(f"  Score: {score:.4f}")

# Find best
best = max(results, key=lambda x: x['score'])
print(f"\nBest configuration: {best}")
```

---

## Comparing Grid vs Random Search

### Efficiency Comparison

```
Same budget: 25 evaluations

Grid Search (5×5 grid):
  lr: [0.0001, 0.001, 0.01, 0.1, 1.0]
  dropout: [0.0, 0.1, 0.2, 0.3, 0.4]
  
  Only 5 unique values per parameter

Random Search (25 samples):
  lr: continuous from 0.0001 to 1.0
  dropout: continuous from 0.0 to 0.5
  
  Up to 25 unique values per parameter!
```

### When Random Beats Grid

| Scenario | Winner |
|----------|--------|
| Few important parameters | Random |
| All parameters equally important | Tie |
| Large search space | Random |
| Small search space | Grid |
| Continuous parameters | Random |
| Need exact reproducibility | Grid |

---

## Advantages and Disadvantages

### Advantages

| Advantage | Description |
|-----------|-------------|
| **More efficient** | Explores more unique values per parameter |
| **Anytime algorithm** | Can stop early and still have good results |
| **Handles continuous** | Natural fit for continuous parameters |
| **Scales better** | Doesn't suffer from curse of dimensionality |
| **Easy to parallelize** | Each sample is independent |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **No guarantee** | May miss optimal configuration |
| **Randomness** | Results vary with random seed |
| **No learning** | Doesn't use past results to guide search |
| **May oversample** | Could sample similar regions multiple times |

---

## Best Practices

### 1. Use Appropriate Distributions

```python
# Good: Log-uniform for learning rate
'learning_rate': loguniform(1e-5, 1e-1)

# Bad: Uniform for learning rate
'learning_rate': uniform(0.00001, 0.1)  # Most samples near 0.1
```

### 2. Set Sufficient Iterations

```
Rule of thumb:
  - 2-3 parameters: 20-50 iterations
  - 4-5 parameters: 50-100 iterations
  - 6+ parameters: 100+ iterations
  
More iterations = higher chance of finding good configuration
```

### 3. Use Cross-Validation

```python
# Always use CV to get robust estimates
random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=100,
    cv=5  # 5-fold CV
)
```

### 4. Analyze Results

```python
# Look at top configurations, not just the best
results_df = pd.DataFrame(random_search.cv_results_)
top_10 = results_df.nsmallest(10, 'rank_test_score')

# Check if top configurations are similar
# If yes → you've found a good region
# If no → might need more iterations
```

---

## Key Takeaways

1. **Random search samples randomly** from parameter distributions
2. **More efficient than grid search** for most problems
3. **Use log-uniform** for parameters spanning orders of magnitude
4. **Explores more unique values** per parameter with same budget
5. **Anytime algorithm** — can stop early with reasonable results
6. **Doesn't learn from past trials** — consider Bayesian optimization for that
7. **Good default choice** when you don't know which parameters matter

---

*Random search is often the best starting point for hyperparameter tuning. It's simple, efficient, and works well in practice. For even better efficiency, consider Bayesian optimization methods like Optuna.*
