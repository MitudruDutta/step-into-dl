# 📋 Practical Guidelines

This guide provides actionable advice for hyperparameter tuning in real-world projects. Learn the optimal tuning order, good starting points, and common mistakes to avoid.

---

## Tuning Order

Not all hyperparameters are equally important. Tune them in order of impact.

### Recommended Order

```
Priority 1: Learning Rate
  ↓ Most important, tune first
Priority 2: Architecture (layers, neurons)
  ↓ Determines model capacity
Priority 3: Batch Size
  ↓ Affects training dynamics
Priority 4: Regularization (dropout, weight decay)
  ↓ Add only if overfitting
Priority 5: Other (activation, optimizer, etc.)
  ↓ Usually defaults work well
```

### Detailed Breakdown

| Priority | Hyperparameter | Why | When to Tune |
|----------|----------------|-----|--------------|
| 1 | Learning rate | Controls convergence speed and stability | Always |
| 2 | Number of layers | Determines model depth/capacity | If underfitting |
| 3 | Neurons per layer | Determines model width | If underfitting |
| 4 | Batch size | Affects gradient noise and speed | If training slow/unstable |
| 5 | Dropout rate | Prevents overfitting | If overfitting |
| 6 | Weight decay (L2) | Prevents overfitting | If overfitting |
| 7 | Optimizer | Different optimizers suit different problems | Rarely needed |
| 8 | Activation function | ReLU usually works | Rarely needed |

---

## Starting Points

Good defaults to begin your experiments.

### Training Hyperparameters

```python
# Recommended starting values
defaults = {
    'learning_rate': 0.001,      # For Adam
    'learning_rate_sgd': 0.01,   # For SGD (needs higher)
    'batch_size': 32,            # Good balance
    'epochs': 100,               # With early stopping
    'optimizer': 'Adam',         # Usually best default
}
```

### Architecture Hyperparameters

```python
# Start simple, increase if underfitting
architecture_defaults = {
    'hidden_layers': 2,          # Start with 2
    'neurons': 128,              # Start moderate
    'activation': 'relu',        # Almost always works
}
```

### Regularization Hyperparameters

```python
# Add only if overfitting
regularization_defaults = {
    'dropout': 0.0,              # Start without
    'weight_decay': 0.0,         # Start without
    'early_stopping_patience': 10,
}
```


### Search Ranges

| Hyperparameter | Starting Value | Search Range | Scale |
|----------------|----------------|--------------|-------|
| Learning rate | 0.001 | 1e-5 to 0.1 | Log |
| Batch size | 32 | 16 to 256 | Linear |
| Hidden layers | 2 | 1 to 5 | Linear |
| Neurons | 128 | 32 to 512 | Linear |
| Dropout | 0.0 | 0.0 to 0.5 | Linear |
| Weight decay | 0.0 | 1e-6 to 1e-2 | Log |
| Momentum (SGD) | 0.9 | 0.8 to 0.99 | Linear |

---

## Common Mistakes

### Mistake 1: Tuning on Test Data

```
WRONG:
  Train → Tune on test set → Report test accuracy
  
  Problem: Test accuracy is optimistically biased
  
CORRECT:
  Train → Tune on validation set → Final evaluation on test set
  
  Or use cross-validation for tuning
```

### Mistake 2: Too Few Trials

```
WRONG:
  "I tried 5 random configurations, this one is best"
  
  Problem: 5 trials is not enough to explore the space
  
CORRECT:
  Random search: 50-100 trials minimum
  Bayesian (Optuna): 50+ trials with pruning
  Grid search: Only for small spaces
```

### Mistake 3: Ignoring Interactions

```
WRONG:
  Tune learning rate → Fix it → Tune batch size separately
  
  Problem: LR and batch size interact!
  
CORRECT:
  Tune learning rate and batch size together
  Or: When changing batch size, adjust LR proportionally
```

### Mistake 4: Not Using Early Stopping

```
WRONG:
  Train every configuration for 100 epochs
  
  Problem: Wastes compute on bad configurations
  
CORRECT:
  Use early stopping or pruning (Optuna)
  Stop bad trials early
```

### Mistake 5: Overfitting to Validation Set

```
WRONG:
  Run 1000 trials, pick best validation score
  
  Problem: With enough trials, you'll find one that's
           lucky on validation but doesn't generalize
  
CORRECT:
  Use cross-validation for more robust estimates
  Limit number of trials
  Use nested CV for unbiased evaluation
```

### Mistake 6: Using Uniform Scale for Learning Rate

```
WRONG:
  lr = np.random.uniform(0.0001, 0.1)
  # Most samples near 0.1
  
CORRECT:
  lr = 10 ** np.random.uniform(-4, -1)
  # Or: loguniform(0.0001, 0.1)
  # Equal probability across orders of magnitude
```

---

## Workflow Template

### Step-by-Step Process

```
Step 1: Establish Baseline
  ├── Use default hyperparameters
  ├── Train model
  ├── Record: train_loss, val_loss, val_accuracy
  └── This is your reference point

Step 2: Quick Learning Rate Search
  ├── Try: [0.1, 0.01, 0.001, 0.0001]
  ├── Pick best based on validation
  └── Or use LR finder for more precision

Step 3: Architecture Search (if underfitting)
  ├── Try different depths: [1, 2, 3, 4] layers
  ├── Try different widths: [64, 128, 256] neurons
  └── Pick simplest that doesn't underfit

Step 4: Batch Size (if needed)
  ├── Try: [16, 32, 64, 128]
  ├── Adjust LR if changing batch size significantly
  └── Consider memory constraints

Step 5: Regularization (if overfitting)
  ├── Add dropout: try [0.1, 0.2, 0.3, 0.5]
  ├── Add weight decay: try [1e-5, 1e-4, 1e-3]
  └── Use early stopping

Step 6: Fine-tuning
  ├── Narrow search around best configuration
  ├── Use Optuna for efficient fine-tuning
  └── Run longer with best hyperparameters

Step 7: Final Evaluation
  ├── Train on full training set with best hyperparameters
  ├── Evaluate on held-out test set
  └── Report final performance
```

### Code Template

```python
import optuna
import torch
import torch.nn as nn

# Step 1: Define objective
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Build model
    model = build_model(n_layers, hidden_size, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training with pruning
    for epoch in range(50):
        train_loss = train_epoch(model, optimizer, train_loader)
        val_acc = evaluate(model, val_loader)
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_acc

# Step 2: Run optimization
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100)

# Step 3: Get best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy: {study.best_value:.4f}")

# Step 4: Train final model
best_model = build_model(**study.best_params)
train_full(best_model, train_loader, epochs=100)

# Step 5: Final evaluation on test set
test_acc = evaluate(best_model, test_loader)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Debugging Tuning Issues

### Issue: All Trials Have Similar Performance

```
Possible causes:
  1. Search range too narrow → Expand ranges
  2. Hyperparameters don't matter much → Focus elsewhere
  3. Model is bottlenecked by data → Get more data
  
Diagnosis:
  - Check parameter importance plot in Optuna
  - Try wider ranges
```

### Issue: Best Trial Doesn't Reproduce

```
Possible causes:
  1. Random seed not fixed → Set all seeds
  2. Non-deterministic operations → Use deterministic mode
  3. Overfitting to validation → Use cross-validation
  
Fix:
  torch.manual_seed(42)
  np.random.seed(42)
  torch.backends.cudnn.deterministic = True
```

### Issue: Tuning Takes Too Long

```
Solutions:
  1. Use pruning (Optuna) → Stop bad trials early
  2. Reduce epochs per trial → Use early stopping
  3. Use smaller dataset for tuning → Subsample
  4. Reduce search space → Focus on important params
  5. Use fewer CV folds → 3 instead of 5
```

### Issue: Validation Score Much Higher Than Test

```
Possible causes:
  1. Overfitting to validation set → Use cross-validation
  2. Data leakage → Check preprocessing pipeline
  3. Too many trials → Limit trials, use nested CV
  
Prevention:
  - Use nested cross-validation
  - Keep test set completely separate
  - Limit number of tuning iterations
```

---

## Method Selection Guide

### Decision Tree

```
How many hyperparameters?
├── 2-3 parameters
│   └── Grid search (if < 100 combinations)
│       or Random search
│
├── 4-6 parameters
│   └── Random search (50-100 trials)
│       or Optuna (better if expensive)
│
└── 7+ parameters
    └── Optuna with pruning
        (focus on important params first)

How expensive is each evaluation?
├── < 1 minute
│   └── Random search is fine
│
├── 1-30 minutes
│   └── Optuna with pruning
│
└── > 30 minutes
    └── Optuna with aggressive pruning
        Reduce trials, use early stopping
```

### Quick Reference

| Scenario | Recommended Method | Trials |
|----------|-------------------|--------|
| Quick experiment | Manual tuning | 5-10 |
| Small search space | Grid search | All |
| First exploration | Random search | 50-100 |
| Production tuning | Optuna | 100+ |
| Very expensive | Optuna + pruning | 30-50 |

---

## Key Takeaways

1. **Tune in order of importance**: LR → Architecture → Batch size → Regularization
2. **Start with good defaults**: lr=0.001, batch=32, 2 layers
3. **Use log scale** for learning rate and weight decay
4. **Never tune on test data** — use validation or CV
5. **Use early stopping/pruning** to save computation
6. **Keep records** of all experiments
7. **Don't overtune** — too many trials can overfit to validation
8. **Verify reproducibility** before reporting final results

---

*Good hyperparameter tuning is systematic, not random. Follow a structured approach, start with important hyperparameters, and always validate your results properly.*
