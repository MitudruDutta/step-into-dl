# ⚖️ L1 and L2 Regularization

L1 and L2 regularization are fundamental techniques that add penalty terms to the loss function based on the magnitude of model weights. They're simple, effective, and widely used across all types of machine learning models.

---

## The Core Idea

### Why Penalize Weights?

Large weights in a neural network often indicate:
- The model is relying too heavily on specific features
- Complex, potentially fragile decision boundaries
- Higher risk of overfitting

By adding a penalty for large weights, we encourage the model to:
- Use smaller, more distributed weights
- Learn simpler, more generalizable patterns
- Avoid memorizing training data

### The Modified Loss Function

```
Total Loss = Original Loss + λ × Weight Penalty

Where:
- Original Loss: Your task loss (CrossEntropy, MSE, etc.)
- λ (lambda): Regularization strength (hyperparameter)
- Weight Penalty: Sum of weight magnitudes (L1 or L2)
```

---

## L2 Regularization (Ridge / Weight Decay)

### The Formula

L2 regularization adds the **sum of squared weights** to the loss:

```
L2 Loss = Original Loss + λ × Σ(w²)

Or equivalently:
L2 Loss = Original Loss + λ × ||w||²
```

### How L2 Works

The gradient of the L2 penalty is proportional to the weight itself:

```
∂(λw²)/∂w = 2λw

This means:
- Large weights get large penalties
- Small weights get small penalties
- Weights are pushed toward zero, but rarely reach exactly zero
```

### Effect on Weights

```
Before L2:  weights = [2.5, -3.1, 0.8, 4.2, -1.9, 0.01]
After L2:   weights = [1.2, -1.5, 0.4, 2.0, -0.9, 0.005]
                       ↓     ↓    ↓    ↓    ↓     ↓
                    Smaller, but all non-zero
```

### Geometric Interpretation

L2 regularization constrains weights to lie within a sphere:

```
        w₂
        │
        │    ╭───╮
        │   ╱     ╲
        │  │   ●   │  ← Optimal point constrained
        │   ╲     ╱     to be within the sphere
        │    ╰───╯
        └──────────── w₁
        
The sphere radius is determined by λ.
Smaller λ = larger sphere = less constraint.
```

### Why "Weight Decay"?

In gradient descent, L2 regularization causes weights to decay toward zero each step:

```
Standard update:    w = w - lr × gradient
With L2:           w = w - lr × gradient - lr × λ × w
                     = w × (1 - lr × λ) - lr × gradient
                           ↑
                     Weight decay factor
```

This is why L2 regularization is called "weight decay" in optimizers.

---

## L1 Regularization (Lasso)

### The Formula

L1 regularization adds the **sum of absolute weights** to the loss:

```
L1 Loss = Original Loss + λ × Σ|w|

Or equivalently:
L1 Loss = Original Loss + λ × ||w||₁
```

### How L1 Works

The gradient of the L1 penalty is constant (the sign of the weight):

```
∂(λ|w|)/∂w = λ × sign(w)

This means:
- All weights get the same magnitude penalty
- Small weights can be pushed to exactly zero
- Creates sparse models
```

### Effect on Weights

```
Before L1:  weights = [2.5, -0.1, 0.8, 0.05, -1.9, 0.01]
After L1:   weights = [1.8,  0.0, 0.3, 0.0,  -1.2, 0.0]
                       ↓     ↓    ↓    ↓     ↓    ↓
                    Some weights become exactly zero
```

### Geometric Interpretation

L1 regularization constrains weights to lie within a diamond:

```
        w₂
        │
        │      ╱╲
        │     ╱  ╲
        │    ╱ ●  ╲  ← Optimal point often lands
        │   ╱      ╲   on a corner (sparse solution)
        │  ╱        ╲
        └──────────── w₁
        
The diamond's corners encourage sparsity.
```

### Feature Selection

Because L1 pushes weights to exactly zero, it effectively performs feature selection:

```
Original features: [age, income, height, weight, zip_code, ...]
After L1:         [age, income, 0,      weight, 0,        ...]
                                ↑               ↑
                          Features removed (weight = 0)
```

---

## L1 vs L2 Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Penalty** | Sum of absolute values: Σ\|w\| | Sum of squares: Σw² |
| **Gradient** | Constant: sign(w) | Proportional: 2w |
| **Sparsity** | Yes (many zero weights) | No (weights stay non-zero) |
| **Feature selection** | Yes | No |
| **Correlated features** | Picks one arbitrarily | Distributes weight among all |
| **Computational** | Less stable (non-differentiable at 0) | Smooth, stable gradients |
| **Common use** | Feature selection, interpretability | General regularization |

### When to Use Each

**Use L2 (Weight Decay) when:**
- You want general regularization
- All features might be relevant
- You want smooth, stable training
- Default choice for deep learning

**Use L1 when:**
- You want feature selection
- You need a sparse model
- Interpretability is important
- Many features are likely irrelevant

---

## Elastic Net (L1 + L2)

Elastic Net combines both penalties:

```
Elastic Net Loss = Original Loss + λ₁ × Σ|w| + λ₂ × Σw²
```

### Benefits

- Gets sparsity from L1
- Gets stability from L2
- Handles correlated features better than L1 alone

### When to Use

- When you want some feature selection but also stability
- When features are correlated
- When L1 alone is too aggressive

---

## PyTorch Implementation

### L2 Regularization (Weight Decay)

**Method 1: Built into Optimizer (Recommended)**

```python
import torch.optim as optim

# Adam with weight decay
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 regularization strength
)

# SGD with weight decay
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# AdamW (correct weight decay for Adam)
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # AdamW uses higher values
)
```

**Method 2: Manual Implementation**

```python
def train_step(model, optimizer, criterion, inputs, targets, l2_lambda=1e-4):
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Add L2 penalty manually
    l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
    total_loss = loss + l2_lambda * l2_penalty
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

### L1 Regularization

L1 is not built into PyTorch optimizers, so you must add it manually:

```python
def train_step_l1(model, optimizer, criterion, inputs, targets, l1_lambda=1e-5):
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Add L1 penalty
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    total_loss = loss + l1_lambda * l1_penalty
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

### Elastic Net (L1 + L2)

```python
def train_step_elastic(model, optimizer, criterion, inputs, targets, 
                       l1_lambda=1e-5, l2_lambda=1e-4):
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Add both penalties
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
    
    total_loss = loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

### Selective Regularization

Sometimes you want to regularize only certain layers:

```python
def selective_l2_penalty(model, l2_lambda=1e-4):
    """Apply L2 only to weight matrices, not biases."""
    penalty = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only weights, not biases
            penalty += param.pow(2).sum()
    return l2_lambda * penalty

# Or exclude certain layers
def selective_l2_exclude_bn(model, l2_lambda=1e-4):
    """Apply L2 but exclude BatchNorm parameters."""
    penalty = 0
    for name, param in model.named_parameters():
        if 'bn' not in name and 'weight' in name:
            penalty += param.pow(2).sum()
    return l2_lambda * penalty
```

---

## Choosing λ (Regularization Strength)

### Typical Values

| Regularization | Typical λ Range | Starting Point |
|----------------|-----------------|----------------|
| **L2 (weight_decay)** | 1e-5 to 1e-2 | 1e-4 |
| **L1** | 1e-6 to 1e-3 | 1e-5 |
| **AdamW weight_decay** | 1e-3 to 1e-1 | 0.01 |

### Effect of λ

```
λ too small:
  - Little regularization effect
  - Model may still overfit
  - Weights remain large

λ just right:
  - Good balance between fit and generalization
  - Weights are reasonably sized
  - Best validation performance

λ too large:
  - Underfitting
  - Weights pushed too close to zero
  - Model can't learn the pattern
```

### Tuning Strategy

```python
# Grid search for optimal weight_decay
weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]

results = []
for wd in weight_decays:
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
    
    train(model, optimizer)
    val_loss = evaluate(model)
    
    results.append((wd, val_loss))
    print(f"weight_decay={wd}: val_loss={val_loss:.4f}")

# Choose the weight_decay with lowest validation loss
```

---

## Adam vs AdamW

### The Problem with Adam + L2

Standard Adam applies weight decay incorrectly:

```
Adam update: w = w - lr × m / (√v + ε)

With weight_decay in Adam:
  gradient = gradient + weight_decay × w  ← Added to gradient
  w = w - lr × m / (√v + ε)
  
The weight decay is scaled by the adaptive learning rate,
which is NOT the same as true L2 regularization.
```

### AdamW: The Fix

AdamW decouples weight decay from the gradient:

```
AdamW update:
  w = w - lr × m / (√v + ε) - lr × weight_decay × w
                              ↑
                    True weight decay, not scaled
```

### When to Use AdamW

```python
# Use AdamW when you want proper L2 regularization with Adam
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # Higher values work better with AdamW
)

# Use standard Adam weight_decay for legacy compatibility
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # Lower values needed
)
```

**Recommendation:** Use AdamW for new projects.

---

## Visualizing Regularization Effects

### Weight Distribution

```python
import matplotlib.pyplot as plt

def plot_weight_distribution(model, title):
    weights = []
    for param in model.parameters():
        weights.extend(param.detach().cpu().numpy().flatten())
    
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.show()

# Compare before and after training
plot_weight_distribution(model_no_reg, "No Regularization")
plot_weight_distribution(model_l2, "With L2 Regularization")
plot_weight_distribution(model_l1, "With L1 Regularization")
```

Expected results:
- **No regularization:** Wide distribution, some very large weights
- **L2:** Narrower distribution, centered around zero
- **L1:** Many weights at exactly zero, others spread out

---

## Common Mistakes

### Mistake 1: Regularizing Biases

```python
# WRONG: Regularizing all parameters including biases
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

# CORRECT: Only regularize weights
l2_penalty = sum(p.pow(2).sum() for name, p in model.named_parameters() 
                 if 'weight' in name)
```

Biases don't contribute to overfitting the same way weights do.

### Mistake 2: Wrong λ Scale

```python
# WRONG: Using same λ for L1 and L2
l1_lambda = 1e-4  # Too strong for L1
l2_lambda = 1e-4  # Okay for L2

# CORRECT: L1 typically needs smaller λ
l1_lambda = 1e-5
l2_lambda = 1e-4
```

### Mistake 3: Not Monitoring Weight Magnitudes

```python
# Good practice: Monitor weight statistics
def log_weight_stats(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.abs().mean():.4f}, max={param.abs().max():.4f}")
```

---

## Key Takeaways

1. **L2 regularization** shrinks weights toward zero but keeps them non-zero
2. **L1 regularization** produces sparse models with some zero weights
3. **Use L2 (weight_decay)** as the default for deep learning
4. **Use L1** when you need feature selection or sparse models
5. **AdamW** implements weight decay correctly for Adam
6. **Start with λ=1e-4** for L2 and tune based on validation performance
7. **Don't regularize biases** — only weights

---

*L1 and L2 regularization are foundational techniques. Understanding them helps you understand more complex regularization methods and when to apply them.*
