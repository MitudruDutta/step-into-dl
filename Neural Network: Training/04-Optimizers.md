# 🚀 Advanced Optimizers: Beyond Vanilla Gradient Descent

While basic gradient descent works, modern deep learning uses enhanced optimizers that converge faster and handle complex loss landscapes better.

---

## Why Advanced Optimizers?

Vanilla SGD has limitations:
- Same learning rate for all parameters
- Can oscillate in ravines
- Slow convergence on flat surfaces
- Gets stuck at saddle points

Advanced optimizers address these issues with:
- **Momentum**: Accumulate velocity to smooth updates
- **Adaptive Learning Rates**: Different LR per parameter
- **Second-Order Information**: Use curvature estimates

---

## SGD with Momentum

Adds a "velocity" term that accumulates past gradients, helping to:
- Smooth out oscillations
- Accelerate through flat regions
- Escape shallow local minima

### Algorithm
```
velocity = momentum × velocity + gradient
weights = weights - learning_rate × velocity
```

### Intuition
Like a ball rolling downhill—it builds up speed and can roll through small bumps.

### PyTorch
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # Typical value
)
```

### When to Use
- CNNs (often outperforms Adam)
- When you want more control
- Large-scale training

---

## Adam (Adaptive Moment Estimation)

The most popular optimizer. Combines momentum with adaptive learning rates.

### Key Features
- Maintains running averages of gradients (momentum)
- Maintains running averages of squared gradients (adaptive LR)
- Bias correction for initial steps

### Algorithm
```
m = β₁ × m + (1 - β₁) × gradient           # First moment (momentum)
v = β₂ × v + (1 - β₂) × gradient²          # Second moment (adaptive LR)
m_hat = m / (1 - β₁ᵗ)                       # Bias correction
v_hat = v / (1 - β₂ᵗ)                       # Bias correction
weights = weights - lr × m_hat / (√v_hat + ε)
```

### PyTorch
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,        # Default, often works well
    betas=(0.9, 0.999),  # Momentum coefficients
    eps=1e-8         # Numerical stability
)
```

### When to Use
- Default choice for most tasks
- Transformers and NLP
- When you want "it just works"

---

## AdamW (Adam with Weight Decay)

Adam with proper weight decay (L2 regularization) decoupled from gradient updates.

### Why AdamW?
Original Adam applies weight decay incorrectly. AdamW fixes this, leading to better generalization.

### PyTorch
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)
```

### When to Use
- Transformers (BERT, GPT, etc.)
- When using weight decay
- Modern best practice over Adam

---

## RMSprop

Adapts learning rate based on recent gradient magnitudes. Predecessor to Adam.

### Algorithm
```
v = decay × v + (1 - decay) × gradient²
weights = weights - lr × gradient / (√v + ε)
```

### PyTorch
```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99  # Decay rate
)
```

### When to Use
- RNNs and LSTMs
- Non-stationary problems
- When Adam doesn't work well

---

## AdaGrad

Adapts learning rate based on historical gradient accumulation. Good for sparse data.

### Behavior
- Parameters with large gradients get smaller LR
- Parameters with small gradients get larger LR
- LR monotonically decreases (can become too small)

### PyTorch
```python
optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=0.01
)
```

### When to Use
- Sparse features (NLP, recommendations)
- When different features have very different frequencies

---

## Comparison Table

| Optimizer | Adaptive LR | Momentum | Memory | Best For |
|-----------|-------------|----------|--------|----------|
| **SGD** | ❌ | ❌ | Low | Baseline |
| **SGD+Momentum** | ❌ | ✅ | Low | CNNs, large-scale |
| **AdaGrad** | ✅ | ❌ | Medium | Sparse data |
| **RMSprop** | ✅ | ❌ | Medium | RNNs |
| **Adam** | ✅ | ✅ | High | General purpose |
| **AdamW** | ✅ | ✅ | High | Transformers |

---

## Optimizer Selection Guide

```
Start Here
    │
    ▼
┌─────────────────────────────────────┐
│ What type of model?                 │
└─────────────────────────────────────┘
    │
    ├── Transformer/NLP ──────────► AdamW
    │
    ├── CNN (image) ──────────────► SGD+Momentum or Adam
    │
    ├── RNN/LSTM ─────────────────► Adam or RMSprop
    │
    ├── Sparse data ──────────────► AdaGrad or Adam
    │
    └── Not sure ─────────────────► Adam (safe default)
```

---

## Hyperparameter Defaults

| Optimizer | Learning Rate | Other |
|-----------|---------------|-------|
| **SGD** | 0.01 - 0.1 | momentum=0.9 |
| **Adam** | 0.001 | betas=(0.9, 0.999) |
| **AdamW** | 0.001 | weight_decay=0.01 |
| **RMSprop** | 0.01 | alpha=0.99 |

---

## Common Patterns

### Learning Rate Warmup
Gradually increase LR at the start of training:
```python
# Linear warmup for first 1000 steps
if step < 1000:
    lr = base_lr * (step / 1000)
```

### Learning Rate Scheduling with Adam
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()
```

### Gradient Clipping
Prevent exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Debugging Optimizer Issues

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Loss not decreasing | LR too low | Increase LR |
| Loss exploding | LR too high | Decrease LR |
| Loss oscillating | LR too high | Decrease LR, add momentum |
| Slow convergence | Wrong optimizer | Try Adam |
| Poor generalization | No weight decay | Use AdamW with decay |

---

*Adam/AdamW is the safe default for most tasks. SGD+Momentum often works better for CNNs with proper tuning. Always start simple and add complexity only when needed.*
