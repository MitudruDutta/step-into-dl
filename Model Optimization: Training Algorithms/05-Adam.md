# ⚡ Adam (Adaptive Moment Estimation)

Adam is the "gold standard" of modern deep learning optimizers. It combines the best of Momentum (direction tracking) and RMSProp (magnitude adaptation) into a single, powerful algorithm.

---

## Why Adam?

### The Best of Both Worlds

| Optimizer | What It Does | Limitation |
|-----------|--------------|------------|
| **Momentum** | Accelerates in consistent directions | Doesn't adapt per-parameter |
| **RMSProp** | Adapts learning rate per-parameter | No momentum/acceleration |
| **Adam** | Both acceleration AND adaptation | (Almost none!) |

Adam tracks two moving averages simultaneously:
1. **First moment (m)**: Mean of gradients → like Momentum
2. **Second moment (v)**: Mean of squared gradients → like RMSProp

---

## How Adam Works

### The Complete Algorithm

```
# Initialize moments
m = 0  # First moment (mean)
v = 0  # Second moment (variance)
t = 0  # Time step

# For each iteration:
t = t + 1

# Update biased moments
m = β₁ × m + (1 - β₁) × gradient        # Momentum-like
v = β₂ × v + (1 - β₂) × gradient²       # RMSProp-like

# Bias correction
m̂ = m / (1 - β₁^t)
v̂ = v / (1 - β₂^t)

# Update weights
weights = weights - lr × m̂ / (√v̂ + ε)
```

### Breaking It Down

| Term | Meaning | Default |
|------|---------|---------|
| `m` | Running mean of gradients | - |
| `v` | Running mean of squared gradients | - |
| `β₁` | Decay rate for first moment | 0.9 |
| `β₂` | Decay rate for second moment | 0.999 |
| `ε` | Numerical stability constant | 1e-8 |
| `lr` | Learning rate | 0.001 |

---

## The Two Moments Explained

### First Moment (m) — The Direction

```
m = β₁ × m + (1 - β₁) × gradient
```

This is essentially **Momentum**:
- Tracks the exponentially weighted average of gradients
- Provides acceleration in consistent directions
- Smooths out noisy gradient estimates

### Second Moment (v) — The Scale

```
v = β₂ × v + (1 - β₂) × gradient²
```

This is essentially **RMSProp**:
- Tracks the exponentially weighted average of squared gradients
- Estimates the variance/magnitude of gradients
- Used to normalize updates per-parameter

### Combined Update

```
update = m̂ / (√v̂ + ε)
```

- **Numerator (m̂)**: Direction and magnitude from momentum
- **Denominator (√v̂)**: Normalization based on gradient history
- **Result**: Adaptive, momentum-enhanced updates

---

## Bias Correction: Why It Matters

### The Problem

At the start of training, `m` and `v` are initialized to 0. This causes them to be biased toward zero in early iterations:

```
Iteration 1 (β₁ = 0.9):
  m = 0.9 × 0 + 0.1 × gradient = 0.1 × gradient  ← Only 10% of gradient!
```

### The Solution

Bias correction compensates for this initialization bias:

```
m̂ = m / (1 - β₁^t)

Iteration 1: m̂ = m / (1 - 0.9¹) = m / 0.1 = 10 × m  ← Corrected!
Iteration 10: m̂ = m / (1 - 0.9¹⁰) = m / 0.65 ≈ 1.5 × m
Iteration 100: m̂ ≈ m  ← Correction becomes negligible
```

As training progresses, the correction factor approaches 1 and has minimal effect.

---

## Step-by-Step Example

### Settings
```
β₁ = 0.9, β₂ = 0.999, lr = 0.001, ε = 1e-8
Initial: m = 0, v = 0
```

### Iteration 1: gradient = 0.5
```
t = 1
m = 0.9 × 0 + 0.1 × 0.5 = 0.05
v = 0.999 × 0 + 0.001 × 0.25 = 0.00025

# Bias correction
m̂ = 0.05 / (1 - 0.9¹) = 0.05 / 0.1 = 0.5
v̂ = 0.00025 / (1 - 0.999¹) = 0.00025 / 0.001 = 0.25

# Update
update = 0.001 × 0.5 / (√0.25 + 1e-8) = 0.001 × 0.5 / 0.5 = 0.001
```

### Iteration 2: gradient = 0.4
```
t = 2
m = 0.9 × 0.05 + 0.1 × 0.4 = 0.085
v = 0.999 × 0.00025 + 0.001 × 0.16 = 0.00041

# Bias correction
m̂ = 0.085 / (1 - 0.9²) = 0.085 / 0.19 = 0.447
v̂ = 0.00041 / (1 - 0.999²) = 0.00041 / 0.002 = 0.205

# Update
update = 0.001 × 0.447 / (√0.205 + 1e-8) ≈ 0.001
```

Notice how the update stays relatively stable around 0.001 (the learning rate).

---

## Default Hyperparameters

The original Adam paper recommends:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 0.001 | Learning rate |
| `β₁` | 0.9 | First moment decay |
| `β₂` | 0.999 | Second moment decay |
| `ε` | 1e-8 | Numerical stability |

These defaults work well for most problems without tuning!

---

## Implementation

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, weights, gradients):
        # Initialize on first call
        if self.m is None:
            self.m = torch.zeros_like(weights)
            self.v = torch.zeros_like(weights)
        
        self.t += 1
        
        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights
        weights = weights - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        return weights
```

### PyTorch Usage

```python
import torch.optim as optim

# Create Adam optimizer (defaults are usually fine)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Or with custom parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
```

---

## Adam Variants

### AdamW (Adam with Decoupled Weight Decay)

Standard Adam applies weight decay incorrectly. AdamW fixes this:

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Use AdamW when**: You're using L2 regularization (weight decay)

### AMSGrad

Fixes a theoretical convergence issue in Adam:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
```

**Use AMSGrad when**: Training is unstable or loss doesn't converge

### RAdam (Rectified Adam)

Automatically handles the warmup period:

```python
# Available in torch.optim since PyTorch 1.9
optimizer = optim.RAdam(model.parameters(), lr=0.001)
```

**Use RAdam when**: You want to skip learning rate warmup

---

## When to Use Adam

### Ideal For

- **Rapid prototyping**: Works well out-of-the-box
- **Complex architectures**: Transformers, deep CNNs
- **Sparse gradients**: NLP, embeddings
- **Non-stationary objectives**: GANs, reinforcement learning
- **When you don't want to tune**: Default parameters usually work

### Consider Alternatives When

- **Best generalization needed**: Tuned SGD+Momentum often generalizes better
- **Simple problems**: SGD may be sufficient and faster
- **Memory constrained**: Adam uses 2x memory for moment storage

---

## Adam vs. SGD: The Debate

| Aspect | Adam | SGD + Momentum |
|--------|------|----------------|
| **Ease of use** | Works out-of-box | Requires tuning |
| **Convergence speed** | Usually faster | Can be slower |
| **Generalization** | Sometimes worse | Often better |
| **Memory usage** | 2x parameters | 1x parameters |
| **Best for** | Prototyping, complex models | Final training, competitions |

### Practical Advice

1. **Start with Adam** for initial experiments
2. **Switch to SGD+Momentum** if you need better generalization
3. **Use AdamW** if using weight decay
4. **Try learning rate schedules** with either optimizer

---

## Common Issues and Solutions

### Issue: Training Loss Plateaus

**Solutions**:
- Reduce learning rate (try 0.0001)
- Add learning rate scheduling
- Check for data issues

### Issue: Validation Loss Diverges (Overfitting)

**Solutions**:
- Add weight decay (use AdamW)
- Reduce model capacity
- Add dropout or other regularization

### Issue: Loss Explodes

**Solutions**:
- Reduce learning rate significantly
- Add gradient clipping
- Check for NaN in data

---

## Key Takeaways

1. **Adam combines Momentum and RMSProp** for adaptive, accelerated optimization
2. **Bias correction** is crucial for stable early training
3. **Default parameters (lr=0.001, β₁=0.9, β₂=0.999)** work for most problems
4. **Use AdamW** when combining with weight decay
5. **Adam is the default choice** for most deep learning tasks

---

*Adam has become the de facto standard optimizer in deep learning. While tuned SGD can sometimes achieve better results, Adam's reliability and ease of use make it the go-to choice for most practitioners.*
