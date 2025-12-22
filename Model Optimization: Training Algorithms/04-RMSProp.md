# 📊 RMSProp (Root Mean Square Propagation)

RMSProp is specifically designed to handle "noisy" gradients—situations where the error fluctuates wildly between updates. It adapts the learning rate for each parameter individually.

---

## The Problem RMSProp Solves

### Uneven Gradient Magnitudes

In many neural networks, different parameters have vastly different gradient magnitudes:
- Some dimensions have large, noisy gradients → cause wild oscillations
- Other dimensions have small, consistent gradients → need larger steps

A single global learning rate can't handle both situations well:
- Too high → large gradient dimensions explode
- Too low → small gradient dimensions learn too slowly

### Visual Example

```
Parameter A: gradient = [100, -95, 105, -98, 102]  ← Large, noisy
Parameter B: gradient = [0.1, 0.1, 0.1, 0.1, 0.1] ← Small, consistent

With fixed LR = 0.01:
- Parameter A: steps of ±1.0 (too large, oscillates)
- Parameter B: steps of 0.001 (too small, slow progress)
```

---

## How RMSProp Works

### The Algorithm

```
# Initialize squared gradient accumulator
s = 0

# For each iteration:
s = β × s + (1 - β) × gradient²    # Accumulate squared gradients
weights = weights - lr × gradient / √(s + ε)
```

### Breaking It Down

| Term | Meaning |
|------|---------|
| `s` | Running average of squared gradients |
| `β` | Decay rate (typically 0.9 or 0.99) |
| `gradient²` | Element-wise squared gradient |
| `ε` | Small constant for numerical stability (1e-8) |
| `√(s + ε)` | Normalization factor |

### The Key Insight

By dividing by the root mean square of recent gradients:
- **Large gradients** → large `s` → smaller effective step
- **Small gradients** → small `s` → larger effective step

This automatically adapts the learning rate per parameter!

---

## Step-by-Step Example

### Scenario: Two Parameters with Different Gradient Behaviors

```
Parameter A (noisy):     gradients = [10, -8, 12, -9, 11]
Parameter B (consistent): gradients = [1, 1, 1, 1, 1]

Settings: β = 0.9, lr = 0.1, ε = 1e-8
```

### For Parameter A (noisy gradients):

```
Iteration 1: g = 10
  s = 0.9 × 0 + 0.1 × 100 = 10
  update = 0.1 × 10 / √10 = 0.316

Iteration 2: g = -8
  s = 0.9 × 10 + 0.1 × 64 = 15.4
  update = 0.1 × (-8) / √15.4 = -0.204

Iteration 3: g = 12
  s = 0.9 × 15.4 + 0.1 × 144 = 28.26
  update = 0.1 × 12 / √28.26 = 0.226
```

The large, noisy gradients get normalized → smaller, stable updates.

### For Parameter B (consistent gradients):

```
Iteration 1: g = 1
  s = 0.9 × 0 + 0.1 × 1 = 0.1
  update = 0.1 × 1 / √0.1 = 0.316

Iteration 2: g = 1
  s = 0.9 × 0.1 + 0.1 × 1 = 0.19
  update = 0.1 × 1 / √0.19 = 0.229

Iteration 3: g = 1
  s = 0.9 × 0.19 + 0.1 × 1 = 0.271
  update = 0.1 × 1 / √0.271 = 0.192
```

The small, consistent gradients maintain reasonable step sizes.

---

## The Decay Rate (β)

### Typical Values

| β Value | Memory | Use Case |
|---------|--------|----------|
| 0.9 | ~10 iterations | Standard choice |
| 0.99 | ~100 iterations | Very noisy gradients |
| 0.999 | ~1000 iterations | Extremely noisy |

### Effect of β

**High β (0.99)**:
- Longer memory of past gradients
- More stable normalization
- Slower to adapt to changes

**Low β (0.9)**:
- Shorter memory
- More responsive to recent gradients
- Can be unstable if gradients vary wildly

---

## RMSProp vs. Momentum

| Aspect | Momentum | RMSProp |
|--------|----------|---------|
| **What it tracks** | Direction (gradient mean) | Magnitude (gradient variance) |
| **Purpose** | Accelerate consistent directions | Normalize per-parameter |
| **Handles** | Narrow valleys | Noisy gradients |
| **Updates** | Same scale for all params | Adaptive per parameter |

### Complementary Approaches

- **Momentum**: "Keep going in the same direction"
- **RMSProp**: "Adjust step size based on gradient history"

This is why Adam combines both!

---

## Implementation

```python
class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = None  # Squared gradient accumulator
    
    def update(self, weights, gradients):
        # Initialize on first call
        if self.s is None:
            self.s = torch.zeros_like(weights)
        
        # Update squared gradient average
        self.s = self.beta * self.s + (1 - self.beta) * gradients ** 2
        
        # Update weights with normalized gradients
        weights = weights - self.lr * gradients / (torch.sqrt(self.s) + self.epsilon)
        
        return weights
```

### PyTorch Usage

```python
import torch.optim as optim

# Create RMSProp optimizer
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
```

Note: PyTorch uses `alpha` instead of `beta` for the decay rate.

---

## When to Use RMSProp

### Ideal For

- **Recurrent Neural Networks (RNNs)**: Gradients can vary wildly across time steps
- **Non-stationary problems**: When the optimal solution changes over time
- **Noisy gradients**: Mini-batch training with high variance
- **Sparse features**: Some features appear rarely but are important

### Less Ideal For

- Simple problems where SGD works fine
- When you want momentum-like acceleration
- Very stable gradient landscapes

---

## Common Issues and Solutions

### Issue: Loss Explodes

**Cause**: Learning rate too high for the normalized gradients

**Solution**: Reduce learning rate (try 0.0001)

### Issue: Training Stalls

**Cause**: `s` accumulates too much, making updates tiny

**Solution**: 
- Reduce β to forget old gradients faster
- Increase learning rate slightly

### Issue: Unstable Early Training

**Cause**: `s` is small initially, causing large updates

**Solution**: Use learning rate warmup or start with smaller LR

---

## Key Takeaways

1. **RMSProp normalizes gradients** per-parameter using squared gradient history
2. **Large gradients get smaller steps**, small gradients get larger steps
3. **β = 0.9** is a good default starting point
4. **Especially useful for RNNs** and problems with noisy gradients
5. **Complements Momentum** — they solve different problems

---

*RMSProp was proposed by Geoffrey Hinton in his Coursera course and quickly became popular for training RNNs. It's a key building block of the Adam optimizer.*
