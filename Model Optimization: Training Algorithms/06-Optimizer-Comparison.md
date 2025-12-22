# 🔄 Optimizer Comparison & Practical Guidelines

This guide provides a comprehensive comparison of optimization algorithms and practical advice for choosing the right optimizer for your deep learning project.

---

## Quick Comparison Table

| Optimizer | Speed | Ease of Use | Generalization | Memory | Best For |
|-----------|-------|-------------|----------------|--------|----------|
| **SGD** | Slow | Hard | Excellent | Low | Final training |
| **SGD + Momentum** | Medium | Medium | Excellent | Low | Most tasks |
| **RMSProp** | Fast | Easy | Good | Medium | RNNs |
| **Adam** | Fast | Very Easy | Good | High | Prototyping |
| **AdamW** | Fast | Very Easy | Very Good | High | With regularization |

---

## Detailed Comparison

### What Each Optimizer Tracks

```
SGD:           Nothing (just current gradient)
Momentum:      v = gradient direction (1st moment)
RMSProp:       s = gradient magnitude (2nd moment)
Adam:          m = direction + v = magnitude (both moments)
```

### Update Rules Side-by-Side

**SGD**:
```
w = w - lr × gradient
```

**SGD + Momentum**:
```
v = β × v + gradient
w = w - lr × v
```

**RMSProp**:
```
s = β × s + (1-β) × gradient²
w = w - lr × gradient / √(s + ε)
```

**Adam**:
```
m = β₁ × m + (1-β₁) × gradient
v = β₂ × v + (1-β₂) × gradient²
w = w - lr × m̂ / (√v̂ + ε)
```

---

## Choosing an Optimizer: Decision Tree

```
Start Here
    │
    ▼
┌─────────────────────────────┐
│ Is this a quick experiment? │
└─────────────────────────────┘
    │
    ├── Yes ──► Use Adam (lr=0.001)
    │
    ▼ No
┌─────────────────────────────┐
│ Is it an RNN/LSTM/GRU?      │
└─────────────────────────────┘
    │
    ├── Yes ──► Use RMSProp or Adam
    │
    ▼ No
┌─────────────────────────────┐
│ Need best generalization?   │
└─────────────────────────────┘
    │
    ├── Yes ──► Use SGD + Momentum (tune LR)
    │
    ▼ No
┌─────────────────────────────┐
│ Using weight decay/L2?      │
└─────────────────────────────┘
    │
    ├── Yes ──► Use AdamW
    │
    ▼ No
    │
    └──► Use Adam (default choice)
```

---

## Learning Rate Guidelines

### Typical Starting Learning Rates

| Optimizer | Starting LR | Range to Try |
|-----------|-------------|--------------|
| SGD | 0.1 | 0.01 - 1.0 |
| SGD + Momentum | 0.01 | 0.001 - 0.1 |
| RMSProp | 0.001 | 0.0001 - 0.01 |
| Adam | 0.001 | 0.0001 - 0.01 |
| AdamW | 0.001 | 0.0001 - 0.01 |

### Learning Rate Schedules

Reducing the learning rate during training often improves results:

**Step Decay**:
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**Cosine Annealing**:
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

**Reduce on Plateau**:
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
```

---

## Optimizer Behavior Visualization

### Loss Landscape Navigation

```
Narrow Valley (Momentum helps):

SGD:        ╱╲╱╲╱╲╱╲→ minimum (slow, oscillates)
Momentum:   ────────→ minimum (fast, smooth)

Noisy Gradients (RMSProp helps):

SGD:        ↗↙↗↙↗↙ (unstable)
RMSProp:    →→→→→→ (stable, normalized)

Both Challenges (Adam handles):

Adam:       ────────→ minimum (fast, stable)
```

---

## Common Scenarios & Recommendations

### Scenario 1: Image Classification (CNN)

**Recommended**: SGD + Momentum or Adam

```python
# Option A: Adam for quick results
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Option B: SGD for best accuracy (requires tuning)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Scenario 2: Natural Language Processing (Transformer)

**Recommended**: AdamW with warmup

```python
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Linear warmup then decay
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=1000, 
    num_training_steps=10000
)
```

### Scenario 3: Recurrent Neural Networks

**Recommended**: RMSProp or Adam with gradient clipping

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# In training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Scenario 4: Generative Adversarial Networks (GANs)

**Recommended**: Adam with specific betas

```python
# Generator
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Discriminator
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

### Scenario 5: Fine-tuning Pretrained Models

**Recommended**: AdamW with small learning rate

```python
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

---

## Troubleshooting Guide

### Problem: Loss Not Decreasing

| Possible Cause | Solution |
|----------------|----------|
| LR too low | Increase learning rate |
| LR too high | Decrease learning rate |
| Bad initialization | Try different weight init |
| Data issue | Check data preprocessing |

### Problem: Loss Oscillates Wildly

| Possible Cause | Solution |
|----------------|----------|
| LR too high | Reduce learning rate |
| No momentum | Add momentum (β=0.9) |
| Batch too small | Increase batch size |

### Problem: Training Slow

| Possible Cause | Solution |
|----------------|----------|
| LR too low | Increase learning rate |
| No momentum | Switch to Momentum/Adam |
| Wrong optimizer | Try Adam for faster convergence |

### Problem: Good Training, Bad Validation

| Possible Cause | Solution |
|----------------|----------|
| Overfitting | Add regularization |
| Adam generalization | Try SGD + Momentum |
| No weight decay | Use AdamW with weight_decay |

---

## Memory Considerations

### Memory Usage per Parameter

| Optimizer | Memory Multiplier | For 100M params |
|-----------|-------------------|-----------------|
| SGD | 1x | 400 MB |
| SGD + Momentum | 2x | 800 MB |
| RMSProp | 2x | 800 MB |
| Adam | 3x | 1.2 GB |

Adam stores: weights + first moment + second moment

### Memory-Constrained Situations

If memory is tight:
1. Use SGD + Momentum instead of Adam
2. Use gradient checkpointing
3. Reduce batch size
4. Use mixed precision training

---

## Best Practices Summary

### Do's

1. **Start with Adam** for initial experiments
2. **Use learning rate schedules** for better convergence
3. **Monitor both training and validation loss**
4. **Use AdamW** when applying weight decay
5. **Clip gradients** for RNNs and unstable training

### Don'ts

1. **Don't use a fixed learning rate** for the entire training
2. **Don't ignore validation metrics** (overfitting risk)
3. **Don't assume one optimizer fits all** problems
4. **Don't skip hyperparameter search** for important projects
5. **Don't use very high learning rates** with Adam

---

## Quick Reference: PyTorch Code

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD + Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD + Nesterov Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop pattern
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # If using a scheduler
```

---

## Key Takeaways

1. **Adam is the safe default** — works well without tuning
2. **SGD + Momentum can generalize better** — worth trying for final models
3. **RMSProp excels with RNNs** — handles sequential data well
4. **AdamW is preferred over Adam** — when using weight decay
5. **Learning rate is the most important hyperparameter** — always tune it
6. **Use schedulers** — reducing LR over time usually helps

---

*The best optimizer depends on your specific problem. Start with Adam, measure results, and experiment with alternatives if needed.*
