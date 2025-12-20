# ⚡ Gradient Descent Variants: Batch, Mini-Batch, and SGD

Different variations of Gradient Descent offer trade-offs between speed, stability, and computational requirements. Choosing the right variant is crucial for efficient training.

---

## Overview

| Variant | Data per Update | Updates per Epoch |
|---------|-----------------|-------------------|
| **Batch GD** | Entire dataset | 1 |
| **Mini-Batch GD** | Small subset (32-512) | Dataset / Batch size |
| **SGD** | Single sample | Dataset size |

---

## Batch Gradient Descent

Uses the **entire dataset** to compute gradients before each weight update.

### Algorithm
```python
for epoch in range(num_epochs):
    gradient = compute_gradient(entire_dataset)
    weights = weights - learning_rate * gradient
```

### Characteristics

| Aspect | Description |
|--------|-------------|
| **Gradient Quality** | Exact gradient (no noise) |
| **Convergence Path** | Smooth, predictable |
| **Memory Usage** | High (entire dataset in memory) |
| **Speed** | Slow (one update per epoch) |
| **GPU Utilization** | Poor (can't parallelize well) |

### Pros
- Stable, predictable convergence
- Guaranteed to converge for convex functions
- Clean gradient estimates

### Cons
- Computationally expensive for large datasets
- Requires entire dataset in memory
- Can get stuck in local minima (smooth path)
- Very slow—only one update per epoch

### Best For
- Small datasets (< 10,000 samples)
- When stability is critical
- Convex optimization problems

---

## Stochastic Gradient Descent (SGD)

Uses a **single random sample** to compute gradients for each update.

### Algorithm
```python
for epoch in range(num_epochs):
    shuffle(dataset)
    for sample in dataset:
        gradient = compute_gradient(sample)
        weights = weights - learning_rate * gradient
```

### Characteristics

| Aspect | Description |
|--------|-------------|
| **Gradient Quality** | Noisy estimate |
| **Convergence Path** | Erratic, high variance |
| **Memory Usage** | Very low (one sample) |
| **Speed** | Fast updates |
| **GPU Utilization** | Poor (no batching) |

### Pros
- Very fast updates (many per epoch)
- Low memory requirements
- Noise helps escape local minima
- Good for online learning (streaming data)

### Cons
- High variance in updates
- Noisy convergence path
- May oscillate around minimum
- Can't leverage GPU parallelism

### Best For
- Very large datasets
- Online/streaming learning
- When memory is severely limited

---

## Mini-Batch Gradient Descent

Uses a **small subset (batch)** of data for each gradient computation. **The industry standard.**

### Algorithm
```python
for epoch in range(num_epochs):
    shuffle(dataset)
    for batch in create_batches(dataset, batch_size=32):
        gradient = compute_gradient(batch)
        weights = weights - learning_rate * gradient
```

### Characteristics

| Aspect | Description |
|--------|-------------|
| **Gradient Quality** | Good estimate (averaged over batch) |
| **Convergence Path** | Moderately smooth |
| **Memory Usage** | Moderate (one batch) |
| **Speed** | Balanced |
| **GPU Utilization** | Excellent (parallel processing) |

### Pros
- Balances speed and stability
- Efficient GPU utilization
- Some noise helps escape local minima
- Works well for most problems

### Cons
- Requires tuning batch size
- Slightly noisier than batch GD

### Best For
- Most real-world applications
- Deep learning (CNNs, Transformers)
- When you have GPU access

---

## Comparison Table

| Feature | Batch GD | Mini-Batch GD | SGD |
|---------|----------|---------------|-----|
| **Data per Update** | All | 32-512 | 1 |
| **Updates per Epoch** | 1 | N/batch_size | N |
| **Convergence Speed** | Slow | Balanced | Fast |
| **Memory Usage** | High | Moderate | Low |
| **Gradient Noise** | None | Low-Medium | High |
| **Local Minima Risk** | High | Medium | Low |
| **GPU Efficiency** | Poor | Excellent | Poor |

---

## Choosing Batch Size

### Common Batch Sizes
- **32**: Good default, works on most GPUs
- **64**: Slightly more stable
- **128**: Good for larger models
- **256-512**: For very large models with lots of GPU memory

### Trade-offs

| Larger Batches | Smaller Batches |
|----------------|-----------------|
| More stable gradients | Noisier gradients |
| Slower convergence | Faster convergence |
| Better GPU utilization | Worse GPU utilization |
| May generalize worse | Often generalizes better |
| Requires more memory | Requires less memory |

### Guidelines

```
Batch Size Selection:
├── Limited GPU memory? → Use smaller batches (16-32)
├── Training unstable? → Increase batch size
├── Converging too slowly? → Decrease batch size
├── Want better generalization? → Use smaller batches
└── Default choice → 32 or 64
```

---

## Practical Tips

### 1. Always Shuffle Data
```python
# Shuffle at the start of each epoch
indices = torch.randperm(len(dataset))
shuffled_data = dataset[indices]
```

### 2. Use Powers of 2
Batch sizes of 32, 64, 128, 256 are optimal for GPU memory alignment.

### 3. Scale Learning Rate with Batch Size
When increasing batch size, consider increasing learning rate:
```python
# Linear scaling rule
lr = base_lr * (batch_size / 32)
```

### 4. Gradient Accumulation
Simulate larger batches with limited memory:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Selection Guide

| Scenario | Recommended | Batch Size |
|----------|-------------|------------|
| Small dataset (< 10K) | Batch or Mini-Batch | Full or 32-64 |
| Medium dataset (10K-1M) | Mini-Batch | 32-128 |
| Large dataset (> 1M) | Mini-Batch or SGD | 64-256 |
| Limited memory | SGD or small Mini-Batch | 8-32 |
| Need stable training | Mini-Batch | 128-256 |
| Online learning | SGD | 1 |
| GPU training | Mini-Batch | 32-256 |

---

## PyTorch DataLoader

```python
from torch.utils.data import DataLoader

# Create dataloader with batching and shuffling
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,      # Shuffle each epoch
    num_workers=4,     # Parallel data loading
    pin_memory=True    # Faster GPU transfer
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Mini-batch gradient descent
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

*Mini-batch gradient descent is the default choice for modern deep learning. Start with batch size 32 and adjust based on your specific needs.*
