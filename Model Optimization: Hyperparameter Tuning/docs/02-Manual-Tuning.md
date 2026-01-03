# 🔧 Manual Hyperparameter Tuning

Manual tuning is the art of adjusting hyperparameters based on experience, intuition, and systematic experimentation. While automated methods exist, understanding manual tuning builds intuition that makes you a better practitioner.

---

## The Manual Tuning Process

### General Workflow

```
1. Start with reasonable defaults
   ↓
2. Train and observe metrics
   ↓
3. Diagnose problems (overfitting? underfitting? slow?)
   ↓
4. Adjust one hyperparameter at a time
   ↓
5. Repeat until satisfied
```

### Key Principles

1. **Change one thing at a time** — isolate the effect
2. **Keep records** — track what you tried and results
3. **Use validation set** — never tune on test data
4. **Be systematic** — don't just guess randomly

---

## Learning Rate Tuning

The learning rate is the most important hyperparameter. Get this right first.

### Starting Point

```python
# Good defaults by optimizer
optimizer_defaults = {
    'Adam': 0.001,      # Most common starting point
    'AdamW': 0.001,
    'SGD': 0.01,        # SGD needs higher LR
    'SGD+Momentum': 0.01,
    'RMSprop': 0.001,
}
```

### Diagnosing Learning Rate Problems

```
Loss behavior → Diagnosis → Action

Exploding/NaN     → LR too high    → Reduce by 10x
Wild oscillations → LR too high    → Reduce by 2-5x
Steady decrease   → LR good        → Keep it
Very slow decrease → LR too low    → Increase by 2-5x
Plateau early     → LR too low     → Increase, or use scheduler
```

### Visual Diagnosis

```
Loss vs Epochs:

LR too high:          LR too low:           LR just right:
    │                     │                     │
    │╲  ╱╲               │────────────        │╲
    │ ╲╱  ╲              │          ──        │ ╲
    │      ╲╱            │            ──      │  ╲___
    │                    │              ──    │      ───
    └──────────          └──────────────      └──────────
    Oscillates           Too slow             Smooth decrease
```

### Learning Rate Finder

A systematic way to find a good learning rate:

```python
import torch
import matplotlib.pyplot as plt

def find_learning_rate(model, train_loader, criterion, 
                       start_lr=1e-7, end_lr=1, num_steps=100):
    """
    Gradually increase LR and track loss.
    Good LR is where loss decreases fastest.
    """
    # Save initial weights
    initial_state = model.state_dict()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    
    # Calculate LR multiplier per step
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    
    lrs, losses = [], []
    lr = start_lr
    
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_steps:
            break
            
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Record
        lrs.append(lr)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Increase LR
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Restore initial weights
    model.load_state_dict(initial_state)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()
    
    return lrs, losses

# Usage
lrs, losses = find_learning_rate(model, train_loader, criterion)
# Look for the LR where loss decreases fastest (steepest slope)
```

### Interpreting the LR Finder Plot

```
Loss
  │
  │────╲
  │     ╲
  │      ╲         ← Steepest descent: good LR region
  │       ╲____
  │            ╲
  │             ╲___╱  ← Loss starts increasing: LR too high
  │
  └─────────────────────► Learning Rate (log scale)
     1e-7    1e-4    1e-1

Choose LR slightly before the minimum (e.g., 1e-3 if minimum at 1e-2)
```

---

## Batch Size Tuning

Batch size affects training dynamics, speed, and generalization.

### Trade-offs

| Batch Size | Pros | Cons |
|------------|------|------|
| **Small (8-32)** | Better generalization, less memory | Slower, noisier gradients |
| **Medium (32-128)** | Good balance | - |
| **Large (256+)** | Faster training, stable gradients | May generalize worse, needs LR adjustment |

### Batch Size and Learning Rate Relationship

```
Rule of thumb: Scale LR with batch size

If batch_size × 2 → learning_rate × √2

Example:
  batch_size=32,  lr=0.001
  batch_size=64,  lr=0.001 × √2 ≈ 0.0014
  batch_size=128, lr=0.001 × 2 ≈ 0.002
```

### Practical Guidelines

```python
# Start with these and adjust
batch_size_guidelines = {
    'small_dataset': 16,      # < 1000 samples
    'medium_dataset': 32,     # 1000-10000 samples
    'large_dataset': 64,      # 10000-100000 samples
    'very_large': 128,        # > 100000 samples
    'limited_memory': 8,      # GPU memory constrained
}
```

### Memory Considerations

```
Batch size limited by GPU memory:

batch_size = GPU_memory / (model_size + activations + gradients)

If out of memory:
  1. Reduce batch size
  2. Use gradient accumulation
  3. Use mixed precision (fp16)
```

### Gradient Accumulation

When you can't fit large batches in memory:

```python
# Simulate batch_size=64 with actual batch_size=16
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Architecture Tuning

### Number of Layers (Depth)

```
Start shallow, go deeper if underfitting:

1-2 layers:  Simple problems, small datasets
3-5 layers:  Most problems
6-10 layers: Complex problems, large datasets
10+ layers:  Very complex, need skip connections

Signs you need more layers:
  - Training loss plateaus high
  - Model can't fit training data
  - Validation and training loss both high
```

### Neurons per Layer (Width)

```
Start narrow, widen if underfitting:

32-64:   Simple problems
128-256: Most problems
512+:    Complex problems, large datasets

Signs you need more neurons:
  - Same as needing more layers
  - Model underfits
```

### Architecture Search Strategy

```python
# Start simple
architectures_to_try = [
    [64],           # 1 layer, 64 neurons
    [128],          # 1 layer, 128 neurons
    [64, 32],       # 2 layers
    [128, 64],      # 2 layers, wider
    [128, 64, 32],  # 3 layers
]

# Train each, compare validation performance
for arch in architectures_to_try:
    model = build_model(arch)
    train(model)
    val_acc = evaluate(model)
    print(f"{arch}: {val_acc:.4f}")
```

---

## Regularization Tuning

### When to Add Regularization

```
Overfitting signs:
  - Training loss << Validation loss
  - Training accuracy >> Validation accuracy
  - Validation loss starts increasing

If overfitting → Add regularization
If underfitting → Remove regularization
```

### Dropout Rate

```python
# Guidelines
dropout_guidelines = {
    'no_overfitting': 0.0,
    'mild_overfitting': 0.1-0.2,
    'moderate_overfitting': 0.3-0.4,
    'severe_overfitting': 0.5,
    'input_layer': 0.1-0.2,  # Lower for inputs
    'hidden_layers': 0.3-0.5,
    'before_output': 0.0,    # Never on output
}
```

### Weight Decay (L2)

```python
# Guidelines
weight_decay_guidelines = {
    'no_regularization': 0,
    'light': 1e-5,
    'moderate': 1e-4,  # Good default
    'strong': 1e-3,
    'very_strong': 1e-2,
}

# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## Systematic Tuning Approach

### Step-by-Step Process

```
Step 1: Establish Baseline
  - Use default hyperparameters
  - Record training and validation metrics
  - This is your reference point

Step 2: Tune Learning Rate
  - Try: 0.1, 0.01, 0.001, 0.0001
  - Use LR finder for more precision
  - Pick best based on validation performance

Step 3: Tune Architecture
  - Start simple (1-2 layers)
  - Increase if underfitting
  - Decrease if overfitting

Step 4: Tune Batch Size
  - Try: 16, 32, 64, 128
  - Adjust LR accordingly
  - Consider memory constraints

Step 5: Tune Regularization
  - Add dropout if overfitting
  - Add weight decay if overfitting
  - Remove if underfitting

Step 6: Fine-tune
  - Small adjustments to best configuration
  - Try LR scheduling
  - Longer training with early stopping
```

### Keeping Records

```python
# Track your experiments
experiments = []

def log_experiment(config, results):
    experiments.append({
        'config': config,
        'train_loss': results['train_loss'],
        'val_loss': results['val_loss'],
        'val_accuracy': results['val_accuracy'],
        'notes': results.get('notes', '')
    })

# Example
log_experiment(
    config={'lr': 0.001, 'batch_size': 32, 'layers': [128, 64]},
    results={'train_loss': 0.15, 'val_loss': 0.22, 'val_accuracy': 0.92,
             'notes': 'Slight overfitting, try dropout'}
)
```

---

## Common Patterns and Fixes

### Pattern: Loss Not Decreasing

```
Possible causes:
  1. Learning rate too low → Increase LR
  2. Learning rate too high → Decrease LR
  3. Bug in code → Check data pipeline, loss function
  4. Model too simple → Add capacity
```

### Pattern: Loss Oscillates

```
Possible causes:
  1. Learning rate too high → Decrease LR
  2. Batch size too small → Increase batch size
  3. Data not shuffled → Enable shuffling
```

### Pattern: Overfitting

```
Possible causes:
  1. Model too complex → Reduce layers/neurons
  2. No regularization → Add dropout, weight decay
  3. Training too long → Use early stopping
  4. Not enough data → Data augmentation
```

### Pattern: Underfitting

```
Possible causes:
  1. Model too simple → Add layers/neurons
  2. Too much regularization → Reduce dropout, weight decay
  3. Learning rate too low → Increase LR
  4. Not enough training → More epochs
```

---

## Key Takeaways

1. **Start with good defaults** and adjust based on observations
2. **Learning rate is most important** — use LR finder
3. **Change one thing at a time** to understand effects
4. **Keep detailed records** of experiments
5. **Diagnose before fixing** — understand the problem first
6. **Batch size and LR are related** — adjust together
7. **Add regularization only if overfitting**

---

*Manual tuning builds intuition that automated methods can't replace. Even when using automated tuning, understanding these principles helps you set up better search spaces and interpret results.*
