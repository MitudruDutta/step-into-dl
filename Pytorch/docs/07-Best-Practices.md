# ✅ PyTorch Best Practices

Guidelines for writing efficient, bug-free PyTorch code.

---

## Memory Management

### Use `torch.no_grad()` During Inference

Disabling gradient tracking saves memory and speeds up computation:

```python
model.eval()  # Set to evaluation mode

with torch.no_grad():
    predictions = model(test_data)
    # No computation graph built
    # ~50% less memory usage
```

### Clear Gradients Before Each Backward Pass

Gradients accumulate by default—always zero them:

```python
# In training loop
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute new gradients
optimizer.step()       # Update weights
```

### Use `.detach()` When Needed

Remove tensors from computation graph when you only need values:

```python
# For logging/visualization
loss_value = loss.detach().item()

# For using intermediate values without gradients
features = model.encoder(x).detach()
```

### Delete Unused Tensors

Free memory explicitly when working with large tensors:

```python
del large_tensor
torch.cuda.empty_cache()  # Free GPU memory
```

---

## Device Management

### Always Check Device Compatibility

All tensors in an operation must be on the same device:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to same device
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

### Use Device-Agnostic Code

Write code that works on both CPU and GPU:

```python
# Bad: hardcoded device
x = torch.rand(3, 3).cuda()

# Good: device-agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand(3, 3, device=device)
```

### Move Entire Models at Once

```python
# Move all parameters and buffers
model.to(device)

# Not this (only moves one layer)
model.layer1.to(device)
```

---

## Data Types

### Use Appropriate dtypes

| Use Case | Recommended dtype |
|----------|-------------------|
| Model weights | `float32` |
| Input data | `float32` |
| Class labels | `long` (int64) |
| Indices | `long` (int64) |
| Masks | `bool` |
| Mixed precision | `float16` |

```python
# Explicit dtype specification
inputs = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)
```

### Consider Mixed Precision Training

For modern GPUs, mixed precision can speed up training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # Automatic mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Training Loop Best Practices

### Standard Training Loop Template

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Set to training mode
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Standard Evaluation Loop Template

```python
def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # For classification
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy
```

---

## Model Checkpointing

### Save Complete State

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_accuracy,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### Save Best Model Only

```python
best_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
```

---

## Reproducibility

### Set Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## DataLoader Best Practices

### Optimal DataLoader Settings

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Shuffle training data
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True,         # Drop incomplete batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,          # Can use larger batch for eval
    shuffle=False,          # Don't shuffle validation
    num_workers=4,
    pin_memory=True,
)
```

### Choose `num_workers` Wisely

- Start with `num_workers=4`
- Increase if data loading is the bottleneck
- Set to 0 for debugging (easier to trace errors)

---

## Debugging Tips

### Check Tensor Shapes

```python
def debug_shapes(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

# Use throughout your code
debug_shapes("input", x)
debug_shapes("output", y)
```

### Check for NaN/Inf

```python
def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"WARNING: {name} contains NaN")
    if torch.isinf(tensor).any():
        print(f"WARNING: {name} contains Inf")
```

### Gradient Checking

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
    else:
        print(f"{name}: no gradient")
```

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Forgetting `model.train()`/`model.eval()` | Dropout/BatchNorm behave wrong | Always set mode |
| Not zeroing gradients | Gradients accumulate | Call `optimizer.zero_grad()` |
| In-place operations on leaf tensors | Autograd errors | Avoid `_` operations during training |
| Tensors on different devices | Runtime error | Move all to same device |
| Not using `torch.no_grad()` for inference | Wasted memory | Wrap inference in context |
| Hardcoding device | Code breaks on CPU-only machines | Use device-agnostic code |

---

*Following these best practices will help you write cleaner, faster, and more reliable PyTorch code.*
