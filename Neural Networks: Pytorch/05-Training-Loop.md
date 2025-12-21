# 🔄 Putting It All Together: The Training Loop

A complete training workflow combines all PyTorch components: models, data loaders, loss functions, and optimizers. This guide shows how to assemble them into a working training pipeline.

---

## The Training Loop Structure

Every PyTorch training loop follows this pattern:

```
1. Define model (nn.Module)
2. Create DataLoaders (batching, shuffling)
3. Choose loss function (BCE or CrossEntropy)
4. Select optimizer (Adam, SGD)
5. Training loop:
   - Forward pass
   - Calculate loss
   - Backward pass
   - Update weights
6. Validation loop
7. Save best model
```

---

## Complete Training Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. Training loop
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()          # Clear gradients
        output = model(batch_x)        # Forward pass
        loss = criterion(output, batch_y)  # Calculate loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item()
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
```

---

## Key Steps Explained

### Step 1: Zero Gradients

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients by default. You must clear them before each batch to prevent gradients from previous batches affecting the current update.

### Step 2: Forward Pass

```python
output = model(batch_x)
```

Data flows through the network, producing predictions. PyTorch builds a computation graph for backpropagation.

### Step 3: Calculate Loss

```python
loss = criterion(output, batch_y)
```

Compare predictions to targets. The loss is a single scalar value representing how wrong the model is.

### Step 4: Backward Pass

```python
loss.backward()
```

Compute gradients for all parameters by backpropagating through the computation graph.

### Step 5: Update Weights

```python
optimizer.step()
```

Adjust weights using the computed gradients and the optimizer's update rule.

---

## Binary vs Multi-Class Setup

| Decision | Binary Classification | Multi-Class Classification |
|----------|----------------------|---------------------------|
| **Final Layer** | 1 neuron | N neurons (N = classes) |
| **Activation** | Sigmoid (or in loss) | None (softmax in loss) |
| **Loss Function** | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| **Target Type** | Float [0.0, 1.0] | Long (integer indices) |
| **Prediction** | `sigmoid(output) > 0.5` | `argmax(output)` |


---

## Loss Function Selection Guide

| Problem Type | Loss Function | Final Activation |
|--------------|---------------|------------------|
| Regression | `MSELoss`, `L1Loss` | None (linear) |
| Binary Classification | `BCEWithLogitsLoss` | None (sigmoid in loss) |
| Multi-Class (single label) | `CrossEntropyLoss` | None (softmax in loss) |
| Multi-Label | `BCEWithLogitsLoss` | None (sigmoid in loss) |

---

## Training vs Evaluation Mode

```python
model.train()  # Training mode
model.eval()   # Evaluation mode
```

**Why it matters:**
- **Dropout**: Active in train, disabled in eval
- **BatchNorm**: Uses batch stats in train, running stats in eval

**Always set the correct mode:**
```python
# Training
model.train()
for batch in train_loader:
    # ... training code

# Validation/Testing
model.eval()
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:
        # ... evaluation code
```

---

## Gradient Context Managers

### torch.no_grad()

Disables gradient computation for efficiency during inference:

```python
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

**Benefits:**
- Faster computation
- Less memory usage
- Required for inference

### torch.enable_grad()

Re-enables gradients inside a no_grad block (rarely needed):

```python
with torch.no_grad():
    # No gradients here
    with torch.enable_grad():
        # Gradients enabled again
```

---

## Common Mistakes to Avoid

### 1. Forgetting to Zero Gradients

```python
# ❌ Wrong: Gradients accumulate
for batch in loader:
    output = model(batch)
    loss.backward()
    optimizer.step()

# ✅ Correct: Zero gradients each iteration
for batch in loader:
    optimizer.zero_grad()  # Add this!
    output = model(batch)
    loss.backward()
    optimizer.step()
```

### 2. Wrong Mode for Validation

```python
# ❌ Wrong: Training mode during validation
for batch in val_loader:
    output = model(batch)  # Dropout still active!

# ✅ Correct: Evaluation mode
model.eval()
with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
```

### 3. Forgetting to Move Data to Device

```python
# ❌ Wrong: Data on CPU, model on GPU
model = model.to('cuda')
output = model(batch)  # Error!

# ✅ Correct: Move data to same device
batch = batch.to(device)
output = model(batch)
```

### 4. Using Softmax with CrossEntropyLoss

```python
# ❌ Wrong: Double softmax
output = torch.softmax(model(x), dim=1)
loss = nn.CrossEntropyLoss()(output, target)

# ✅ Correct: Raw logits
output = model(x)
loss = nn.CrossEntropyLoss()(output, target)
```

---

## Adding Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Option 1: Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Option 2: Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

# In training loop
for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = validate()
    
    # Update scheduler
    scheduler.step()  # For StepLR
    # or
    scheduler.step(val_loss)  # For ReduceLROnPlateau
```

---

## Early Stopping

```python
patience = 5
best_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## Saving and Loading Models

### Save Model

```python
# Save entire model (not recommended)
torch.save(model, 'model.pth')

# Save state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Save checkpoint (for resuming training)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### Load Model

```python
# Load state dict
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Quick Reference Checklist

Before training:
- [ ] Model moved to device (`model.to(device)`)
- [ ] Loss function chosen correctly
- [ ] Optimizer configured
- [ ] DataLoaders created with appropriate settings

During training:
- [ ] `model.train()` called
- [ ] `optimizer.zero_grad()` before each batch
- [ ] Data moved to device
- [ ] `loss.backward()` called
- [ ] `optimizer.step()` called

During validation:
- [ ] `model.eval()` called
- [ ] `torch.no_grad()` context used
- [ ] Data moved to device

---

*The training loop is where everything comes together. Master this pattern and you can train any neural network in PyTorch.*
