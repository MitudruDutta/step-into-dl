# 📊 Monitoring Training: Metrics, Debugging, and Early Stopping

Effective training requires careful monitoring. This guide covers what to watch, how to interpret metrics, and when to stop training.

---

## Key Metrics to Monitor

### Training Loss
The primary optimization target—should decrease over time.

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Steadily decreasing | Healthy training | Continue |
| Stuck/plateaued | Learning rate too low or converged | Adjust LR or stop |
| Increasing | Learning rate too high | Reduce LR |
| Oscillating wildly | Learning rate too high | Reduce LR significantly |
| NaN/Inf | Numerical instability | Check data, reduce LR, add gradient clipping |

### Validation Loss
Measures generalization—how well the model performs on unseen data.

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Decreasing with training loss | Healthy learning | Continue |
| Increasing while training decreases | Overfitting | Stop training, add regularization |
| Higher than training from start | Possible data issue | Check data distribution |
| Same as training loss | Good generalization | Continue |

### The Training-Validation Gap

```
Loss
  │
  │  Training ────────────────
  │                    ╲
  │                     ╲ Validation
  │                      ╲
  │                       ╲────────
  └──────────────────────────────── Epochs
         ↑
    Overfitting starts here
```

---

## Detecting Common Problems

### Overfitting
**Symptoms:**
- Training loss continues to decrease
- Validation loss starts increasing
- Large gap between training and validation metrics

**Solutions:**
- Early stopping
- Add dropout
- Data augmentation
- L2 regularization (weight decay)
- Reduce model size

### Underfitting
**Symptoms:**
- Both training and validation loss are high
- Model performs poorly on training data
- Loss plateaus early

**Solutions:**
- Increase model capacity (more layers/neurons)
- Train longer
- Reduce regularization
- Use a more expressive architecture

### Vanishing Gradients
**Symptoms:**
- Early layers don't learn (weights don't change)
- Loss decreases very slowly
- Gradient norms near zero

**Solutions:**
- Use ReLU instead of sigmoid/tanh
- Add batch normalization
- Use residual connections
- Proper weight initialization

### Exploding Gradients
**Symptoms:**
- Loss becomes NaN or Inf
- Weights become very large
- Training becomes unstable

**Solutions:**
- Gradient clipping
- Lower learning rate
- Proper weight initialization
- Batch normalization

---

## Monitoring Tools

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

### Weights & Biases
```python
import wandb

wandb.init(project="my-project")

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch
    })
```

### Simple Logging
```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

---

## Early Stopping

Stop training when validation loss stops improving to prevent overfitting.

### Implementation
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(1000):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Patience Selection

| Dataset Size | Recommended Patience |
|--------------|---------------------|
| Small (< 10K) | 5-10 epochs |
| Medium (10K-100K) | 10-20 epochs |
| Large (> 100K) | 20-50 epochs |

---

## Model Checkpointing

Save model weights periodically to recover from crashes and keep best models.

### Save Best Model
```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pt')
```

### Load Checkpoint
```python
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Gradient Monitoring

### Check Gradient Norms
```python
def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# During training
for epoch in range(num_epochs):
    loss.backward()
    grad_norm = get_gradient_norm(model)
    print(f"Gradient norm: {grad_norm:.4f}")
    optimizer.step()
```

### Healthy Gradient Norms

| Gradient Norm | Interpretation |
|---------------|----------------|
| 0.1 - 10 | Healthy range |
| < 0.001 | Vanishing gradients |
| > 100 | Exploding gradients |
| NaN | Numerical instability |

---

## When to Stop Training

### Stop When:
1. **Validation loss stops improving** (early stopping triggered)
2. **Training loss converges** (no longer decreasing)
3. **Time/compute budget exhausted**
4. **Target metric achieved**

### Don't Stop When:
1. Training loss is still decreasing significantly
2. Validation loss is still improving
3. You haven't trained long enough to see patterns

---

## Training Checklist

Before training:
- [ ] Data is properly normalized
- [ ] Train/val/test splits are correct
- [ ] Model architecture is appropriate
- [ ] Learning rate is reasonable
- [ ] Batch size fits in memory

During training:
- [ ] Loss is decreasing
- [ ] No NaN/Inf values
- [ ] Gradients are in healthy range
- [ ] Validation loss tracks training loss

After training:
- [ ] Best model is saved
- [ ] Final metrics are logged
- [ ] Model generalizes to test set

---

*Monitoring is not optional—it's essential for successful training. Set up logging from the start and check metrics regularly.*
