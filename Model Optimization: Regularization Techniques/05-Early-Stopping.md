# ⏱️ Early Stopping

Early stopping is the simplest and most intuitive regularization technique—stop training when the model starts to overfit. Despite its simplicity, it's remarkably effective and should be part of every practitioner's toolkit.

---

## The Core Idea

### Why Early Stopping Works

During training, a model typically goes through three phases:

```
Phase 1: Underfitting
  - Both training and validation loss decrease
  - Model is learning useful patterns

Phase 2: Optimal
  - Training loss continues to decrease
  - Validation loss reaches minimum
  - Best generalization

Phase 3: Overfitting
  - Training loss keeps decreasing
  - Validation loss starts increasing
  - Model is memorizing training data
```

Early stopping captures the model at Phase 2, before overfitting begins.

### Visual Representation

```
Loss
  │
  │ ╲
  │  ╲  Training Loss
  │   ╲
  │    ╲_______________
  │     ╲
  │      ╲
  │       ╲____________
  │
  │  ╲
  │   ╲   Validation Loss
  │    ╲
  │     ╲____●________    ← Stop here! (minimum val loss)
  │          ╱
  │         ╱
  │        ╱
  │       ╱
  └─────────────────────► Epochs
         ↑
    Best checkpoint
```

---

## How Early Stopping Works

### The Algorithm

```
1. Split data into training and validation sets
2. Train for one epoch
3. Evaluate on validation set
4. If validation loss improved:
   - Save model weights (checkpoint)
   - Reset patience counter
5. If validation loss didn't improve:
   - Increment patience counter
6. If patience counter exceeds threshold:
   - Stop training
   - Restore best weights
7. Otherwise, go to step 2
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **patience** | Epochs to wait for improvement | 5-20 |
| **min_delta** | Minimum change to qualify as improvement | 0.0001-0.001 |
| **restore_best** | Whether to restore best weights | True |

---

## PyTorch Implementation

### Basic Implementation

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss, model):
        """
        Call after each epoch with validation loss.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                # Deep copy of model weights
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def restore_best_weights(self, model):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
    
    def get_best_loss(self):
        """Return the best validation loss seen."""
        return self.best_loss
```

### Usage in Training Loop

```python
# Initialize
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Training loop
max_epochs = 100
for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    # Early stopping check
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch+1}")
        early_stopping.restore_best_weights(model)
        break

print(f"Best validation loss: {early_stopping.get_best_loss():.4f}")
```

### Enhanced Implementation with More Features

```python
class EarlyStoppingAdvanced:
    def __init__(
        self,
        patience=10,
        min_delta=0.0,
        mode='min',
        baseline=None,
        restore_best=True,
        verbose=True
    ):
        """
        Advanced early stopping with more options.
        
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            baseline: Baseline value to compare against
            restore_best: Whether to restore best weights
            verbose: Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False
        self.best_epoch = 0
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:  # mode == 'max'
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score, model, epoch=None):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, epoch)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model, epoch)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best score: {self.best_score:.4f}")
        
        return self.should_stop
    
    def _save_checkpoint(self, model, epoch):
        if self.restore_best:
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        self.best_epoch = epoch if epoch is not None else self.best_epoch + 1
    
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch}")
```

---

## Choosing Patience

### Factors to Consider

| Factor | Lower Patience | Higher Patience |
|--------|---------------|-----------------|
| **Dataset size** | Small datasets | Large datasets |
| **Model complexity** | Simple models | Complex models |
| **Learning rate** | High LR | Low LR |
| **Training time** | Limited time | Plenty of time |
| **Noise in validation** | Low noise | High noise |

### Typical Values

```
Quick experiments:     patience = 3-5
Standard training:     patience = 5-10
Thorough training:     patience = 10-20
Very noisy validation: patience = 15-30
```

### Adaptive Patience

Some practitioners increase patience as training progresses:

```python
class AdaptiveEarlyStopping(EarlyStopping):
    def __init__(self, initial_patience=5, patience_increase=2, max_patience=20, **kwargs):
        super().__init__(patience=initial_patience, **kwargs)
        self.initial_patience = initial_patience
        self.patience_increase = patience_increase
        self.max_patience = max_patience
    
    def __call__(self, val_loss, model):
        result = super().__call__(val_loss, model)
        
        # Increase patience when improvement is found
        if val_loss < self.best_loss - self.min_delta:
            self.patience = min(self.patience + self.patience_increase, self.max_patience)
        
        return result
```

---

## Early Stopping with Learning Rate Scheduling

### Combining with ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
early_stopping = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Check early stopping
    if early_stopping(val_loss, model):
        early_stopping.restore_best_weights(model)
        break
```

### Patience Relationship

```
Scheduler patience < Early stopping patience

Example:
- Scheduler patience: 3 (reduce LR after 3 epochs without improvement)
- Early stopping patience: 10 (stop after 10 epochs without improvement)

This allows the model to try lower learning rates before stopping.
```

---

## Monitoring Multiple Metrics

### Using Validation Accuracy Instead of Loss

```python
# For classification, you might prefer accuracy
early_stopping = EarlyStoppingAdvanced(
    patience=10,
    mode='max',  # Higher is better for accuracy
    min_delta=0.001
)

for epoch in range(max_epochs):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_accuracy = evaluate_accuracy(model, val_loader)
    
    if early_stopping(val_accuracy, model, epoch):
        early_stopping.restore_best_weights(model)
        break
```

### Monitoring Multiple Metrics

```python
class MultiMetricEarlyStopping:
    def __init__(self, metrics_config, patience=10):
        """
        Args:
            metrics_config: dict of {metric_name: {'mode': 'min'/'max', 'weight': float}}
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.counter = 0
        self.best_combined_score = None
        self.best_weights = None
    
    def __call__(self, metrics, model):
        # Compute weighted combined score
        combined_score = 0
        for name, value in metrics.items():
            config = self.metrics_config[name]
            if config['mode'] == 'min':
                combined_score -= value * config['weight']
            else:
                combined_score += value * config['weight']
        
        if self.best_combined_score is None or combined_score > self.best_combined_score:
            self.best_combined_score = combined_score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# Usage
early_stopping = MultiMetricEarlyStopping({
    'val_loss': {'mode': 'min', 'weight': 0.5},
    'val_accuracy': {'mode': 'max', 'weight': 0.5}
})

metrics = {'val_loss': val_loss, 'val_accuracy': val_accuracy}
if early_stopping(metrics, model):
    # Stop training
    pass
```

---

## Saving and Loading Checkpoints

### Saving Full Training State

```python
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']
```

### Early Stopping with Checkpoints

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=10, checkpoint_path='best_model.pt'):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save checkpoint to disk
            save_checkpoint(model, optimizer, epoch, val_loss, self.checkpoint_path)
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def load_best(self, model, optimizer):
        return load_checkpoint(model, optimizer, self.checkpoint_path)
```

---

## Common Patterns and Best Practices

### Pattern 1: Standard Training Loop

```python
def train_with_early_stopping(model, train_loader, val_loader, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = sum(
            criterion(model(x), y).item() 
            for x, y in train_loader
        ) / len(train_loader)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(x), y).item() 
                for x, y in val_loader
            ) / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            early_stopping.restore_best_weights(model)
            break
    
    return history
```

### Pattern 2: With Validation Frequency

For large datasets, validate less frequently:

```python
validate_every = 5  # Validate every 5 epochs

for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer, criterion)
    
    if (epoch + 1) % validate_every == 0:
        val_loss = validate(model, val_loader, criterion)
        if early_stopping(val_loss, model):
            break
```

---

## Advantages and Limitations

### Advantages

1. **Simple to implement** — No hyperparameters to tune (except patience)
2. **Computationally free** — No additional computation during training
3. **Works with any model** — Architecture agnostic
4. **Prevents wasted computation** — Stops when no longer improving
5. **Automatic regularization** — Finds optimal training duration

### Limitations

1. **Requires validation set** — Reduces training data
2. **Noisy validation** — May stop too early or too late
3. **Not differentiable** — Can't be optimized end-to-end
4. **Patience tuning** — Wrong patience can hurt performance

### When Early Stopping Might Not Help

- Very small datasets (validation set too noisy)
- When you need every training sample
- When validation metric is very noisy
- When other regularization is sufficient

---

## Key Takeaways

1. **Early stopping prevents overfitting** by stopping training at the right time
2. **Monitor validation loss** (or accuracy) to detect overfitting
3. **Patience controls sensitivity** — higher patience is more conservative
4. **Always restore best weights** after stopping
5. **Combine with LR scheduling** for best results
6. **Save checkpoints** for long training runs
7. **Simple but effective** — should be part of every training pipeline

---

*Early stopping is like knowing when to stop studying for an exam. More studying doesn't always mean better results—there's an optimal point where you've learned the material without overthinking it.*
