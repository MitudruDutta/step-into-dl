# 📊 Batch Normalization

Batch Normalization (BatchNorm) is a technique that normalizes the inputs to each layer, dramatically improving training stability and speed. While its primary purpose isn't regularization, it has a significant regularizing effect as a side benefit.

---

## The Problem BatchNorm Solves

### Internal Covariate Shift

As training progresses, the distribution of inputs to each layer changes because the previous layers' weights are updated. This is called "internal covariate shift."

```
Epoch 1: Layer 2 receives inputs with mean=0.5, std=1.2
Epoch 10: Layer 2 receives inputs with mean=2.1, std=3.5
Epoch 50: Layer 2 receives inputs with mean=-0.8, std=0.3

The layer must constantly adapt to changing input distributions.
```

### Consequences

- Training is slow (must use small learning rates)
- Gradients can vanish or explode
- Careful weight initialization is critical
- Deep networks are hard to train

---

## How Batch Normalization Works

### The Algorithm

For each mini-batch, BatchNorm:

1. **Compute batch statistics:**
   ```
   μ = (1/m) × Σxᵢ           # Batch mean
   σ² = (1/m) × Σ(xᵢ - μ)²   # Batch variance
   ```

2. **Normalize:**
   ```
   x̂ᵢ = (xᵢ - μ) / √(σ² + ε)  # Zero mean, unit variance
   ```

3. **Scale and shift (learnable):**
   ```
   yᵢ = γ × x̂ᵢ + β           # Restore representational power
   ```

### The Complete Formula

```
BatchNorm(x) = γ × (x - μ_batch) / √(σ²_batch + ε) + β

Where:
- μ_batch = mean of current mini-batch
- σ²_batch = variance of current mini-batch
- γ = learnable scale parameter (initialized to 1)
- β = learnable shift parameter (initialized to 0)
- ε = small constant for numerical stability (typically 1e-5)
```

### Why γ and β?

Without γ and β, the network would be constrained to only learn functions of normalized inputs. The learnable parameters allow the network to:
- Undo the normalization if needed
- Learn the optimal scale and shift for each layer

```
If γ = σ and β = μ, then BatchNorm(x) = x (identity)
The network can learn to bypass normalization if beneficial.
```

---

## Training vs Inference

### During Training

BatchNorm uses the current mini-batch's statistics:

```python
# Training mode
μ = batch.mean()
σ² = batch.var()
output = γ × (input - μ) / √(σ² + ε) + β
```

It also maintains running averages for inference:

```python
running_mean = momentum × running_mean + (1 - momentum) × μ
running_var = momentum × running_var + (1 - momentum) × σ²
```

### During Inference

BatchNorm uses the accumulated running statistics:

```python
# Eval mode
output = γ × (input - running_mean) / √(running_var + ε) + β
```

This ensures consistent outputs regardless of batch size or composition.

### Critical: Mode Switching

```python
model.train()  # Use batch statistics
model.eval()   # Use running statistics

# Common mistake: forgetting to switch modes!
```

---

## Why BatchNorm Has a Regularizing Effect

### 1. Noise from Batch Statistics

Each sample is normalized using statistics from its mini-batch:

```
Sample in Batch A: normalized with μ_A, σ_A
Same sample in Batch B: normalized with μ_B, σ_B

Different batches → different normalizations → noise
```

This noise acts like a form of regularization, similar to dropout.

### 2. Reduced Sensitivity to Initialization

Without BatchNorm, poor initialization can cause:
- Activations to saturate
- Gradients to vanish
- Model to memorize instead of generalize

BatchNorm keeps activations in a reasonable range, reducing these issues.

### 3. Allows Higher Learning Rates

Higher learning rates provide implicit regularization through:
- Larger gradient noise
- Faster escape from sharp minima
- Better generalization

---

## PyTorch Implementation

### For Fully Connected Layers

```python
import torch.nn as nn

class FCNetworkWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),  # BatchNorm1d for FC layers
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)
```

### For Convolutional Layers

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm2d for conv layers
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

### BatchNorm Parameters

```python
nn.BatchNorm1d(
    num_features,           # Number of features (channels)
    eps=1e-5,              # Numerical stability
    momentum=0.1,          # Running stats momentum
    affine=True,           # Learn γ and β
    track_running_stats=True  # Maintain running mean/var
)
```

---

## BatchNorm Placement

### Common Patterns

**Pattern 1: After Linear/Conv, Before Activation (Most Common)**

```python
nn.Linear(in_features, out_features),
nn.BatchNorm1d(out_features),
nn.ReLU(),
```

**Pattern 2: After Activation (Less Common)**

```python
nn.Linear(in_features, out_features),
nn.ReLU(),
nn.BatchNorm1d(out_features),
```

### Which is Better?

Research shows both work well. Pattern 1 is more common because:
- Normalizes before non-linearity
- Keeps activations in optimal range for ReLU
- Original paper used this order

### Complete Block Examples

```python
# FC Block with BatchNorm
def fc_block(in_features, out_features, dropout=0.0):
    layers = [
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

# Conv Block with BatchNorm
def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
```

---

## BatchNorm Variants

### Layer Normalization

Normalizes across features instead of batch. Preferred for:
- RNNs (variable sequence lengths)
- Transformers
- Small batch sizes

```python
# LayerNorm normalizes across the feature dimension
nn.LayerNorm(normalized_shape)

# Example for Transformer
nn.LayerNorm(hidden_size)  # Normalizes each token independently
```

**Comparison:**
```
BatchNorm: normalize across batch dimension
  Input shape: [batch, features]
  Normalize over: batch dimension (for each feature)

LayerNorm: normalize across feature dimension
  Input shape: [batch, features]
  Normalize over: feature dimension (for each sample)
```

### Instance Normalization

Normalizes each sample independently. Used in:
- Style transfer
- Image generation

```python
nn.InstanceNorm2d(num_features)
```

### Group Normalization

Divides channels into groups and normalizes within each group:

```python
nn.GroupNorm(num_groups, num_channels)

# Example: 32 groups for 256 channels
nn.GroupNorm(32, 256)  # Each group has 8 channels
```

**When to use:**
- Small batch sizes (where BatchNorm fails)
- Detection/segmentation tasks

### Comparison Table

| Normalization | Normalizes Over | Best For |
|---------------|-----------------|----------|
| **BatchNorm** | Batch dimension | CNNs, large batches |
| **LayerNorm** | Feature dimension | Transformers, RNNs |
| **InstanceNorm** | Each sample | Style transfer |
| **GroupNorm** | Channel groups | Small batches, detection |

---

## BatchNorm with Small Batches

### The Problem

BatchNorm estimates population statistics from mini-batch statistics. With small batches:
- Estimates are noisy
- Training becomes unstable
- Performance degrades

```
Batch size 256: Good estimates, stable training
Batch size 32:  Okay estimates, some noise
Batch size 4:   Poor estimates, unstable training
Batch size 1:   Undefined (can't compute batch statistics)
```

### Solutions

**1. Use Group Normalization:**
```python
# Replace BatchNorm with GroupNorm for small batches
nn.GroupNorm(num_groups=32, num_channels=256)
```

**2. Use Layer Normalization:**
```python
nn.LayerNorm(normalized_shape)
```

**3. Synchronized BatchNorm (Multi-GPU):**
```python
# Compute statistics across all GPUs
nn.SyncBatchNorm(num_features)
```

**4. Increase Virtual Batch Size:**
```python
# Accumulate gradients over multiple mini-batches
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## BatchNorm and Dropout Together

### The Debate

There's ongoing discussion about combining BatchNorm and Dropout:

**Arguments against:**
- BatchNorm already provides regularization
- Dropout changes activation statistics, confusing BatchNorm
- Some research shows they interfere

**Arguments for:**
- They regularize differently
- Many successful models use both
- Empirically works in many cases

### Practical Recommendations

```python
# Option 1: Use both (common in practice)
nn.Linear(in_features, out_features),
nn.BatchNorm1d(out_features),
nn.ReLU(),
nn.Dropout(p=0.3),  # After BatchNorm and activation

# Option 2: BatchNorm only (simpler)
nn.Linear(in_features, out_features),
nn.BatchNorm1d(out_features),
nn.ReLU(),
# No dropout

# Option 3: Dropout only (when BatchNorm causes issues)
nn.Linear(in_features, out_features),
nn.ReLU(),
nn.Dropout(p=0.5),
```

**Recommendation:** Start with BatchNorm only. Add Dropout if still overfitting.

---

## Common Mistakes

### Mistake 1: Forgetting to Switch Modes

```python
# WRONG: Evaluating in training mode
model.train()
predictions = model(test_data)  # Uses batch statistics!

# CORRECT: Switch to eval mode
model.eval()
with torch.no_grad():
    predictions = model(test_data)  # Uses running statistics
```

### Mistake 2: BatchNorm with Batch Size 1

```python
# WRONG: Single sample batch
model.train()
output = model(single_sample)  # Can't compute batch statistics!

# CORRECT: Use eval mode for single samples
model.eval()
output = model(single_sample)  # Uses running statistics
```

### Mistake 3: BatchNorm Before First Layer

```python
# WRONG: Normalizing raw input
nn.BatchNorm1d(input_features),  # Usually unnecessary
nn.Linear(input_features, hidden),

# CORRECT: Normalize input data during preprocessing instead
# Or start with Linear layer
nn.Linear(input_features, hidden),
nn.BatchNorm1d(hidden),
```

### Mistake 4: Not Using Bias with BatchNorm

```python
# INEFFICIENT: Bias is redundant with BatchNorm
nn.Linear(in_features, out_features, bias=True),  # Bias learned
nn.BatchNorm1d(out_features),  # β parameter does the same thing

# EFFICIENT: Disable bias when using BatchNorm
nn.Linear(in_features, out_features, bias=False),
nn.BatchNorm1d(out_features),  # β handles the shift
```

---

## Key Takeaways

1. **BatchNorm normalizes layer inputs** to zero mean and unit variance
2. **Learnable γ and β** restore representational power
3. **Training uses batch statistics**, inference uses running statistics
4. **Always switch to `model.eval()`** before inference
5. **Place BatchNorm after Linear/Conv**, before activation
6. **Use LayerNorm or GroupNorm** for small batches or RNNs
7. **BatchNorm provides implicit regularization** through batch noise
8. **Disable bias** in layers followed by BatchNorm

---

*Batch Normalization revolutionized deep learning by making deep networks much easier to train. It's now a standard component in most architectures.*
