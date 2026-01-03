# 🏗️ Building Models with `nn.Module`

In PyTorch, **`nn.Module`** is the absolute foundation for all neural network development. It acts as the base class for every layer and every model you build.

---

## What is nn.Module?

`nn.Module` is a PyTorch class that provides:
- A container for neural network layers
- Automatic parameter tracking
- Methods for training/evaluation modes
- Serialization (save/load) capabilities
- GPU/CPU device management

Every neural network you build in PyTorch should inherit from `nn.Module`.

---

## How to Define a Model

### Step 1: Subclassing

The standard practice is to create a custom Python class that inherits from `nn.Module`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # Always call parent's __init__
        # Define layers here
```

This gives your model access to all of PyTorch's neural network functionality.

### Step 2: Define Layers in `__init__`

All layers and learnable parameters must be defined in the `__init__` method:

```python
def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.activation = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, output_size)
```

### Step 3: Implement the `forward` Method

The `forward` method specifies exactly how data flows from input to output:

```python
def forward(self, x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    return x
```

This is the blueprint for your network's computation.

---

## Complete Model Example

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Create model instance
model = BinaryClassifier(input_features=10)
```

---

## Implicit Calling

When you run `model(train_data)`, PyTorch internally executes your `forward` logic:

```python
# These are equivalent:
output = model(input_data)      # ✅ Correct way
output = model.forward(input_data)  # ❌ Don't do this directly
```

**Why not call `forward()` directly?**
- Calling the model triggers hooks (for debugging, profiling)
- Ensures proper gradient tracking
- Handles training/eval mode correctly

---

## Key Components Reference

| Component | Purpose | Example |
|-----------|---------|---------|
| `__init__` | Define all layers and learnable parameters | `self.fc = nn.Linear(10, 5)` |
| `forward` | Specify how input transforms to output | `return self.fc(x)` |
| `nn.Linear` | Fully connected (dense) layer | `nn.Linear(in_features, out_features)` |
| `nn.Conv2d` | 2D convolutional layer | `nn.Conv2d(in_channels, out_channels, kernel_size)` |
| `nn.ReLU` | ReLU activation function | `nn.ReLU()` |
| `nn.Sigmoid` | Sigmoid activation | `nn.Sigmoid()` |
| `nn.Softmax` | Softmax activation | `nn.Softmax(dim=1)` |
| `nn.Dropout` | Dropout regularization | `nn.Dropout(p=0.5)` |
| `nn.BatchNorm1d` | Batch normalization | `nn.BatchNorm1d(num_features)` |

---

## Using nn.Sequential

For simple architectures, `nn.Sequential` chains layers together:

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```

**When to use Sequential:**
- Simple feed-forward networks
- No branching or skip connections
- Quick prototyping

**When to use custom nn.Module:**
- Complex architectures (ResNet, U-Net)
- Multiple inputs or outputs
- Conditional logic in forward pass
- Skip connections or residual blocks

---

## Why nn.Module?

### 1. Automatic Parameter Tracking

All parameters defined as `nn.Module` attributes are automatically tracked:

```python
model = BinaryClassifier(10)
print(list(model.parameters()))  # Lists all learnable parameters
print(model.state_dict())        # Dictionary of parameter tensors
```

### 2. Easy GPU Support

Move entire model to GPU with one call:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 3. Save and Load Models

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = BinaryClassifier(10)
model.load_state_dict(torch.load('model.pth'))
```

### 4. Training vs Evaluation Mode

```python
model.train()  # Enable dropout, batch norm in training mode
model.eval()   # Disable dropout, use running stats for batch norm
```

**Always set the correct mode:**
- `model.train()` before training loop
- `model.eval()` before validation/inference

---

## Accessing Model Information

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print model architecture
print(model)

# List named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

---

## Common Patterns

### Pattern 1: Configurable Depth

```python
class FlexibleMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Usage
model = FlexibleMLP([784, 256, 128, 10])
```

### Pattern 2: Skip Connections

```python
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.fc2 = nn.Linear(features, features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual  # Skip connection
        return self.relu(x)
```

---

## Best Practices

1. **Always call `super().__init__()`** in your `__init__` method
2. **Define all layers in `__init__`**, not in `forward`
3. **Use `model.train()` and `model.eval()`** appropriately
4. **Don't call `forward()` directly** — call the model itself
5. **Initialize weights properly** for better convergence
6. **Use `nn.ModuleList`** for lists of layers (not Python lists)

---

*Understanding `nn.Module` is essential for building any neural network in PyTorch. Master this foundation before moving to complex architectures.*
