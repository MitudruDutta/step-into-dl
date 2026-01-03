# 📦 PyTorch Tensor Basics: The Core Data Structure

A **Tensor** is the fundamental object in PyTorch. While it sounds complex, it is simply a generic term for any multi-dimensional array—the building block for all data in deep learning.

---

## Understanding Dimensions

Tensors can have any number of dimensions:

| Dimensions | Name | Example | Common Use Case |
|------------|------|---------|-----------------|
| **0D** | Scalar | `5` | Loss value, single prediction |
| **1D** | Vector | `[1, 2, 3]` | Bias terms, single sample features |
| **2D** | Matrix | `[[1,2], [3,4]]` | Batch of samples, weight matrices |
| **3D** | 3D Tensor | `[H, W, C]` | Single RGB image, sequence data |
| **4D** | 4D Tensor | `[N, C, H, W]` | Mini-batch of images |
| **5D** | 5D Tensor | `[N, C, D, H, W]` | Video data (batch of frame sequences) |

### Visual Representation

```
0D (Scalar):     5

1D (Vector):     [1, 2, 3, 4, 5]

2D (Matrix):     [[1, 2, 3],
                  [4, 5, 6]]

3D (Cube):       [[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]]
```

---

## Creating Tensors

### From Python Data

```python
import torch

# From a single value (scalar)
scalar = torch.tensor(5)

# From a list (vector)
vector = torch.tensor([1, 2, 3, 4])

# From nested lists (matrix)
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# From NumPy array
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
```

**⚠️ Memory Sharing:** `torch.from_numpy()` returns a tensor that **shares memory** with the original NumPy array. Modifying one will affect the other:

```python
np_array[0] = 100
print(tensor)  # tensor([100, 2, 3]) - also changed!
```

To create an independent copy instead:
- Use `torch.tensor(np_array)` — always creates a copy
- Or call `.clone()`: `torch.from_numpy(np_array).clone()`

### Initialized Tensors

```python
# Zeros
zeros = torch.zeros(3, 4)        # 3×4 matrix of zeros

# Ones
ones = torch.ones(2, 3)          # 2×3 matrix of ones

# Random uniform [0, 1)
rand = torch.rand(5, 5)          # 5×5 random values

# Random normal (mean=0, std=1)
randn = torch.randn(3, 3)        # Can have negative values

# Specific value
full = torch.full((2, 3), 7.0)   # 2×3 matrix filled with 7.0

# Identity matrix
eye = torch.eye(4)               # 4×4 identity matrix

# Range of values
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### Like Existing Tensors

```python
template = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Same shape, dtype, device as template
zeros_like = torch.zeros_like(template)
ones_like = torch.ones_like(template)
rand_like = torch.rand_like(template)
```

---

## Tensor Attributes

Every tensor has three critical properties:

### 1. Shape (`.shape` or `.size()`)

The dimensions of the tensor.

```python
x = torch.rand(2, 3, 4)
print(x.shape)      # torch.Size([2, 3, 4])
print(x.size())     # torch.Size([2, 3, 4])
print(x.size(0))    # 2 (first dimension)
print(len(x))       # 2 (first dimension)
print(x.numel())    # 24 (total elements: 2×3×4)
```

### 2. Data Type (`.dtype`)

The type of elements stored.

| dtype | Description | Use Case |
|-------|-------------|----------|
| `torch.float32` | 32-bit float (default) | Most training |
| `torch.float64` | 64-bit float | High precision |
| `torch.float16` | 16-bit float | Mixed precision training |
| `torch.int64` | 64-bit integer | Indices, labels |
| `torch.int32` | 32-bit integer | Smaller indices |
| `torch.bool` | Boolean | Masks |

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)  # torch.int64

# Convert dtype
x_float = x.float()           # to float32
x_double = x.double()         # to float64
x_int = x.int()               # to int32
x_custom = x.to(torch.float16)  # to specific type
```

### 3. Device (`.device`)

Where the tensor lives (CPU or GPU).

```python
x = torch.rand(3, 3)
print(x.device)  # cpu

# Move to GPU
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(x_gpu.device)  # cuda:0
    
    # Or create directly on GPU
    y = torch.rand(3, 3, device='cuda')
```

---

## Indexing and Slicing

Tensors support NumPy-style indexing:

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Single element
x[0, 0]          # tensor(1)

# Row
x[0]             # tensor([1, 2, 3])
x[0, :]          # tensor([1, 2, 3])

# Column
x[:, 0]          # tensor([1, 4, 7])

# Slice
x[0:2, 1:3]      # tensor([[2, 3], [5, 6]])

# Negative indexing
x[-1]            # tensor([7, 8, 9]) - last row
x[:, -1]         # tensor([3, 6, 9]) - last column

# Boolean indexing
mask = x > 5
x[mask]          # tensor([6, 7, 8, 9])
```

---

## Modifying Tensors

```python
x = torch.tensor([[1, 2], [3, 4]])

# Modify single element
x[0, 0] = 10

# Modify row
x[1] = torch.tensor([30, 40])

# Modify with condition
x[x > 5] = 0

# In-place operations (end with _)
x.add_(1)        # Add 1 to all elements in-place
x.mul_(2)        # Multiply all elements by 2 in-place
x.zero_()        # Set all elements to 0
```

**⚠️ Warning:** In-place operations can cause issues with autograd. Avoid them during training when gradients are needed.

---

## Common Operations Quick Reference

| Operation | Code | Description |
|-----------|------|-------------|
| Shape | `x.shape` | Get dimensions |
| Reshape | `x.view(2, 3)` | Change shape |
| Flatten | `x.flatten()` | Convert to 1D |
| Transpose | `x.T` | Swap dimensions |
| Concatenate | `torch.cat([a, b], dim=0)` | Join along dimension |
| Stack | `torch.stack([a, b])` | Create new dimension |
| Split | `torch.split(x, 2)` | Split into chunks |

---

*Tensors are the foundation of PyTorch. Master these basics before moving on to neural network operations.*
