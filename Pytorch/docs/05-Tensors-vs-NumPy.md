# 🔄 PyTorch Tensors vs. NumPy Arrays

While PyTorch Tensors look and feel like NumPy arrays, they offer critical benefits for deep learning. Understanding when to use each is important for efficient code.

---

## Feature Comparison

| Feature | NumPy Arrays | PyTorch Tensors |
|---------|--------------|-----------------|
| **GPU Support** | ❌ CPU only | ✅ Native GPU acceleration |
| **Autograd** | ❌ No gradient tracking | ✅ Automatic differentiation |
| **DL Ecosystem** | ❌ General purpose | ✅ Built for neural networks |
| **API Similarity** | Baseline | Very similar to NumPy |
| **Maturity** | Older, more stable | Newer, rapidly evolving |
| **Community** | Huge (all of scientific Python) | Large (ML/DL focused) |

---

## When to Use Each

### Use NumPy When:
- Data preprocessing before training
- Working with non-ML libraries (pandas, scipy, matplotlib)
- CPU-only computations
- You don't need gradients
- Interfacing with legacy code

### Use PyTorch Tensors When:
- Training neural networks
- Need GPU acceleration
- Need automatic differentiation
- Building deep learning models
- Inference with trained models

---

## Seamless Conversion

PyTorch makes it easy to convert between the two:

### NumPy → Tensor

```python
import numpy as np
import torch

np_array = np.array([1, 2, 3, 4, 5])

# Method 1: from_numpy (shares memory!)
tensor1 = torch.from_numpy(np_array)

# Method 2: torch.tensor (creates copy)
tensor2 = torch.tensor(np_array)
```

### Tensor → NumPy

```python
tensor = torch.tensor([1, 2, 3, 4, 5])

# Convert to NumPy (shares memory if on CPU!)
np_array = tensor.numpy()

# For GPU tensors, must move to CPU first
gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
np_array = gpu_tensor.cpu().numpy()
```

---

## ⚠️ Memory Sharing Warning

`torch.from_numpy()` and `.numpy()` **share memory** by default:

```python
import numpy as np
import torch

# Create NumPy array
np_arr = np.array([1, 2, 3])

# Convert to tensor (shares memory)
tensor = torch.from_numpy(np_arr)

# Modify the tensor
tensor[0] = 100

# NumPy array is also modified!
print(np_arr)  # [100, 2, 3]
```

### To Avoid Memory Sharing

```python
# Option 1: Use torch.tensor() (always copies)
tensor = torch.tensor(np_arr)

# Option 2: Clone after conversion
tensor = torch.from_numpy(np_arr).clone()

# Option 3: Copy NumPy array first
tensor = torch.from_numpy(np_arr.copy())
```

---

## GPU Acceleration

The biggest advantage of PyTorch tensors is GPU support:

```python
import torch

# Check GPU availability
print(torch.cuda.is_available())  # True if GPU available
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0))  # GPU name

# Create tensor on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Method 1: Create directly on GPU
x = torch.rand(1000, 1000, device=device)

# Method 2: Move existing tensor to GPU
y = torch.rand(1000, 1000)
y = y.to(device)
# or
y = y.cuda()
```

### Performance Comparison

```python
import torch
import time

size = 10000

# CPU computation
a_cpu = torch.rand(size, size)
b_cpu = torch.rand(size, size)

start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start

# GPU computation
a_gpu = torch.rand(size, size, device='cuda')
b_gpu = torch.rand(size, size, device='cuda')

torch.cuda.synchronize()  # Ensure GPU is ready
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

Typical speedup: **10-100x** for large matrix operations.

---

## API Similarities

Most NumPy operations have PyTorch equivalents:

| NumPy | PyTorch | Description |
|-------|---------|-------------|
| `np.array([1,2,3])` | `torch.tensor([1,2,3])` | Create array/tensor |
| `np.zeros((3,4))` | `torch.zeros(3,4)` | Zeros |
| `np.ones((3,4))` | `torch.ones(3,4)` | Ones |
| `np.random.rand(3,4)` | `torch.rand(3,4)` | Random uniform |
| `np.random.randn(3,4)` | `torch.randn(3,4)` | Random normal |
| `a.reshape(2,3)` | `a.reshape(2,3)` | Reshape |
| `a.T` | `a.T` | Transpose |
| `np.concatenate` | `torch.cat` | Concatenate |
| `np.stack` | `torch.stack` | Stack |
| `a.sum()` | `a.sum()` | Sum |
| `a.mean()` | `a.mean()` | Mean |
| `np.matmul(a,b)` | `torch.matmul(a,b)` or `a @ b` | Matrix multiply |

**Note on reshape vs view:** PyTorch has both `.reshape()` and `.view()`. Use `.reshape()` for NumPy parity—it handles non-contiguous tensors automatically. `.view()` requires the tensor to be contiguous in memory and will raise an error otherwise.

---

## Working with Both

A common pattern is to use NumPy for data loading and PyTorch for training:

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load data with NumPy/Pandas
data = np.loadtxt('data.csv', delimiter=',')
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch_X, batch_y in dataloader:
    # Move to GPU if available
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    
    # Train...
```

---

## Common Gotchas

| Issue | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: expected device cuda:0 but got device cpu` | Tensors on different devices | Move all tensors to same device |
| Unexpected data modification | Memory sharing | Use `.clone()` or `torch.tensor()` |
| `TypeError: can't convert cuda:0 device type tensor to numpy` | GPU tensor to NumPy | Call `.cpu()` first |
| Slow GPU operations | Small tensors | Batch operations, use larger tensors |

---

## Best Practices

1. **Use NumPy for preprocessing**, PyTorch for training
2. **Be explicit about device** — always specify where tensors should live
3. **Avoid unnecessary conversions** — they add overhead
4. **Use `.clone()` when you need independent copies**
5. **Profile your code** — sometimes CPU is faster for small operations

---

*Understanding the relationship between NumPy and PyTorch helps you write efficient code that leverages the strengths of both libraries.*
