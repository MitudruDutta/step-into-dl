# 🚀 Getting Started with PyTorch

This module covers the mathematical foundations and PyTorch fundamentals needed for deep learning—matrices, tensors, calculus, and automatic differentiation.

---

## 📚 Documentation

| File                                                          | Topic            | Description                                                |
| ------------------------------------------------------------- | ---------------- | ---------------------------------------------------------- |
| [01-Matrix-Fundamentals.md](docs/01-Matrix-Fundamentals.md)   | Matrices         | Why matrices matter, element-wise vs matrix multiplication |
| [02-Tensor-Basics.md](docs/02-Tensor-Basics.md)               | Tensors          | Dimensions, creation, attributes, indexing                 |
| [03-Calculus-Basics.md](docs/03-Calculus-Basics.md)           | Calculus         | Derivatives, partial derivatives, chain rule               |
| [04-Autograd.md](docs/04-Autograd.md)                         | Autograd         | Automatic differentiation, computation graphs              |
| [05-Tensors-vs-NumPy.md](docs/05-Tensors-vs-NumPy.md)         | NumPy Comparison | When to use each, GPU acceleration                         |
| [06-Operations-Reference.md](docs/06-Operations-Reference.md) | Operations       | Quick reference for common tensor operations               |
| [07-Best-Practices.md](docs/07-Best-Practices.md)             | Best Practices   | Memory, devices, training loops, debugging                 |

---

## 💻 Notebooks

| Notebook                                   | Description                                             |
| ------------------------------------------ | ------------------------------------------------------- |
| [tensor1.ipynb](notebooks/tensor1.ipynb)   | Tensor creation, arithmetic, matrix multiplication, GPU |
| [tensor2.ipynb](notebooks/tensor2.ipynb)   | Tensor attributes, reshaping, initialization            |
| [autograd.ipynb](notebooks/autograd.ipynb) | Gradient tracking, backward(), chain rule               |

---

## 🎯 Learning Path

1. **Matrix Fundamentals** → Understand why matrices are the language of AI
2. **Tensor Basics** → Learn to create and manipulate tensors
3. **Practice with tensor1.ipynb** → Hands-on tensor operations
4. **Calculus Basics** → Understand derivatives and the chain rule
5. **Autograd** → Learn automatic differentiation
6. **Practice with autograd.ipynb** → Hands-on gradient computation
7. **Operations Reference** → Bookmark for quick lookups
8. **Best Practices** → Write efficient, bug-free code

---

## 🔑 Key Concepts

### Tensor Dimensions

```
0D: Scalar      → 5
1D: Vector      → [1, 2, 3]
2D: Matrix      → [[1, 2], [3, 4]]
3D: 3D Tensor   → Images (H×W×C)
4D: 4D Tensor   → Batches (N×C×H×W)
```

### Essential Attributes

| Attribute | Description | Example              |
| --------- | ----------- | -------------------- |
| `.shape`  | Dimensions  | `torch.Size([3, 4])` |
| `.dtype`  | Data type   | `torch.float32`      |
| `.device` | Location    | `cpu` or `cuda:0`    |

### Autograd Basics

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])
```

---

## ⚡ Quick Reference

### Create Tensors

```python
torch.tensor([1, 2, 3])      # From data
torch.zeros(3, 4)            # Zeros
torch.ones(3, 4)             # Ones
torch.rand(3, 4)             # Random [0, 1)
torch.randn(3, 4)            # Random normal
```

### Move to GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
model = model.to(device)
```

### Disable Gradients

```python
with torch.no_grad():
    predictions = model(inputs)
```

---

## 📖 Prerequisites

Before this module, you should understand:

- Basic Python programming
- NumPy basics (helpful but not required)
- High school algebra

---

_Master these PyTorch fundamentals before moving on to building neural network architectures._
