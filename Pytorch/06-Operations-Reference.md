# 📋 Common Tensor Operations Reference

A quick reference guide for the most commonly used PyTorch tensor operations.

---

## Arithmetic Operations

### Element-wise Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `a + b` or `torch.add(a, b)` | Add corresponding elements |
| Subtraction | `a - b` or `torch.sub(a, b)` | Subtract corresponding elements |
| Multiplication | `a * b` or `torch.mul(a, b)` | Multiply corresponding elements |
| Division | `a / b` or `torch.div(a, b)` | Divide corresponding elements |
| Power | `a ** n` or `torch.pow(a, n)` | Raise to power |
| Square root | `torch.sqrt(a)` | Element-wise square root |
| Absolute | `torch.abs(a)` | Element-wise absolute value |
| Negation | `-a` or `torch.neg(a)` | Negate all elements |

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)      # tensor([5, 7, 9])
print(a * b)      # tensor([4, 10, 18])
print(a ** 2)     # tensor([1, 4, 9])
```

### Matrix Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Matrix multiply | `a @ b` or `torch.matmul(a, b)` | Matrix multiplication |
| Batch matmul | `torch.bmm(a, b)` | Batched matrix multiply |
| Dot product | `torch.dot(a, b)` | Dot product (1D only) |
| Outer product | `torch.outer(a, b)` | Outer product |

```python
A = torch.rand(2, 3)
B = torch.rand(3, 4)

C = A @ B           # Shape: [2, 4]
C = torch.matmul(A, B)  # Same result
```

---

## Reduction Operations

Operations that reduce dimensions:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Sum | `x.sum()` or `x.sum(dim=0)` | Sum elements |
| Mean | `x.mean()` or `x.mean(dim=0)` | Average elements |
| Product | `x.prod()` | Product of elements |
| Max | `x.max()` or `x.max(dim=0)` | Maximum value |
| Min | `x.min()` or `x.min(dim=0)` | Minimum value |
| Argmax | `x.argmax()` | Index of maximum |
| Argmin | `x.argmin()` | Index of minimum |
| Std | `x.std()` | Standard deviation |
| Var | `x.var()` | Variance |

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.sum())        # tensor(21) - all elements
print(x.sum(dim=0))   # tensor([5, 7, 9]) - sum columns
print(x.sum(dim=1))   # tensor([6, 15]) - sum rows
print(x.mean())       # tensor(3.5)
print(x.max())        # tensor(6)
print(x.argmax())     # tensor(5) - flattened index
```

---

## Shape Operations

### Reshaping

| Operation | Syntax | Description |
|-----------|--------|-------------|
| View | `x.view(2, 3)` | Reshape (must be contiguous) |
| Reshape | `x.reshape(2, 3)` | Reshape (handles non-contiguous) |
| Flatten | `x.flatten()` | Convert to 1D |
| Squeeze | `x.squeeze()` | Remove dims of size 1 |
| Unsqueeze | `x.unsqueeze(0)` | Add dim at position |

```python
x = torch.rand(2, 3, 4)  # 24 elements

y = x.view(6, 4)         # [6, 4]
y = x.view(-1)           # [24] - flatten
y = x.view(2, -1)        # [2, 12] - infer second dim
y = x.flatten()          # [24]
```

### Transpose & Permute

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Transpose | `x.T` | Transpose 2D tensor |
| Transpose | `x.transpose(0, 1)` | Swap two dimensions |
| Permute | `x.permute(2, 0, 1)` | Reorder all dimensions |

```python
x = torch.rand(2, 3, 4)

y = x.transpose(0, 2)    # [4, 3, 2]
y = x.permute(2, 0, 1)   # [4, 2, 3]
```

### Combining Tensors

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Concatenate | `torch.cat([a, b], dim=0)` | Join along existing dim |
| Stack | `torch.stack([a, b], dim=0)` | Join along new dim |
| Split | `torch.split(x, 2, dim=0)` | Split into chunks |
| Chunk | `torch.chunk(x, 3, dim=0)` | Split into n chunks |

```python
a = torch.rand(2, 3)
b = torch.rand(2, 3)

c = torch.cat([a, b], dim=0)   # [4, 3]
c = torch.cat([a, b], dim=1)   # [2, 6]
c = torch.stack([a, b], dim=0) # [2, 2, 3]
```

---

## Comparison Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Equal | `a == b` or `torch.eq(a, b)` | Element-wise equality |
| Not equal | `a != b` or `torch.ne(a, b)` | Element-wise inequality |
| Greater | `a > b` or `torch.gt(a, b)` | Element-wise greater than |
| Less | `a < b` or `torch.lt(a, b)` | Element-wise less than |
| Greater/equal | `a >= b` or `torch.ge(a, b)` | Greater than or equal |
| Less/equal | `a <= b` or `torch.le(a, b)` | Less than or equal |

```python
x = torch.tensor([1, 2, 3, 4, 5])

mask = x > 3           # tensor([False, False, False, True, True])
filtered = x[mask]     # tensor([4, 5])
```

---

## Mathematical Functions

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Exponential | `torch.exp(x)` | e^x |
| Logarithm | `torch.log(x)` | Natural log |
| Log base 10 | `torch.log10(x)` | Log base 10 |
| Sine | `torch.sin(x)` | Sine |
| Cosine | `torch.cos(x)` | Cosine |
| Tanh | `torch.tanh(x)` | Hyperbolic tangent |
| Sigmoid | `torch.sigmoid(x)` | Sigmoid function |
| ReLU | `torch.relu(x)` | ReLU activation |
| Softmax | `torch.softmax(x, dim=0)` | Softmax |
| Clamp | `torch.clamp(x, min, max)` | Clip values |

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(torch.relu(x))      # tensor([0., 0., 0., 1., 2.])
print(torch.sigmoid(x))   # tensor([0.12, 0.27, 0.50, 0.73, 0.88])
print(torch.clamp(x, -1, 1))  # tensor([-1., -1., 0., 1., 1.])
```

---

## Linear Algebra

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Matrix inverse | `torch.inverse(A)` | Inverse of square matrix |
| Determinant | `torch.det(A)` | Determinant |
| Eigenvalues | `torch.linalg.eig(A)` | Eigenvalues and eigenvectors |
| SVD | `torch.linalg.svd(A)` | Singular value decomposition |
| Norm | `torch.norm(x)` | Vector/matrix norm |
| Solve | `torch.linalg.solve(A, b)` | Solve Ax = b |

```python
A = torch.rand(3, 3)

det = torch.det(A)
inv = torch.inverse(A)
norm = torch.norm(A)
```

---

## In-Place Operations

Operations ending with `_` modify tensors in-place:

```python
x = torch.tensor([1.0, 2.0, 3.0])

x.add_(1)      # x is now [2, 3, 4]
x.mul_(2)      # x is now [4, 6, 8]
x.zero_()      # x is now [0, 0, 0]
x.fill_(5)     # x is now [5, 5, 5]
```

**⚠️ Warning:** Avoid in-place operations on tensors that require gradients—they can cause autograd errors.

---

## Device Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Move to device | `x.to(device)` | Move tensor to CPU/GPU |
| Move to CPU | `x.cpu()` | Move to CPU |
| Move to GPU | `x.cuda()` | Move to GPU |
| Check device | `x.device` | Get current device |

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.rand(3, 3)
x = x.to(device)
print(x.device)  # cuda:0 or cpu
```

---

## Type Conversion

| Operation | Syntax | Description |
|-----------|--------|-------------|
| To float | `x.float()` | Convert to float32 |
| To double | `x.double()` | Convert to float64 |
| To int | `x.int()` | Convert to int32 |
| To long | `x.long()` | Convert to int64 |
| To bool | `x.bool()` | Convert to boolean |
| To specific | `x.to(torch.float16)` | Convert to specific dtype |

```python
x = torch.tensor([1, 2, 3])  # int64 by default

x_float = x.float()    # float32
x_double = x.double()  # float64
```

---

*Keep this reference handy when working with PyTorch tensors. Most operations follow intuitive naming conventions similar to NumPy.*
