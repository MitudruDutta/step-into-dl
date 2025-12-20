# ⚡ PyTorch Autograd: Automatic Differentiation

In the past, researchers had to calculate derivatives by hand—tedious and error-prone. PyTorch's **Autograd** feature automates this entire process, making deep learning practical.

---

## What is Autograd?

Autograd is PyTorch's automatic differentiation engine. It:
1. Tracks all operations on tensors
2. Builds a computation graph
3. Computes gradients automatically via backpropagation

This means you write the forward pass, and PyTorch figures out the backward pass for you.

---

## How Autograd Works

### Step 1: Enable Gradient Tracking

```python
import torch

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
```

### Step 2: Perform Operations

PyTorch builds a computation graph as you compute:

```python
y = x ** 2 + 3 * x  # y = x² + 3x
z = y.sum()         # Reduce to scalar for backward()
```

### Step 3: Compute Gradients

```python
z.backward()  # Compute dz/dx
```

### Step 4: Access Gradients

```python
print(x.grad)  # tensor([7., 9.])
```

**Verification:**
- dz/dx = d(x² + 3x)/dx = 2x + 3
- At x=2: 2(2) + 3 = 7 ✓
- At x=3: 2(3) + 3 = 9 ✓

---

## The Computation Graph

When you perform operations, PyTorch builds a directed acyclic graph (DAG):

```
x (leaf tensor, requires_grad=True)
│
├── x ** 2 ──┐
│            ├── + ── y
└── 3 * x ───┘
             │
             └── sum() ── z
```

**Text description:** The computation graph shows x as the input (leaf node), which branches into two operations: x² and 3×x. These are added together to form y, which is then summed to produce the scalar z. Gradients flow backwards from z to x.

### Key Terms

| Term | Description |
|------|-------------|
| **Leaf tensor** | Input tensors you create (like weights) |
| **Non-leaf tensor** | Results of operations |
| **grad_fn** | Function that created the tensor (for backprop) |

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2

print(x.is_leaf)    # True
print(y.is_leaf)    # False
print(y.grad_fn)    # <MulBackward0 object>
```

---

## Gradient Computation Rules

### 1. Only Scalars Can Call `.backward()` Directly

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2  # Shape: [2]

# This won't work:
# y.backward()  # Error: grad can be implicitly created only for scalar outputs

# Solution 1: Reduce to scalar
y.sum().backward()

# Solution 2: Provide gradient argument
y.backward(torch.ones_like(y))
```

### 2. Gradients Accumulate by Default

```python
x = torch.tensor([1.0], requires_grad=True)

# First backward
y1 = x * 2
y1.backward()
print(x.grad)  # tensor([2.])

# Second backward - gradients ADD to existing!
y2 = x * 3
y2.backward()
print(x.grad)  # tensor([5.])  # 2 + 3 = 5

# Always zero gradients in training loops!
x.grad.zero_()
```

### 3. Graph is Destroyed After `.backward()`

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()

# This will fail:
# y.backward()  # Error: graph already freed

# Solution: retain_graph=True
y = x ** 2
y.backward(retain_graph=True)
y.backward()  # Now works
```

---

## Disabling Gradient Tracking

### `torch.no_grad()` Context Manager

Use during inference/evaluation to save memory and speed up computation:

```python
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    predictions = model(test_data)
    # No computation graph built
    # No gradients computed
    # Faster and uses less memory
```

### `.detach()` Method

Remove a tensor from the computation graph:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# Detach y from graph
y_detached = y.detach()

print(y.requires_grad)          # True
print(y_detached.requires_grad) # False
```

**Use case:** When you need tensor values but don't want gradients to flow through.

### `requires_grad_(False)`

Permanently disable gradient tracking:

```python
x = torch.tensor([1.0], requires_grad=True)
x.requires_grad_(False)
print(x.requires_grad)  # False
```

---

## Practical Training Loop

```python
import torch
import torch.nn as nn

# Model and data
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

inputs = torch.randn(32, 10)
targets = torch.randn(32, 1)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients (IMPORTANT!)
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Gradient Checking

Verify your gradients are correct:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³, dy/dx = 3x² = 12 at x=2

y.backward()
print(f"Autograd gradient: {x.grad.item()}")  # 12.0

# Numerical gradient (finite differences)
eps = 1e-5
numerical_grad = ((2.0 + eps)**3 - (2.0 - eps)**3) / (2 * eps)
print(f"Numerical gradient: {numerical_grad:.4f}")  # ~12.0
```

---

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| `grad is None` | Tensor doesn't require grad | Set `requires_grad=True` |
| Gradients accumulating | Forgot to zero gradients | Call `optimizer.zero_grad()` |
| `RuntimeError: element 0 of tensors does not require grad` | Leaf tensor modified in-place | Avoid in-place ops on leaf tensors |
| Graph already freed | Called backward twice | Use `retain_graph=True` |
| Out of memory | Graph too large | Use `torch.no_grad()` for inference |

---

## Key Functions Reference

| Function | Purpose |
|----------|---------|
| `requires_grad=True` | Enable gradient tracking |
| `.backward()` | Compute gradients |
| `.grad` | Access computed gradients |
| `.grad.zero_()` | Clear gradients |
| `torch.no_grad()` | Disable tracking (context) |
| `.detach()` | Remove from graph |
| `.retain_grad()` | Keep grad for non-leaf tensors |

---

*Autograd transforms the tedious manual calculus of backpropagation into a single `.backward()` call. Understanding it is essential for debugging training issues.*
