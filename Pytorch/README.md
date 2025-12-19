# 🚀 Getting Started with PyTorch: Matrices, Tensors & Calculus

This documentation provides an in-depth look at the mathematical engine of Deep Learning and introduces **PyTorch**, the primary tool for building modern AI. We explore why matrices are the language of AI, how Tensors handle data, and how Calculus allows models to learn automatically.

---

## 1. Matrix Fundamentals: The Language of AI

In neural networks, data isn't just a list of numbers—it is organized into **Matrices**. A matrix is essentially a table-like arrangement of numbers with rows and columns, providing a structured way to represent and manipulate data efficiently.

### 🛠️ Why Matrix Arithmetic Matters

* **Business Logic**: Matrix addition, subtraction, and multiplication are used to solve complex business problems, from recommendation systems to financial modeling.
* **Weight Processing**: In a neural network, "weights" (the model's learnable parameters) are multiplied by the output of the previous layer using matrix multiplication. This is the core computation that happens billions of times during training.
* **The GPU Advantage**: Neural networks require millions of matrix multiplications. GPUs are popular for deep learning because they use thousands of cores to compute dot products in **parallel**, making training 10-100x faster than CPUs.

### 📐 Rules of Engagement

There are two primary ways to multiply matrices:

1. **Element-wise (Hadamard Product)**: Multiplying each corresponding element in two matrices of the same size. If A and B are both 3×3 matrices, the result C[i,j] = A[i,j] × B[i,j].

2. **Matrix Multiplication (Dot Product)**: A specific mathematical operation where the **number of columns in the first matrix must equal the number of rows in the second matrix**. For matrices A (m×n) and B (n×p), the result is a matrix C (m×p).

### 🔢 Matrix Multiplication Example

```
Matrix A (2×3):        Matrix B (3×2):        Result C (2×2):
[1, 2, 3]              [7, 8]                 [58, 64]
[4, 5, 6]              [9, 10]                [139, 154]
                       [11, 12]
```

The computation: C[0,0] = (1×7) + (2×9) + (3×11) = 7 + 18 + 33 = 58

---

## 2. PyTorch Tensor Basics: The Core Data Structure

A **Tensor** is the fundamental object in PyTorch. While it sounds complex, it is simply a generic term for any multi-dimensional array—the building block for all data in deep learning.

### 📏 Understanding Dimensions

| Dimensions | Name | Example | Common Use Case |
| :--- | :--- | :--- | :--- |
| **0D** | Scalar | `5` | Loss value, single prediction |
| **1D** | Vector | `[1, 2, 3]` | Bias terms, single sample features |
| **2D** | Matrix | `[[1,2], [3,4]]` | Batch of samples, weight matrices |
| **3D** | 3D Tensor | Image (H×W×C) | Single RGB image, sequence data |
| **4D** | 4D Tensor | Batch of images | Mini-batch of images (N×C×H×W) |

### ⚙️ Working with Tensors

**Creation Methods:**
```python
import torch

# From Python lists
x = torch.tensor([1, 2, 3])

# Initialized tensors
zeros = torch.zeros(3, 4)      # 3×4 matrix of zeros
ones = torch.ones(2, 3)        # 2×3 matrix of ones
rand = torch.rand(5, 5)        # 5×5 matrix of random values [0, 1)
randn = torch.randn(3, 3)      # 3×3 matrix from normal distribution
```

**Key Attributes:**
Every tensor has three critical properties you should monitor:

| Attribute | Description | Example |
| :--- | :--- | :--- |
| `dtype` | Data type of elements | `torch.float32`, `torch.int64` |
| `shape` | Dimensions of the tensor | `torch.Size([3, 4])` |
| `device` | Where tensor lives | `cpu` or `cuda:0` |

**Essential Operations:**

* `view()` / `reshape()`: Reshape data without changing content. Critical for preparing data for different layers.
* `squeeze()` / `unsqueeze()`: Remove or add dimensions of size 1.
* `transpose()` / `permute()`: Rearrange dimensions.
* `cat()` / `stack()`: Combine multiple tensors.

### 🔄 Reshaping Example

```python
x = torch.rand(2, 3, 4)  # Shape: [2, 3, 4] - 24 elements
y = x.view(6, 4)         # Shape: [6, 4] - same 24 elements, different arrangement
z = x.view(-1)           # Shape: [24] - flatten to 1D (-1 infers the size)
```

---

## 3. Calculus: The Engine of Learning

To improve, a neural network must know how to adjust its weights. This is where **Calculus** comes in—specifically, derivatives tell us which direction to move weights to reduce error.

### 📉 Derivatives & Slopes

* **Derivative**: Measures the rate of change of a function—essentially the slope at a specific point. It answers: "If I change the input slightly, how much does the output change?"

* **Why It Matters**: During training, we want to know how changing each weight affects the loss. The derivative tells us exactly this.

* **Power Rule**: A fundamental formula: d/dx(xⁿ) = n × x^(n-1)
  - Example: d/dx(x³) = 3x²

* **Partial Derivatives**: In deep learning, we have many variables (thousands to billions of weights). A partial derivative measures how the function changes as **one** variable varies while all others remain constant.

### ⛓️ The Chain Rule

The **Chain Rule** is the mathematical technique used to calculate derivatives of composite functions—functions made up of other functions.

**Why It's Essential:**
Neural networks are essentially chains of functions: input → layer1 → layer2 → ... → output. To find how the input affects the output, we need to multiply the derivatives along the chain.

**Formula:** If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x)

**In Neural Networks:**
```
Loss = f(weights)
     = f(layer3(layer2(layer1(input))))

To find ∂Loss/∂weight1, we multiply:
∂Loss/∂layer3 × ∂layer3/∂layer2 × ∂layer2/∂layer1 × ∂layer1/∂weight1
```

This is exactly what **backpropagation** does—it applies the chain rule backwards through the network.

---

## 4. PyTorch Autograd: Automatic Calculus

In the past, researchers had to calculate derivatives by hand—tedious and error-prone. PyTorch's **Autograd** feature automates this entire process, making deep learning practical.

### 🔧 How Autograd Works

1. **Computation Graph**: When you perform operations on tensors with `requires_grad=True`, PyTorch builds a graph tracking all operations.

2. **Backward Pass**: Calling `.backward()` traverses this graph in reverse, computing gradients using the chain rule.

3. **Gradient Storage**: Gradients are stored in the `.grad` attribute of each tensor.

### 💡 Practical Usage

```python
# Create tensor that tracks gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Perform operations (builds computation graph)
y = x ** 2 + 3 * x
z = y.sum()

# Compute gradients
z.backward()

# Access gradients: dz/dx = 2x + 3
print(x.grad)  # tensor([7., 9.]) because 2(2)+3=7, 2(3)+3=9
```

### ⚡ Efficiency Tips

* **`torch.no_grad()`**: Disable gradient tracking when not needed (inference, evaluation). Saves memory and speeds up computation.

```python
with torch.no_grad():
    predictions = model(test_data)  # No gradients computed
```

* **`detach()`**: Remove a tensor from the computation graph while keeping its value.

* **`zero_grad()`**: Clear accumulated gradients before each training step (gradients accumulate by default).

---

## 5. Why PyTorch Tensors vs. NumPy Arrays?

While PyTorch Tensors look and feel like NumPy arrays, they offer three critical benefits for deep learning:

| Feature | NumPy Arrays | PyTorch Tensors |
| :--- | :--- | :--- |
| **GPU Support** | ❌ CPU only | ✅ Native GPU acceleration |
| **Autograd** | ❌ No gradient tracking | ✅ Automatic differentiation |
| **DL Ecosystem** | ❌ General purpose | ✅ Built for neural networks |

### 🔄 Seamless Conversion

PyTorch makes it easy to work with both:

```python
import numpy as np
import torch

# NumPy to Tensor
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Tensor to NumPy
back_to_numpy = tensor.numpy()
```

**⚠️ Note:** These share memory! Changes to one affect the other. Use `.clone()` if you need independent copies.

### 🚀 GPU Acceleration

Moving tensors to GPU is straightforward:

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create tensor on GPU
x = torch.rand(1000, 1000, device=device)

# Move existing tensor to GPU
y = existing_tensor.to(device)
```

---

## 6. Common Tensor Operations Reference

### Arithmetic Operations

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Addition | `a + b` or `torch.add(a, b)` | Element-wise addition |
| Subtraction | `a - b` or `torch.sub(a, b)` | Element-wise subtraction |
| Multiplication | `a * b` or `torch.mul(a, b)` | Element-wise multiplication |
| Division | `a / b` or `torch.div(a, b)` | Element-wise division |
| Matrix Multiply | `a @ b` or `torch.matmul(a, b)` | Matrix multiplication |

### Reduction Operations

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Sum | `x.sum()` | Sum all elements |
| Mean | `x.mean()` | Average of all elements |
| Max | `x.max()` | Maximum value |
| Argmax | `x.argmax()` | Index of maximum value |

### Shape Operations

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| Reshape | `x.view(2, 3)` | Change shape (must have same total elements) |
| Flatten | `x.flatten()` | Convert to 1D |
| Transpose | `x.T` or `x.transpose(0, 1)` | Swap dimensions |
| Squeeze | `x.squeeze()` | Remove dimensions of size 1 |
| Unsqueeze | `x.unsqueeze(0)` | Add dimension at position |

---

## 7. Best Practices

### ✅ Memory Management
- Use `torch.no_grad()` during inference
- Clear gradients with `optimizer.zero_grad()` before each backward pass
- Use `.detach()` when you need tensor values without gradient history

### ✅ Device Management
- Always check device compatibility before operations
- Keep all tensors in an operation on the same device
- Use `model.to(device)` to move entire models

### ✅ Data Types
- Use `float32` for most training (balance of precision and speed)
- Consider `float16` (mixed precision) for faster training on modern GPUs
- Use `long` (int64) for class labels and indices

---

*This guide covers the foundational PyTorch concepts. Master these basics before moving on to building neural network architectures.*
