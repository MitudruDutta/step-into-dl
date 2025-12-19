# 🚀 Step Into Deep Learning

A hands-on learning repository for understanding deep learning fundamentals from the ground up. This project provides comprehensive documentation and practical code examples to help you master neural networks and modern AI concepts.

---

## 📁 Repository Structure

```
step-into-dl/
├── Getting Started/
│   └── README.md          # Deep learning foundations & overview
├── Neural Networks: Basics/
│   ├── README.md          # Neurons & activation functions theory
│   └── functions.ipynb    # Activation function implementations
├── Pytorch/
│   ├── README.md          # Matrices, tensors & calculus fundamentals
│   ├── tensor1.ipynb      # Tensor operations, matrix math, GPU acceleration
│   ├── tensor2.ipynb      # Tensor attributes, reshaping, initialization
│   └── autograd.ipynb     # Automatic differentiation & gradients
└── README.md              # You are here
```

---

## 📚 What's Covered

### 1. Getting Started
A comprehensive introduction to deep learning covering:
- Neural network architecture (input, hidden, output layers)
- Deep Learning vs. Statistical ML decision matrix
- Popular architectures (FNN, CNN, RNN, Transformers) and their use cases
- Developer toolkit: PyTorch, TensorFlow, GPUs/TPUs
- Training fundamentals: loss functions, backpropagation, optimizers
- Common challenges: overfitting, underfitting, vanishing gradients
- Evaluation metrics for classification and regression
- Best practices and learning resources

### 2. Neural Networks: Basics
Deep dive into the building blocks of neural networks:
- What is a neuron and how it processes information
- Evolution from Perceptrons to Multilayer Perceptrons (MLPs)
- Intuitive examples (insurance prediction model)
- Comprehensive guide to activation functions:
  - Sigmoid, Softmax, Tanh, ReLU, Leaky ReLU
  - When to use each function
  - Mathematical formulas and characteristics

### 3. PyTorch Fundamentals
Introduction to PyTorch and the math behind deep learning:
- Matrix fundamentals and why they matter for AI
- Tensor basics: dimensions, attributes, and operations
- Calculus for learning: derivatives, chain rule, and gradients
- Autograd: automatic differentiation in PyTorch
- PyTorch tensors vs. NumPy arrays
- Common tensor operations reference

### 4. Practical Implementations
Jupyter notebooks with working code and detailed explanations:

| Notebook | Topics |
|----------|--------|
| `functions.ipynb` | Sigmoid, Softmax, Tanh, ReLU implementations with NumPy |
| `tensor1.ipynb` | Tensor creation, arithmetic, matrix multiplication, GPU acceleration |
| `tensor2.ipynb` | Shape, dtype, device attributes, reshaping, initialization |
| `autograd.ipynb` | Gradient tracking, backward(), chain rule, torch.no_grad() |

---

## 🛠️ Prerequisites

- Python 3.8+
- NumPy
- PyTorch
- Jupyter Notebook (for running `.ipynb` files)

```bash
pip install numpy torch jupyter
```

---

## 🎯 Learning Path

1. **Start here** → `Getting Started/README.md` for foundational concepts
2. **Go deeper** → `Neural Networks: Basics/README.md` for neuron mechanics
3. **Practice activations** → `Neural Networks: Basics/functions.ipynb`
4. **Learn PyTorch** → `Pytorch/README.md` for tensors and calculus
5. **Tensor operations** → `Pytorch/tensor1.ipynb` and `tensor2.ipynb`
6. **Master autograd** → `Pytorch/autograd.ipynb` for automatic differentiation

---

## 📖 Recommended Resources

- **Courses**: fast.ai, Coursera Deep Learning Specialization, CodeBasics Deep Learning
- **Books**: *Deep Learning* by Goodfellow et al., *Hands-On Machine Learning* by Géron
- **Practice**: Kaggle, Google Colab, Hugging Face

---

## 🤝 Contributing

Feel free to open issues or submit PRs to improve the documentation or add new topics.

---

*Happy learning! Start small, experiment often, and don't be afraid to break things.* 🧠
