# 🚀 Step Into Deep Learning

A hands-on learning repository for understanding deep learning fundamentals from the ground up. This project provides comprehensive documentation and practical code examples to help you master neural networks and modern AI concepts.

---

## 📁 Repository Structure

```
step-into-dl/
├── Getting Started/
│   └── README.md                  # Deep learning foundations & overview
├── Neural Networks: Basics/
│   ├── README.md                  # Neurons & activation functions theory
│   └── functions.ipynb            # Activation function implementations
├── Pytorch/
│   ├── README.md                  # Matrices, tensors & calculus fundamentals
│   ├── tensor1.ipynb              # Tensor operations, matrix math, GPU
│   ├── tensor2.ipynb              # Tensor attributes, reshaping, init
│   └── autograd.ipynb             # Automatic differentiation & gradients
├── Neural Network: Training/
│   ├── README.md                  # Training module overview
│   ├── 01-Backpropagation.md      # How networks learn from errors
│   ├── 02-Gradient-Descent.md     # Optimization fundamentals
│   ├── 03-GD-Variants.md          # Batch vs Mini-Batch vs SGD
│   ├── 04-Optimizers.md           # Adam, SGD+Momentum, RMSprop
│   ├── 05-Monitoring-Training.md  # Metrics, debugging, early stopping
│   ├── data_generation.ipynb      # Generate synthetic training data
│   ├── gradient_descent.ipynb     # GD implementation from scratch
│   └── gd_vs_mini_gd_vs_sgd.ipynb # Compare GD variants
└── README.md                      # You are here
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

### 4. Neural Network Training
Complete guide to training neural networks:
- Backpropagation: how networks learn from errors
- Gradient Descent: the optimization engine
- GD Variants: Batch, Mini-Batch, and SGD comparison
- Advanced Optimizers: Adam, SGD+Momentum, RMSprop, AdamW
- Monitoring Training: metrics, debugging, early stopping

### 5. Practical Implementations
Jupyter notebooks with working code and detailed explanations:

| Notebook | Location | Topics |
|----------|----------|--------|
| `functions.ipynb` | Neural Networks: Basics | Sigmoid, Softmax, Tanh, ReLU with NumPy |
| `tensor1.ipynb` | Pytorch | Tensor creation, arithmetic, matrix multiplication, GPU |
| `tensor2.ipynb` | Pytorch | Shape, dtype, device, reshaping, initialization |
| `autograd.ipynb` | Pytorch | Gradient tracking, backward(), chain rule |
| `data_generation.ipynb` | Neural Network: Training | Generate synthetic employee bonus dataset |
| `gradient_descent.ipynb` | Neural Network: Training | Implement GD from scratch in PyTorch |
| `gd_vs_mini_gd_vs_sgd.ipynb` | Neural Network: Training | Compare Batch GD, Mini-Batch GD, SGD |

---

## 🛠️ Prerequisites

- Python 3.8+
- NumPy
- Pandas
- PyTorch
- Matplotlib
- Jupyter Notebook

```bash
pip install numpy pandas torch matplotlib jupyter
```

---

## 🎯 Learning Path

1. **Start here** → `Getting Started/README.md` for foundational concepts
2. **Go deeper** → `Neural Networks: Basics/README.md` for neuron mechanics
3. **Practice activations** → `Neural Networks: Basics/functions.ipynb`
4. **Learn PyTorch** → `Pytorch/README.md` for tensors and calculus
5. **Tensor operations** → `Pytorch/tensor1.ipynb` and `tensor2.ipynb`
6. **Master autograd** → `Pytorch/autograd.ipynb` for automatic differentiation
7. **Understand training** → `Neural Network: Training/` documentation
8. **Hands-on GD** → `gradient_descent.ipynb` and `gd_vs_mini_gd_vs_sgd.ipynb`

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
