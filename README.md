# 🧠 Step Into Deep Learning

A structured, hands-on learning repository for mastering deep learning fundamentals. From neurons to optimizers, this project provides comprehensive documentation and practical PyTorch implementations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What You'll Learn

| Module | Topics | Difficulty |
|--------|--------|------------|
| Getting Started | DL foundations, architectures, toolkit | ⭐ Beginner |
| Neural Networks: Basics | Neurons, perceptrons, activation functions | ⭐ Beginner |
| PyTorch Fundamentals | Tensors, autograd, GPU computing | ⭐⭐ Intermediate |
| Neural Network Training | Backprop, gradient descent, optimizers | ⭐⭐ Intermediate |
| Neural Networks in PyTorch | nn.Module, DataLoaders, loss functions | ⭐⭐ Intermediate |
| Model Optimization: Training Algorithms | Momentum, RMSProp, Adam | ⭐⭐⭐ Advanced |
| Model Optimization: Regularization | Dropout, L1/L2, BatchNorm, Early Stopping | ⭐⭐⭐ Advanced |

---

## 📁 Repository Structure

```
step-into-dl/
│
├── 📘 Getting Started/
│   ├── README.md                      # Module index
│   ├── 01-Neural-Networks-Foundation.md
│   ├── 02-DL-vs-Statistical-ML.md
│   ├── 03-NN-Architectures.md
│   ├── 04-Developer-Toolkit.md
│   ├── 05-Training-Fundamentals.md
│   ├── 06-Common-Challenges.md
│   ├── 07-Evaluation-Metrics.md
│   ├── 08-Best-Practices.md
│   ├── 09-Learning-Resources.md
│   └── 10-Glossary.md
│
├── 🔬 Neural Networks: Basics/
│   ├── README.md
│   ├── 01-What-is-a-Neuron.md
│   ├── 02-Perceptrons-to-MLPs.md
│   ├── 03-Insurance-Prediction-Intuition.md
│   ├── 04-Role-of-Activation-Functions.md
│   ├── 05-Activation-Functions-Guide.md
│   ├── 06-Practical-Tips.md
│   └── functions.ipynb                # 📓 Activation implementations
│
├── 🔥 Pytorch/
│   ├── README.md
│   ├── 01-Matrix-Fundamentals.md
│   ├── 02-Tensor-Basics.md
│   ├── 03-Calculus-for-Learning.md
│   ├── 04-Autograd-Explained.md
│   ├── 05-Tensors-vs-NumPy.md
│   ├── 06-Common-Operations.md
│   ├── 07-Best-Practices.md
│   ├── tensor1.ipynb                  # 📓 Tensor operations & GPU
│   ├── tensor2.ipynb                  # 📓 Reshaping & initialization
│   └── autograd.ipynb                 # 📓 Automatic differentiation
│
├── 📈 Neural Network: Training/
│   ├── README.md
│   ├── 01-Backpropagation.md
│   ├── 02-Gradient-Descent.md
│   ├── 03-GD-Variants.md
│   ├── 04-Optimizers.md
│   ├── 05-Monitoring-Training.md
│   ├── data_generation.ipynb          # 📓 Synthetic data creation
│   ├── gradient_descent.ipynb         # 📓 GD from scratch
│   └── gd_vs_mini_gd_vs_sgd.ipynb     # 📓 GD variants comparison
│
├── ⚡ Neural Networks: Pytorch/
│   ├── README.md
│   ├── 01-nn-Module.md
│   ├── 02-Datasets-DataLoaders.md
│   ├── 03-Binary-Cross-Entropy.md
│   ├── 04-Categorical-Cross-Entropy.md
│   ├── 05-Training-Loop.md
│   ├── log_loss.ipynb                 # 📓 MSE vs BCE
│   ├── cross_entropy_loss.ipynb       # 📓 Multi-class loss
│   ├── dataset_dataloader.ipynb       # 📓 Data pipelines
│   └── handwritten_digits.ipynb       # 📓 MNIST classifier
│
├── 🚀 Model Optimization: Training Algorithms/
│   ├── README.md
│   ├── 01-What-is-Model-Optimization.md
│   ├── 02-EWMA-Foundation.md
│   ├── 03-Momentum.md
│   ├── 04-RMSProp.md
│   ├── 05-Adam.md
│   ├── 06-Optimizer-Comparison.md
│   └── optimizers.ipynb               # 📓 Optimizer comparison
│
├── 🛡️ Model Optimization: Regularization Techniques/
│   ├── README.md                      # Module overview and learning path
│   ├── 01-Understanding-Regularization.md  # Overfitting and bias-variance
│   ├── 02-Dropout.md                  # Dropout regularization
│   ├── 03-L1-L2-Regularization.md     # Weight penalties and decay
│   ├── 04-Batch-Normalization.md      # Normalizing layer inputs
│   ├── 05-Early-Stopping.md           # Optimal stopping point
│   ├── 06-Data-Augmentation.md        # Expanding training data
│   ├── dropout_regularization.ipynb   # 📓 Dropout comparison
│   ├── l2_regularization.ipynb        # 📓 Weight decay demo
│   ├── batch_norm.ipynb               # 📓 BatchNorm on MNIST
│   └── early_stopping.ipynb           # 📓 Early stopping implementation
│
└── README.md                          # You are here
```

---

## 🛤️ Learning Path

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Getting        │     │  Neural Nets    │     │    PyTorch      │
│  Started        │ ──► │  Basics         │ ──► │  Fundamentals   │
│  (Theory)       │     │  (Neurons)      │     │  (Tensors)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  NN Training    │     │  NNs in         │     │    Model        │
│  (Backprop,     │ ──► │  PyTorch        │ ──► │  Optimization   │
│   GD)           │     │  (nn.Module)    │     │  (Adam, etc.)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Quick Start

1. **New to Deep Learning?** → Start with `Getting Started/01-Neural-Networks-Foundation.md`
2. **Know the basics?** → Jump to `Neural Network: Training/` for hands-on practice
3. **Ready to build?** → Go to `Neural Networks: Pytorch/handwritten_digits.ipynb`

---

## 📓 Notebooks Overview

| Notebook | Module | What You'll Build |
|----------|--------|-------------------|
| `functions.ipynb` | Basics | Sigmoid, Softmax, Tanh, ReLU from scratch |
| `tensor1.ipynb` | PyTorch | Tensor ops, matrix multiplication, GPU usage |
| `tensor2.ipynb` | PyTorch | Reshaping, broadcasting, initialization |
| `autograd.ipynb` | PyTorch | Gradient computation, computational graphs |
| `gradient_descent.ipynb` | Training | GD optimizer from scratch |
| `gd_vs_mini_gd_vs_sgd.ipynb` | Training | Compare Batch/Mini-Batch/SGD |
| `log_loss.ipynb` | PyTorch NN | MSE vs BCE for classification |
| `cross_entropy_loss.ipynb` | PyTorch NN | Multi-class classification loss |
| `dataset_dataloader.ipynb` | PyTorch NN | FashionMNIST data pipeline |
| `handwritten_digits.ipynb` | PyTorch NN | Complete MNIST classifier |
| `optimizers.ipynb` | Optimization | SGD vs Momentum vs Adam |
| `dropout_regularization.ipynb` | Regularization | Dropout effect on Sonar dataset |
| `l2_regularization.ipynb` | Regularization | Weight decay and weight distributions |
| `batch_norm.ipynb` | Regularization | BatchNorm impact on MNIST training |
| `early_stopping.ipynb` | Regularization | Patience-based stopping with checkpoints |

---

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/step-into-dl.git
cd step-into-dl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision numpy pandas matplotlib jupyter scikit-learn
```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📚 Module Details

### 1. Getting Started
Foundational concepts for understanding deep learning:
- Neural network architecture and information flow
- When to use DL vs. traditional ML
- Popular architectures: FNN, CNN, RNN, Transformers
- Developer toolkit: PyTorch, TensorFlow, hardware options
- Common challenges: overfitting, vanishing gradients

### 2. Neural Networks: Basics
The building blocks of neural networks:
- Biological inspiration and artificial neurons
- From Perceptrons to Multi-Layer Perceptrons
- Activation functions: why non-linearity matters
- Comprehensive guide with formulas and use cases

### 3. PyTorch Fundamentals
Essential PyTorch skills:
- Tensors: creation, operations, GPU acceleration
- Autograd: automatic differentiation explained
- NumPy interoperability and best practices

### 4. Neural Network Training
How networks learn:
- Backpropagation: the chain rule in action
- Gradient Descent variants: Batch, Mini-Batch, SGD
- Monitoring training: loss curves, debugging tips

### 5. Neural Networks in PyTorch
Building real models:
- `nn.Module`: the foundation of PyTorch models
- Datasets and DataLoaders for efficient training
- Loss functions: BCE, Cross Entropy, when to use each

### 6. Model Optimization: Training Algorithms
Advanced training techniques:
- EWMA: the math behind modern optimizers
- Momentum: accelerating convergence
- RMSProp: adaptive learning rates
- Adam: the gold standard optimizer

### 7. Model Optimization: Regularization Techniques
Preventing overfitting:
- Dropout: randomly deactivating neurons
- L1/L2 regularization: weight penalties
- Batch Normalization: stabilizing training
- Early Stopping: knowing when to stop
- Data Augmentation: expanding training data

---

## 📖 Recommended Resources

### Courses
- [fast.ai](https://www.fast.ai/) — Practical deep learning
- [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) — Andrew Ng
- [CodeBasics Deep Learning](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO) — YouTube series

### Books
- *Deep Learning* by Goodfellow, Bengio, Courville
- *Hands-On Machine Learning* by Aurélien Géron
- *PyTorch Documentation* — [pytorch.org/docs](https://pytorch.org/docs)

### Practice
- [Kaggle](https://www.kaggle.com/) — Competitions and datasets
- [Google Colab](https://colab.research.google.com/) — Free GPU notebooks
- [Hugging Face](https://huggingface.co/) — Pre-trained models

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit PRs to improve documentation
- Add new topics or notebooks

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Start small, experiment often, and don't be afraid to break things.</i> 🚀
</p>
