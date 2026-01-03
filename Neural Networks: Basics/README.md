# 🧠 Deep Learning Fundamentals: Neurons & Activation Functions

This module provides an in-depth technical exploration of the **Neuron**, the core architectural component of neural networks, and the mathematical logic behind **Activation Functions**.

---

## 📚 Documentation

| File                                                                              | Topic                  | Description                                                             |
| --------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------- |
| [01-What-is-a-Neuron.md](docs/01-What-is-a-Neuron.md)                             | Neurons                | The basic building block, computational process, biological inspiration |
| [02-Perceptrons-to-MLPs.md](docs/02-Perceptrons-to-MLPs.md)                       | Architecture Evolution | From single perceptrons to multilayer networks                          |
| [03-Insurance-Prediction-Intuition.md](docs/03-Insurance-Prediction-Intuition.md) | Intuition              | How networks extract patterns and learn features automatically          |
| [04-Role-of-Activation-Functions.md](docs/04-Role-of-Activation-Functions.md)     | Why Activation?        | Non-linearity, the detective analogy, gradient flow                     |
| [05-Activation-Functions-Guide.md](docs/05-Activation-Functions-Guide.md)         | Activation Guide       | Sigmoid, Softmax, Tanh, ReLU, Leaky ReLU in detail                      |
| [06-Practical-Tips.md](docs/06-Practical-Tips.md)                                 | Practical Tips         | Weight initialization, debugging, scaling                               |

---

## 💻 Notebooks

| Notebook                                     | Description                                    |
| -------------------------------------------- | ---------------------------------------------- |
| [functions.ipynb](notebooks/functions.ipynb) | Activation function implementations with NumPy |

---

## 🎯 Learning Path

1. **What is a Neuron** → Understand the fundamental building block
2. **Perceptrons to MLPs** → See how architectures evolved
3. **Insurance Prediction Intuition** → Grasp how networks learn features
4. **Role of Activation Functions** → Understand why non-linearity matters
5. **Activation Functions Guide** → Deep dive into each activation function
6. **Practical Tips** → Apply knowledge to real implementations
7. **Practice with functions.ipynb** → Implement activations hands-on

---

## 🔑 Key Concepts

### The Neuron Formula

```
z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
output = activation(z)
```

### Activation Function Quick Reference

| Function   | Range   | Best For                                   |
| ---------- | ------- | ------------------------------------------ |
| Sigmoid    | (0, 1)  | Binary classification output               |
| Softmax    | (0, 1)  | Multi-class classification output          |
| Tanh       | (-1, 1) | Hidden layers needing zero-centered output |
| ReLU       | [0, ∞)  | Default for hidden layers                  |
| Leaky ReLU | (-∞, ∞) | When ReLU neurons are dying                |

### Architecture Comparison

| Aspect   | Perceptron      | MLP                   |
| -------- | --------------- | --------------------- |
| Layers   | Single          | Multiple              |
| Problems | Linear only     | Linear and non-linear |
| Learning | Perceptron rule | Backpropagation       |

---

## 📖 Prerequisites

Before this module, you should understand:

- Basic Python programming
- High school algebra
- Basic calculus concepts (helpful but not required)

---

_Master these fundamentals before moving on to training neural networks and more complex architectures._
