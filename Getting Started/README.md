# 🧠 Deep Learning Basics

Welcome to the comprehensive documentation for Deep Learning (DL) foundations. This guide is designed for students and practitioners to understand the "why" and "how" behind neural networks, architectures, and the modern AI tooling landscape.

---

## 📚 Documentation

| File | Topic | Description |
|------|-------|-------------|
| [01-Neural-Networks-Foundation.md](01-Neural-Networks-Foundation.md) | Neural Networks | Core architecture, neurons, how information flows |
| [02-DL-vs-Statistical-ML.md](02-DL-vs-Statistical-ML.md) | Decision Matrix | When to use DL vs. traditional ML |
| [03-NN-Architectures.md](03-NN-Architectures.md) | Architectures | FNN, CNN, RNN, Transformers and use cases |
| [04-Developer-Toolkit.md](04-Developer-Toolkit.md) | Tools | PyTorch, TensorFlow, GPUs, cloud options |
| [05-Training-Fundamentals.md](05-Training-Fundamentals.md) | Training | Loss functions, backpropagation, optimizers |
| [06-Common-Challenges.md](06-Common-Challenges.md) | Challenges | Overfitting, underfitting, vanishing gradients |
| [07-Evaluation-Metrics.md](07-Evaluation-Metrics.md) | Metrics | Classification and regression evaluation |
| [08-Best-Practices.md](08-Best-Practices.md) | Best Practices | Data prep, model development, experimentation |
| [09-Learning-Resources.md](09-Learning-Resources.md) | Resources | Courses, books, practice platforms |
| [10-Glossary.md](10-Glossary.md) | Glossary | Key terminology reference |

---

## 🎯 Learning Path

1. **Neural Networks Foundation** → Understand the building blocks
2. **DL vs. Statistical ML** → Know when to use each approach
3. **Architectures** → Learn about FNN, CNN, RNN, Transformers
4. **Developer Toolkit** → Set up your environment
5. **Training Fundamentals** → Master the learning process
6. **Common Challenges** → Recognize and fix problems
7. **Evaluation Metrics** → Measure model performance
8. **Best Practices** → Build robust systems


---

## 🔑 Quick Reference

### Neural Network Layers
```
Input Layer → Hidden Layer(s) → Output Layer
    ↓              ↓                ↓
 Raw Data    Feature Learning    Prediction
```

### Architecture Selection

| Data Type | Recommended Architecture |
|-----------|-------------------------|
| Tabular | FNN (or traditional ML) |
| Images | CNN |
| Sequences | RNN/LSTM or Transformer |
| Text/NLP | Transformer |

### Common Optimizers

| Optimizer | When to Use |
|-----------|-------------|
| **Adam** | Default choice for most tasks |
| **SGD + Momentum** | When you need better generalization |
| **AdamW** | When using weight decay |

### Key Hyperparameters

| Parameter | Typical Values |
|-----------|---------------|
| Learning Rate | 0.001 - 0.0001 (Adam) |
| Batch Size | 32, 64, 128, 256 |
| Dropout | 0.2 - 0.5 |

---

## 📖 Prerequisites

Before diving in, you should have:
- Basic Python programming skills
- High school mathematics (algebra, basic calculus helpful)
- Curiosity and willingness to experiment

---

*Happy learning! Remember: the best way to understand deep learning is to build things. Start small, experiment often, and don't be afraid to break things.* 🚀
