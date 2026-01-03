# 📜 Sequence Models: RNNs, LSTMs, and GRUs

This module provides an in-depth exploration of **Sequence Models**—the specialized neural network architectures designed to process data where the **order of elements** and their **contextual relationships** are critical for making accurate predictions. From basic RNNs to sophisticated gated architectures, we cover everything you need to understand and build powerful sequence processing systems.

---

## 📚 Topics

| File                                                                                | Topic                           | Description                                              |
| :---------------------------------------------------------------------------------- | :------------------------------ | :------------------------------------------------------- |
| [01-Introduction-to-Sequence-Models.md](docs/01-Introduction-to-Sequence-Models.md) | Introduction to Sequence Models | What is sequential data, why order matters, applications |
| [02-Recurrent-Neural-Networks.md](docs/02-Recurrent-Neural-Networks.md)             | Recurrent Neural Networks (RNN) | RNN architecture, hidden states, BPTT, RNN variants      |
| [03-Vanishing-Gradient-Problem.md](docs/03-Vanishing-Gradient-Problem.md)           | Vanishing Gradient Problem      | Why gradients vanish, impact on learning, solutions      |
| [04-LSTM-and-GRU.md](docs/04-LSTM-and-GRU.md)                                       | LSTM & GRU Architectures        | Gated mechanisms, cell states, when to use each          |

---

## 🎯 Learning Path

1. **Understand sequential data** → [01-Introduction-to-Sequence-Models.md](docs/01-Introduction-to-Sequence-Models.md)
2. **Learn RNN fundamentals** → [02-Recurrent-Neural-Networks.md](docs/02-Recurrent-Neural-Networks.md)
3. **Study the vanishing gradient problem** → [03-Vanishing-Gradient-Problem.md](docs/03-Vanishing-Gradient-Problem.md)
4. **Master gated architectures** → [04-LSTM-and-GRU.md](docs/04-LSTM-and-GRU.md)

---

## 🔑 Key Concepts

### Why Sequence Models?

Traditional neural networks treat each input independently and cannot handle sequential dependencies. Sequence models solve this by maintaining **memory** of previous inputs:

```
Feedforward Network (Bad for sequences):
  x₁ → y₁ (isolated)
  x₂ → y₂ (isolated)
  x₃ → y₃ (isolated)
  - Each prediction is independent
  - No memory of previous inputs
  - Can't understand "The cat sat on the ___"

Sequence Model (Designed for sequences):
  x₁ → h₁ → y₁
       ↓
  x₂ → h₂ → y₂
       ↓
  x₃ → h₃ → y₃
  - Hidden state carries information forward
  - Each prediction considers context
  - Understands sequential dependencies
```

### The Sequence Model Pipeline

```
Sequential Input (e.g., words, time steps)
    ↓
┌─────────────────────────────────────┐
│  SEQUENTIAL PROCESSING              │
│  ┌─────────┐   ┌─────────┐         │
│  │  RNN/   │ → │  RNN/   │ → ...   │
│  │  LSTM   │   │  LSTM   │         │
│  └─────────┘   └─────────┘         │
│  Step 1:       Step 2:             │
│  Process x₁    Process x₂          │
│  Update h₁     Update h₂           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  OUTPUT GENERATION                  │
│  ┌─────────┐   ┌─────────┐         │
│  │  Dense  │ → │ Softmax │ → Output│
│  │  Layer  │   │         │         │
│  └─────────┘   └─────────┘         │
│  Transform     Generate             │
│  hidden state  predictions          │
└─────────────────────────────────────┘
```

### Architecture Comparison at a Glance

| Feature              |       RNN       |          LSTM           |        GRU         |
| :------------------- | :-------------: | :---------------------: | :----------------: |
| **Parameters**       |     Fewest      |          Most           |       Medium       |
| **Training Speed**   |     Fastest     |         Slowest         |       Medium       |
| **Long-term Memory** |     ❌ Poor     |      ✅ Excellent       |      ✅ Good       |
| **Gates**            |        0        |            3            |         2          |
| **Best For**         | Short sequences | Complex, long sequences | Moderate sequences |

---

## 📊 Quick Reference

### PyTorch Sequence Model Building Blocks

```python
import torch.nn as nn

# Basic RNN
nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)

# LSTM
nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

# GRU
nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)

# Embedding layer (for text)
nn.Embedding(vocab_size, embedding_dim)

# Fully connected output
nn.Linear(hidden_size, output_size)
```

### Simple LSTM Example

```python
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch, seq_len, hidden)
        out = self.fc(h_n[-1])                # Use last hidden state
        return out
```

---

## 🚀 Real-World Applications

| Application             | Architecture | Description                        |
| :---------------------- | :----------- | :--------------------------------- |
| **Machine Translation** | LSTM/GRU     | Translating text between languages |
| **Speech Recognition**  | LSTM         | Converting audio to text           |
| **Text Generation**     | LSTM/GRU     | Generating coherent text sequences |
| **Sentiment Analysis**  | RNN/LSTM     | Classifying emotions in text       |
| **Stock Prediction**    | LSTM         | Forecasting financial time series  |
| **Music Generation**    | LSTM         | Composing musical sequences        |
| **Video Captioning**    | LSTM         | Describing video content in text   |

---

## 🎓 Prerequisites

Before diving into Sequence Models, you should understand:

- Basic neural network concepts (neurons, layers, activation functions)
- PyTorch fundamentals (tensors, nn.Module)
- Training loops and backpropagation
- Matrix operations and linear algebra

---

## 🔗 Additional Resources

- 📄 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah's legendary blog post
- 📖 [Deep Learning Book - Chapter 10](https://www.deeplearningbook.org/contents/rnn.html) - Goodfellow et al.
- 🎥 [Stanford CS231n RNN Lecture](https://www.youtube.com/watch?v=6niqTuYFZLQ) - Andrej Karpathy

---

_Sequence models are the foundation of natural language processing and time series analysis. Master these concepts to build text generators, translators, and predictive systems!_
