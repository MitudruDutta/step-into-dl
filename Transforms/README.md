# 🤖 Transformers: The State-of-the-Art Architecture

Welcome to the **Transformers** module! This comprehensive guide explores the foundational architecture that powers modern Large Language Models (LLMs) like GPT, BERT, and their successors.

---

## 📖 Learning Path

Follow these topics in order for a complete understanding of Transformers:

| #   | Topic                                                      | Description                                                    |
| :-- | :--------------------------------------------------------- | :------------------------------------------------------------- |
| 1   | [Word Embeddings](01-Word-Embeddings.md)                   | How words become vectors that capture semantic meaning         |
| 2   | [Architecture Overview](02-Architecture-Overview.md)       | Encoder, Decoder, and their variants (BERT vs GPT vs T5)       |
| 3   | [Attention Mechanism](03-Attention-Mechanism.md)           | The core innovation: Query, Key, Value, and scaled dot-product |
| 4   | [Multi-Head Attention](04-Multi-Head-Attention.md)         | How multiple attention heads capture diverse patterns          |
| 5   | [Decoder Mechanics](05-Decoder-Mechanics.md)               | Masked self-attention and cross-attention for generation       |
| 6   | [Self-Supervised Training](06-Self-Supervised-Training.md) | CLM (GPT-style) vs MLM (BERT-style) training                   |

---

## 📂 Repository Contents

| File                        | Description                                                             |
| :-------------------------- | :---------------------------------------------------------------------- |
| `BERT.ipynb`                | Hands-on implementation of BERT (Bidirectional Encoder Representations) |
| `GPT2.ipynb`                | Practical notebook demonstrating GPT-2 architecture and text generation |
| `spam_classification.ipynb` | Real-world application: Building a spam classifier with Transformers    |
| `spam.csv`                  | Dataset for the spam classification task                                |

---

## 🎯 Quick Overview

### What are Transformers?

Transformers are neural network architectures that process sequences using **self-attention** instead of recurrence. This enables:

- **Parallel Processing**: All tokens processed simultaneously
- **Long-Range Dependencies**: Direct connections between any positions
- **Scalability**: Can be trained on massive datasets

### The Core Components

```text
┌─────────────────────────────────────────────────────────────┐
│                     TRANSFORMER                            │
│                                                            │
│   ┌─────────────┐          ┌─────────────┐                 │
│   │   ENCODER   │  ──────► │   DECODER   │                 │
│   │             │ Context  │             │                 │
│   │ Bidirectional│         │ Autoregressive│               │
│   │ Self-Attention│        │ + Cross-Attn │                │
│   └─────────────┘          └─────────────┘                  │ 
│         ↑                         │                        │
│    Input Tokens             Output Tokens                  │
└─────────────────────────────────────────────────────────────┘
```

### Model Variants

| Type                | Examples      | Best For                   |
| :------------------ | :------------ | :------------------------- |
| **Encoder-Only**    | BERT, RoBERTa | Classification, NER, Q&A   |
| **Decoder-Only**    | GPT, LLaMA    | Text generation, chat      |
| **Encoder-Decoder** | T5, BART      | Translation, summarization |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers torch datasets
```

### Quick Example

```python
from transformers import pipeline

# Text generation with GPT-2
generator = pipeline("text-generation", model="gpt2")
result = generator("Transformers are", max_length=50)
print(result[0]['generated_text'])

# Classification with BERT
classifier = pipeline("sentiment-analysis")
result = classifier("I love learning about transformers!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```

---

## 🔑 Key Concepts Summary

| Concept            | One-Line Description                                |
| :----------------- | :-------------------------------------------------- |
| **Embeddings**     | Dense vectors that capture word meaning             |
| **Self-Attention** | Every token attends to every other token            |
| **Multi-Head**     | Multiple attention patterns in parallel             |
| **Encoder**        | Creates contextual representations (bidirectional)  |
| **Decoder**        | Generates sequences (autoregressive)                |
| **CLM**            | Causal Language Modeling (predict next token) - GPT |
| **MLM**            | Masked Language Modeling (fill in blanks) - BERT    |

---

## 📚 Further Reading

- **"Attention Is All You Need"** - Vaswani et al., 2017
- **"BERT"** - Devlin et al., 2018
- **"GPT-2"** - Radford et al., 2019
- **The Illustrated Transformer** - Jay Alammar

---

_Start with [01-Word-Embeddings.md](01-Word-Embeddings.md) to begin your journey!_ 🎓
