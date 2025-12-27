# 🎯 The Attention Mechanism: Focused Learning

The **Attention Mechanism** is the core innovation of the Transformer architecture. It allows the model to dynamically focus on relevant parts of the input when producing each output, enabling parallel processing and capturing long-range dependencies.

---

## The Core Intuition

Imagine translating "The cat sat on the mat" to French. When generating "chat" (cat), you focus on "cat." When generating "tapis" (mat), you shift focus to "mat." Attention automates this selective focus.

```text
English: "The  cat  sat  on  the  mat"
              ↓              ↓
           HIGH           HIGH
          attention      attention
              ↓              ↓
French:   "Le chat s'est assis sur le tapis"
```

---

## Self-Attention: Relating Tokens

Self-attention computes relationships between ALL input tokens simultaneously, capturing dependencies regardless of distance.

### The Three Components

| Component     | Symbol    | Description           | Intuition                  |
| :------------ | :-------- | :-------------------- | :------------------------- |
| **Query (Q)** | `Q = XWq` | What I'm looking for  | "What am I searching for?" |
| **Key (K)**   | `K = XWk` | What I contain        | "What do I represent?"     |
| **Value (V)** | `V = XWv` | My actual information | "What info do I carry?"    |

### The Attention Formula

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

### Step-by-Step Breakdown

```text
Step 1: Compute Similarity Scores
        QKᵀ → dot products between queries and keys

        "The"   "cat"   "sat"   "on"
   "The"  0.8    0.3     0.1    0.2
   "cat"  0.2    0.9     0.7    0.1
   "sat"  0.1    0.6     0.8    0.4
   "on"   0.2    0.1     0.3    0.9

Step 2: Scale
        Divide by √dₖ to prevent extreme values

Step 3: Softmax
        Convert to probabilities (sum to 1)

        "The"   "cat"   "sat"   "on"
   "The" 0.45   0.25    0.15   0.15    = 1.0
   "cat" 0.15   0.40    0.35   0.10    = 1.0
   ...

Step 4: Weighted Sum
        Multiply by Values to get output
        Each token is now a weighted combination of all tokens
```

---

## Visualizing Attention

```text
Input: "The cat sat on the mat"

Attention weights for "sat":

Token:    The   cat   sat   on   the   mat
Weight:  0.05  0.35  0.30  0.05  0.05  0.20
              ▲▲▲              ▲▲▲
         subject peak      object peak

The verb "sat" strongly attends to:
  - "cat" (who is sitting)
  - "mat" (where sitting occurred)
```

### Attention Heatmap

```text
         The  cat  sat  on  the  mat
    ┌────────────────────────────────┐
The │ ███  ░░░  ░░░  ░░░  ░░░  ░░░  │
cat │ ░░░  ███  ▓▓▓  ░░░  ░░░  ░░░  │
sat │ ░░░  ▓▓▓  ███  ░░░  ░░░  ▓▓░  │
on  │ ░░░  ░░░  ▓▓░  ███  ░░░  ▓▓▓  │
the │ ░░░  ░░░  ░░░  ░░░  ███  ░▓░  │
mat │ ░░░  ░▓░  ▓▓░  ▓▓▓  ░▓░  ███  │
    └────────────────────────────────┘

Legend: ███ = high attention, ▓▓▓ = medium, ░░░ = low
```

---

## Why Scale by √dₖ?

```text
Problem without scaling:

When dₖ (dimension) is large (e.g., 64):
  dot product values become very large/small

  QKᵀ might produce: [25.3, 2.1, -18.7, 31.2]

  softmax([25.3, 2.1, -18.7, 31.2])
  = [0.003, 0.000, 0.000, 0.997]  ← Almost one-hot!

  Gradients become tiny (vanishing gradient)

Solution - scale by √dₖ:

  [25.3, 2.1, -18.7, 31.2] / √64
  = [3.16, 0.26, -2.34, 3.90]

  softmax = [0.21, 0.11, 0.01, 0.47]  ← Smoother distribution!
```

---

## Self-Attention vs Traditional Approaches

### Compared to RNNs

```text
RNN Processing (Sequential):

  x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → x₄ → h₄
           ↑         ↑         ↑
        Info from x₁ must pass through
        every intermediate step

  Distance from x₁ to x₄: 3 steps
  Information degrades with distance

Self-Attention (Parallel):

       x₁ ←──────────────────────→ x₄
        ↑                          ↑
  Direct connection! No degradation

  Distance from any token to any other: 1 step
```

### Comparison Table

| Aspect              | RNN        | Self-Attention  |
| :------------------ | :--------- | :-------------- |
| **Max Path Length** | O(n)       | O(1)            |
| **Parallelization** | Sequential | Fully parallel  |
| **Long-range deps** | Difficult  | Easy            |
| **Computation**     | O(n)       | O(n²) per layer |

---

## Implementation in PyTorch

### Basic Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)

        Q = self.W_q(x)  # Queries
        K = self.W_k(x)  # Keys
        V = self.W_v(x)  # Values

        # Attention scores: QKᵀ / √dₖ
        d_k = self.embed_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# Example
attention = SelfAttention(embed_dim=64)
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10
output, weights = attention(x)
print(f"Output shape: {output.shape}")  # (2, 10, 64)
print(f"Attention weights: {weights.shape}")  # (2, 10, 10)
```

### Masked Self-Attention (for Decoders)

```python
def create_causal_mask(seq_len):
    """Create mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Usage in attention:
# scores = scores + causal_mask
# attn_weights = F.softmax(scores, dim=-1)
```

---

## Attention Patterns

Different attention heads learn different patterns:

```text
HEAD 1: Syntactic relationships
┌─────────────────────────┐
│ "The cat that I saw ran"│
│   ↑___________↑         │
│ subject-verb agreement  │
└─────────────────────────┘

HEAD 2: Adjacent tokens
┌─────────────────────────┐
│ "The  cat  sat"         │
│   ↔    ↔               │
│ Neighboring attention   │
└─────────────────────────┘

HEAD 3: Semantic relationships
┌─────────────────────────┐
│ "Paris is the capital"  │
│   ↑____________↑        │
│ Paris-capital link      │
└─────────────────────────┘
```

---

## Summary

| Concept              | Description                        |
| :------------------- | :--------------------------------- |
| **Query (Q)**        | What each token is searching for   |
| **Key (K)**          | What each token represents         |
| **Value (V)**        | The actual information to retrieve |
| **Attention Scores** | QKᵀ measures similarity            |
| **Scaling**          | Divide by √dₖ for stable gradients |
| **Softmax**          | Convert scores to probabilities    |
| **Output**           | Weighted sum of values             |

---

## What's Next?

A single attention head captures one type of relationship. To capture multiple patterns simultaneously:

➡️ **Next:** [Multi-Head Attention](04-Multi-Head-Attention.md)

---

_Self-attention is what replaced recurrence in Transformers. Its ability to directly connect any two positions regardless of distance revolutionized sequence modeling._
