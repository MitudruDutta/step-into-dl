# 🔀 Multi-Head Attention

A single attention head captures one type of relationship. **Multi-Head Attention** runs multiple attention operations in parallel, allowing the model to jointly attend to information from different representation subspaces.

---

## Why Multiple Heads?

```text
Single Attention Head:
  Can only capture ONE type of pattern at a time

Multi-Head Attention:
  Multiple heads capture DIFFERENT patterns simultaneously!

┌─────────────────────────────────────────────┐
│ Head 1: Syntactic (subject-verb agreement)  │
│ Head 2: Semantic (related concepts)         │
│ Head 3: Positional (adjacent words)         │
│ Head 4: Coreference (pronoun-antecedent)    │
│ ...                                         │
└─────────────────────────────────────────────┘
```

---

## How Multi-Head Attention Works

```text
                MULTI-HEAD ATTENTION

Input X ─┬──► Head 1 (Q₁,K₁,V₁) ──► Attention₁ ─┐
         │                                       │
         ├──► Head 2 (Q₂,K₂,V₂) ──► Attention₂ ─┤
         │                                       ├──► Concat ──► Linear ──► Output
         ├──► Head 3 (Q₃,K₃,V₃) ──► Attention₃ ─┤
         │                                       │
         └──► Head h (Qₕ,Kₕ,Vₕ) ──► Attentionₕ ─┘
```

### Step-by-Step Process

```text
1. PROJECT: Split input into h different subspaces

   Input: X (batch, seq, d_model=512)

   For each head i (with d_k = d_model/h = 64):
     Qᵢ = X × Wᵢᵠ   →  (batch, seq, 64)
     Kᵢ = X × Wᵢᴷ   →  (batch, seq, 64)
     Vᵢ = X × Wᵢⱽ   →  (batch, seq, 64)

2. COMPUTE: Each head computes attention independently

   headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)
         = softmax(QᵢKᵢᵀ / √dₖ) × Vᵢ

3. CONCATENATE: Join all head outputs

   MultiHead = Concat(head₁, head₂, ..., headₕ)
             = (batch, seq, h×d_k) = (batch, seq, 512)

4. PROJECT: Final linear transformation

   Output = MultiHead × Wᴼ
          = (batch, seq, d_model)
```

---

## Mathematical Formulation

```text
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × Wᴼ

where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)

Dimensions:
  Wᵢᵠ ∈ ℝ^(d_model × d_k)    # Query projection
  Wᵢᴷ ∈ ℝ^(d_model × d_k)    # Key projection
  Wᵢⱽ ∈ ℝ^(d_model × d_v)    # Value projection
  Wᴼ  ∈ ℝ^(h×d_v × d_model)  # Output projection

  d_k = d_v = d_model / h
```

---

## What Different Heads Learn

### Observed Attention Patterns

```text
Example: "The cat that I saw yesterday ran away"

HEAD 1 - Syntactic Structure:
  "cat" ←────────────────────→ "ran"
  (Subject attends to its verb)

HEAD 2 - Local Context:
  "cat" ← "The" → "that"
  (Adjacent word attention)

HEAD 3 - Relative Clauses:
  "that" ←→ "I" ←→ "saw"
  (Relative clause connections)

HEAD 4 - Long-Range:
  "cat" ←──────────────────→ "ran" ←→ "away"
  (Action and result)
```

### Attention Heatmaps per Head

```text
HEAD 1 (Syntactic):     HEAD 2 (Local):      HEAD 3 (Semantic):
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│ ░ ░ ░ ░ █ ░ ░ │    │ █ ▓ ░ ░ ░ ░ ░ │    │ ░ ░ ░ ░ ░ ░ █ │
│ ░ ░ ░ ░ ░ █ ░ │    │ ▓ █ ▓ ░ ░ ░ ░ │    │ ░ ░ ░ ░ █ ░ ░ │
│ ░ ░ ░ ░ ░ ░ █ │    │ ░ ▓ █ ▓ ░ ░ ░ │    │ ░ ░ ░ ░ ░ █ ░ │
└────────────────┘    └────────────────┘    └────────────────┘
 Long-range deps       Diagonal pattern      Semantic links
```

---

## Benefits of Multi-Head Attention

| Benefit                     | Explanation                                   |
| :-------------------------- | :-------------------------------------------- |
| **Diverse Relationships**   | Different heads capture different patterns    |
| **Richer Representations**  | Combined output has multiple perspectives     |
| **Parallel Processing**     | All heads compute simultaneously              |
| **Long-Range Dependencies** | At least one head can focus on distant tokens |
| **Robustness**              | If one head fails, others compensate          |

---

## Typical Configurations

| Model       | d_model | Heads | d_k = d_v |
| :---------- | :------ | :---- | :-------- |
| BERT-Base   | 768     | 12    | 64        |
| BERT-Large  | 1024    | 16    | 64        |
| GPT-2 Small | 768     | 12    | 64        |
| GPT-2 Large | 1280    | 20    | 64        |
| GPT-3       | 12288   | 96    | 128       |

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for all heads (combined)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape to (batch, heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights

# Example usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq=10
output, weights = mha(x, x, x)
print(f"Output: {output.shape}")   # (2, 10, 512)
print(f"Weights: {weights.shape}") # (2, 8, 10, 10)
```

---

## Visualization of Multi-Head Outputs

```text
Sentence: "The quick brown fox jumps"

             Head 1    Head 2    Head 3    Head 4
           ┌────────┬────────┬────────┬────────┐
    "The"  │ [0.2,  │ [0.1,  │ [0.3,  │ [0.0,  │
           │  0.8,  │  0.5,  │  0.2,  │  0.9,  │
           │  ...]  │  ...]  │  ...]  │  ...]  │
           ├────────┼────────┼────────┼────────┤
   "quick" │ [0.5,  │ [0.3,  │ [0.1,  │ [0.4,  │
           │  ...]  │  ...]  │  ...]  │  ...]  │
           └────────┴────────┴────────┴────────┘
                              ↓
                      CONCATENATE
                              ↓
             ┌────────────────────────────────┐
    "The"    │ [0.2, 0.8, ..., 0.1, 0.5, ...] │
    "quick"  │ [0.5, ..., 0.3, ..., 0.1, ...] │
             └────────────────────────────────┘
                              ↓
                      LINEAR (Wᴼ)
                              ↓
                        Final Output
```

---

## Summary

| Concept               | Description                                   |
| :-------------------- | :-------------------------------------------- |
| **Multiple Heads**    | Run h attention operations in parallel        |
| **Subspaces**         | Each head works in d_model/h dimensions       |
| **Diverse Patterns**  | Different heads learn different relationships |
| **Concatenation**     | Combine all head outputs                      |
| **Output Projection** | Final linear layer mixes head information     |

---

## What's Next?

The decoder uses specialized attention mechanisms for safe text generation:

➡️ **Next:** [Decoder Mechanics: Masked & Cross-Attention](05-Decoder-Mechanics.md)

---

_Multi-head attention is what makes Transformers so powerful. By attending to information in multiple ways simultaneously, the model builds a rich, multi-faceted understanding of input sequences._
