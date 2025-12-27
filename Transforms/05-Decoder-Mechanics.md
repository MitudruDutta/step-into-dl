# 🎭 Decoder Mechanics: Masked & Cross-Attention

The decoder uses specialized attention mechanisms to generate text safely and accurately. Understanding **masked self-attention** and **cross-attention** is crucial for grasping how models like GPT generate text.

---

## The Decoder's Challenge

The decoder must generate sequences **autoregressively** (one token at a time) while:

1. Not "cheating" by seeing future tokens during training
2. Incorporating context from the source/input sequence

```text
Generation: "The cat sat on the ___"

At training time:
  ❌ WRONG: Model sees "mat" and trivially copies it
  ✅ RIGHT: Model must predict "mat" from only prior context

Solution: MASKING prevents seeing future tokens
```

---

## Masked Self-Attention

### Why Masking?

```text
During training, we have the full target sequence:
  "The cat sat on the mat"

Without masking - MODEL CHEATS:
  To predict position 5 ("mat"):
  Model sees: "The cat sat on the MAT"
                                  ↑ No learning happens!

With masking - MODEL LEARNS:
  To predict position 5 ("mat"):
  Model sees: "The cat sat on the [MASKED]"
                                  ↑ Must actually predict!
```

### The Causal Mask

```text
Original Attention Matrix (all tokens visible):

        The  cat  sat  on   the  mat
   ┌─────────────────────────────────┐
The│  1    1    1    1    1    1    │
cat│  1    1    1    1    1    1    │
sat│  1    1    1    1    1    1    │
on │  1    1    1    1    1    1    │
the│  1    1    1    1    1    1    │
mat│  1    1    1    1    1    1    │
   └─────────────────────────────────┘

Causal (Masked) Attention Matrix:

        The  cat  sat  on   the  mat
   ┌─────────────────────────────────┐
The│  1   -∞   -∞   -∞   -∞   -∞    │ ← Can only see itself
cat│  1    1   -∞   -∞   -∞   -∞    │ ← Can see The, cat
sat│  1    1    1   -∞   -∞   -∞    │ ← Can see The, cat, sat
on │  1    1    1    1   -∞   -∞    │
the│  1    1    1    1    1   -∞    │
mat│  1    1    1    1    1    1    │ ← Can see everything
   └─────────────────────────────────┘

-∞ becomes 0 after softmax → No attention to future tokens!
```

### Visual Representation

```text
Causal Masking Pattern:

Position:    1    2    3    4    5    6
           ┌────────────────────────────┐
Token 1    │ ✓   ✗   ✗   ✗   ✗   ✗   │
Token 2    │ ✓   ✓   ✗   ✗   ✗   ✗   │
Token 3    │ ✓   ✓   ✓   ✗   ✗   ✗   │
Token 4    │ ✓   ✓   ✓   ✓   ✗   ✗   │
Token 5    │ ✓   ✓   ✓   ✓   ✓   ✗   │
Token 6    │ ✓   ✓   ✓   ✓   ✓   ✓   │
           └────────────────────────────┘

           Lower triangular matrix!
```

---

## Cross-Attention

In encoder-decoder models (like T5, BART), the decoder needs to reference the source sequence. **Cross-attention** enables this.

### How Cross-Attention Differs from Self-Attention

```text
SELF-ATTENTION (within decoder):
  Queries: from decoder tokens
  Keys:    from decoder tokens     ← Same source
  Values:  from decoder tokens

CROSS-ATTENTION (decoder ← encoder):
  Queries: from decoder tokens
  Keys:    from ENCODER output     ← Different source!
  Values:  from ENCODER output
```

### Cross-Attention in Action

```text
Translation: "Hello world" → "Bonjour le monde"

ENCODER processes source:
  "Hello" → h₁ (encoder hidden state)
  "world" → h₂ (encoder hidden state)

DECODER generates target with cross-attention:

Step 1: Generate "Bonjour"
  Query: decoder state for position 1
  Keys/Values: [h₁, h₂] from encoder

  Attention weights:
    "Hello" → 0.8  (high - translating Hello!)
    "world" → 0.2

  Output: Weighted combination → helps predict "Bonjour"

Step 2: Generate "le"
  Query: decoder state for position 2
  Keys/Values: [h₁, h₂] from encoder

  (article "le" might attend broadly)

Step 3: Generate "monde"
  Query: decoder state for position 3
  Keys/Values: [h₁, h₂] from encoder

  Attention weights:
    "Hello" → 0.2
    "world" → 0.8  (high - translating world!)

  Output: Weighted combination → helps predict "monde"
```

### Cross-Attention Visualization

```text
Encoder Output:  "Hello"  "world"
                   ↑  ↑
                   │  │
              ┌────┴──┴────┐
              │ Keys/Values│
              └────────────┘
                   ↑
                   │ Cross-Attention
                   │
              ┌────┴────┐
              │ Queries │
              └─────────┘
                   ↑
Decoder:      "Bonjour" "le" "monde"
```

---

## Complete Decoder Layer

```text
                   DECODER LAYER

    Previous Decoder Hidden States
               │
               ▼
    ┌────────────────────────────────┐
    │   MASKED SELF-ATTENTION       │
    │   (causal: only past tokens)  │
    └────────────────┬───────────────┘
                     │
    ┌────────────────┴───────────────┐
    │        Add & LayerNorm        │
    └────────────────┬───────────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │      CROSS-ATTENTION           │ ← Encoder Output
    │   (Q from decoder,             │
    │    K,V from encoder)           │
    └────────────────┬───────────────┘
                     │
    ┌────────────────┴───────────────┐
    │        Add & LayerNorm        │
    └────────────────┬───────────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │      FEED-FORWARD NETWORK     │
    └────────────────┬───────────────┘
                     │
    ┌────────────────┴───────────────┐
    │        Add & LayerNorm        │
    └────────────────┬───────────────┘
                     │
                     ▼
              Decoder Output
```

---

## Autoregressive Generation Process

```text
Task: Complete "The quick brown"

Step 1:
  Input:  <START> The quick brown
  Mask:   ████████████████████████  (see all)
  Output: Probability distribution
  Select: "fox" (highest probability)

Step 2:
  Input:  <START> The quick brown fox
  Mask:   █████████████████████████████
  Output: Probability distribution
  Select: "jumps"

Step 3:
  Input:  <START> The quick brown fox jumps
  Output: "over"

... continue until <END> token
```

### Decoding Strategies

| Strategy            | Description                     | When to Use               |
| :------------------ | :------------------------------ | :------------------------ |
| **Greedy**          | Always pick highest probability | Fast, deterministic       |
| **Beam Search**     | Track top-k candidates          | Better quality, expensive |
| **Top-k Sampling**  | Sample from top-k tokens        | Creative, diverse         |
| **Top-p (Nucleus)** | Sample from top cumulative p    | Balanced diversity        |
| **Temperature**     | Scale logits before softmax     | Control randomness        |

---

## PyTorch Implementation

### Creating Causal Mask

```python
import torch

def create_causal_mask(seq_len):
    """Create mask for causal (autoregressive) attention."""
    # Upper triangular matrix of -inf
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'),
        diagonal=1
    )
    return mask

# Example
mask = create_causal_mask(5)
print(mask)
# tensor([[0., -inf, -inf, -inf, -inf],
#         [0.,   0., -inf, -inf, -inf],
#         [0.,   0.,   0., -inf, -inf],
#         [0.,   0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.,   0.]])
```

### Decoder Layer with Both Attentions

```python
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None):
        # 1. Masked self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-attention (if encoder output provided)
        if encoder_output is not None:
            attn_out, _ = self.cross_attn(x, encoder_output, encoder_output)
            x = self.norm2(x + self.dropout(attn_out))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x
```

---

## Decoder-Only vs Encoder-Decoder

### Decoder-Only (GPT-style)

```text
No cross-attention needed - just masked self-attention

Input: "The quick brown"
       ↓
   [Masked Self-Attention]
       ↓
   [Feed-Forward]
       ↓
Predict: "fox"

Used by: GPT, GPT-2, GPT-3, LLaMA, Claude
```

### Encoder-Decoder (T5-style)

```text
Cross-attention bridges encoder and decoder

Encoder: "Translate to French: Hello"
              ↓
         [Encoder Stack]
              ↓
         Context Vectors
              ↓
Decoder: <START>
              ↓
         [Masked Self-Attention]
              ↓
         [Cross-Attention] ← Context from encoder
              ↓
         [Feed-Forward]
              ↓
Predict: "Bonjour"

Used by: T5, BART, mT5
```

---

## Summary

| Mechanism                 | Purpose                      | Key Feature                      |
| :------------------------ | :--------------------------- | :------------------------------- |
| **Masked Self-Attention** | Prevent seeing future        | Causal mask (lower triangular)   |
| **Cross-Attention**       | Reference source sequence    | Q from decoder, K/V from encoder |
| **Autoregressive**        | Generate one token at a time | Use previous outputs as input    |
| **Decoding Strategies**   | Select next token            | Greedy, beam search, sampling    |

---

## What's Next?

How do Transformers learn without labeled data? Through self-supervised training:

➡️ **Next:** [Self-Supervised Training](06-Self-Supervised-Training.md)

---

_Masked and cross-attention are what enable Transformers to generate coherent, contextually relevant text. Understanding these mechanisms is key to building and using generative models._
