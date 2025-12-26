# 🔄 Recurrent Neural Networks (RNN)

Recurrent Neural Networks are the foundation of sequential processing in deep learning. They are designed to maintain a **hidden state** that acts as memory, allowing information to persist and influence future predictions as the network processes a sequence step by step.

---

## The Core Idea

The fundamental innovation of RNNs is the **recurrent connection** — a loop that allows information to flow from one step to the next:

```
Traditional Feedforward:                 Recurrent Neural Network:

  x → [Layer] → y                        ┌────────────────┐
                                         │                ↓
  No memory between inputs               │    ┌───────────────┐
                                         │    │      RNN      │
                                         │    │     Cell      │
                                         │    └───────┬───────┘
                                         │            │
                                         └────────────┘

                                         Hidden state loops back!
```

---

## RNN Architecture

### Unrolled View

When we "unroll" an RNN across time, we can see how information flows through the network:

```
Unrolled RNN across 4 time steps:

        x₁          x₂          x₃          x₄
         ↓           ↓           ↓           ↓
      ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
h₀ →  │ RNN  │→h₁│ RNN  │→h₂│ RNN  │→h₃│ RNN  │→ h₄
      │ Cell │   │ Cell │   │ Cell │   │ Cell │
      └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
         ↓           ↓           ↓           ↓
        y₁          y₂          y₃          y₄

Key: Same RNN Cell (weights) is used at every step!
```

### Inside the RNN Cell

```
┌─────────────────────────────────────────────────┐
│                    RNN Cell                    │
│                                                │
│    ┌─────────┐                                 │
│    │   xₜ    │ Input at time t                 │
│    └────┬────┘                                 │
│         │                                      │
│         ↓                                      │
│    ┌─────────────────────────────┐              │
│    │  Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + b  │              │
│    └──────────────┬──────────────┘              │
│                   │                             │
│                   ↓                             │
│             ┌──────────┐                        │
│             │   tanh   │ Activation             │
│             └────┬─────┘                        │
│                  │                              │
│                  ↓                              │
│             ┌─────────┐                         │
│             │   hₜ    │ New hidden state        │
│             └────┬────┘                         │
│                  │                              │
│         ┌────────┴────────┐                     │
│         ↓                 ↓                     │
│    ┌─────────┐      ┌──────────┐                │
│    │   yₜ    │      │ To next  │               │
│    │ (output)│      │   step   │               │
│    └─────────┘      └──────────┘                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Forward Pass Equations

The forward pass of an RNN consists of two main computations:

```
1. Hidden State Update:
   hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)

2. Output Computation:
   yₜ = Wₕᵧ · hₜ + bᵧ

Where:
   xₜ   ∈ ℝᵈ      Input vector at time t (d = input dimension)
   hₜ   ∈ ℝʰ      Hidden state at time t (h = hidden dimension)
   yₜ   ∈ ℝᵒ      Output at time t (o = output dimension)

   Wₓₕ  ∈ ℝʰˣᵈ    Input-to-hidden weights
   Wₕₕ  ∈ ℝʰˣʰ    Hidden-to-hidden weights (recurrent weights)
   Wₕᵧ  ∈ ℝᵒˣʰ    Hidden-to-output weights
   bₕ   ∈ ℝʰ      Hidden bias
   bᵧ   ∈ ℝᵒ      Output bias
```

### Step-by-Step Example

```
Let's trace through a sequence of length 3:

Initial state: h₀ = [0, 0] (zeros)

Step 1 (t=1):
  Input: x₁ = "The" (embedded as vector)
  h₁ = tanh(Wₓₕ · x₁ + Wₕₕ · h₀ + bₕ)
       = tanh(Wₓₕ · x₁ + Wₕₕ · [0,0] + bₕ)
       = tanh(Wₓₕ · x₁ + bₕ)  # h₀ is zeros
  y₁ = Wₕᵧ · h₁ + bᵧ

Step 2 (t=2):
  Input: x₂ = "cat"
  h₂ = tanh(Wₓₕ · x₂ + Wₕₕ · h₁ + bₕ)
       # Now h₁ contains information about "The"
  y₂ = Wₕᵧ · h₂ + bᵧ

Step 3 (t=3):
  Input: x₃ = "sat"
  h₃ = tanh(Wₓₕ · x₃ + Wₕₕ · h₂ + bₕ)
       # h₂ contains information about "The cat"
  y₃ = Wₕᵧ · h₃ + bᵧ

Final hidden state h₃ encodes the entire sequence!
```

---

## Backpropagation Through Time (BPTT)

Training RNNs requires a special form of backpropagation called **Backpropagation Through Time (BPTT)**:

### The Challenge

```
To compute gradients, we must trace back through ALL time steps:

Loss at t=4:  L₄ = loss(y₄, target₄)

To update Wₕₕ, we need:
  ∂L₄/∂Wₕₕ = ∂L₄/∂y₄ · ∂y₄/∂h₄ · ∂h₄/∂Wₕₕ

But h₄ depends on h₃, which depends on h₂, which depends on h₁...

  h₄ → h₃ → h₂ → h₁ → h₀

We must sum gradients across ALL these dependencies!
```

### BPTT Algorithm

```
BPTT Process:

1. FORWARD PASS: Compute all hidden states and outputs
   h₁ → h₂ → h₃ → ... → hₜ
   y₁    y₂    y₃    ...   yₜ

2. COMPUTE LOSS: Sum losses across all time steps
   L = L₁ + L₂ + L₃ + ... + Lₜ

3. BACKWARD PASS: Propagate gradients back through time
   ∂L/∂hₜ → ∂L/∂hₜ₋₁ → ... → ∂L/∂h₁

4. ACCUMULATE GRADIENTS: Sum gradients for shared weights
   ∂L/∂Wₕₕ = Σₜ (∂Lₜ/∂Wₕₕ)
```

### Gradient Flow Visualization

```
Forward:           Backward (BPTT):

x₁ → h₁ → y₁       δ₁ ← δ₁ ← ∂L₁
      ↓                 ↑
x₂ → h₂ → y₂       δ₂ ← δ₂ ← ∂L₂
      ↓                 ↑
x₃ → h₃ → y₃       δ₃ ← δ₃ ← ∂L₃
      ↓                 ↑
x₄ → h₄ → y₄       δ₄ ← ──── ∂L₄

Gradients flow backward and ACCUMULATE through each step
```

---

## RNN Architectural Variants

### One-to-Many

```
Used for: Image captioning, music generation

Architecture:
         x (single input, e.g., image features)
         ↓
      ┌──────┐
      │ RNN  │→ h₁ → y₁ ("A")
      └──────┘    ↓
                  │
      ┌──────────────┐
      │     RNN      │→ h₂ → y₂ ("cat")
      └──────────────┘    ↓
                          │
      ┌──────────────────────┐
      │         RNN          │→ h₃ → y₃ ("sitting")
      └──────────────────────┘

Output: Sequence of words describing the image
```

### Many-to-One

```
Used for: Sentiment analysis, document classification

Architecture:
    x₁        x₂        x₃        x₄
("This")  ("movie") ("is")   ("great")
     ↓         ↓         ↓         ↓
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ RNN  │→│ RNN  │→│ RNN  │→│ RNN  │
  └──────┘ └──────┘ └──────┘ └──┬───┘
                                 ↓
                            ┌────────┐
                            │ Dense  │
                            └────┬───┘
                                 ↓
                               y = 😊 Positive

Only the FINAL hidden state produces output
```

### Many-to-Many (Synchronized)

```
Used for: Part-of-speech tagging, named entity recognition

Architecture:
    x₁         x₂         x₃
  ("The")   ("cat")    ("sat")
     ↓          ↓          ↓
  ┌──────┐  ┌──────┐  ┌──────┐
  │ RNN  │→ │ RNN  │→ │ RNN  │
  └──┬───┘  └──┬───┘  └──┬───┘
     ↓          ↓          ↓
   y₁=DET    y₂=NOUN   y₃=VERB

Output at EVERY time step (same length as input)
```

### Many-to-Many (Encoder-Decoder)

```
Used for: Machine translation, text summarization

Architecture:
         ENCODER                    DECODER
    x₁      x₂      x₃         y₁      y₂      y₃
 ("Hello")("world")(<EOS>)   (<SOS>)("Bonjour")("monde")
     ↓       ↓       ↓           ↓       ↓       ↓
  ┌─────┐ ┌─────┐ ┌─────┐    ┌─────┐ ┌─────┐ ┌─────┐
  │ RNN │→│ RNN │→│ RNN │→→→→│ RNN │→│ RNN │→│ RNN │
  └─────┘ └─────┘ └─────┘    └──┬──┘ └──┬──┘ └──┬──┘
                                ↓       ↓       ↓
                            "Bonjour" "monde" <EOS>

Encoder processes input → Context vector → Decoder generates output
```

---

## Bidirectional RNN

Sometimes, context from **both directions** is important:

```
Example: "The _____ barked loudly"
  - Forward context: "The" (could be many things)
  - Backward context: "barked" (must be a dog!)

Bidirectional RNN Architecture:

Forward:  x₁ ──→ h₁→ ──→ h₂→ ──→ h₃→

Backward: x₁ ←── h₁← ←── h₂← ←── h₃←

Combined: [h₁→; h₁←]  [h₂→; h₂←]  [h₃→; h₃←]
               ↓           ↓           ↓
              y₁          y₂          y₃

Each output has access to BOTH past and future context
```

---

## PyTorch Implementation

### Basic RNN

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input shape: (batch, seq_len, features)
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # x shape: (batch_size, seq_length, input_size)

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward through RNN
        # out: (batch, seq_len, hidden_size) - all hidden states
        # hn: (num_layers, batch, hidden_size) - final hidden state
        out, hn = self.rnn(x, h0)

        # Take output from last time step for classification
        out = self.fc(out[:, -1, :])  # Many-to-one

        return out, hn

# Example usage
model = SimpleRNN(input_size=10, hidden_size=64, output_size=5)
x = torch.randn(32, 20, 10)  # batch=32, seq_len=20, features=10
output, hidden = model(x)
print(f"Output shape: {output.shape}")  # (32, 5)
```

### RNN for Text Classification

```python
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()

        # Embedding layer: converts word indices to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True  # Use both directions
        )

        # Output layer (hidden_size * 2 for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len) - word indices

        # Convert indices to embeddings
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Forward through RNN
        output, hidden = self.rnn(embedded)

        # For bidirectional: concatenate final hidden states
        # hidden shape: (2, batch, hidden_size)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Classification
        out = self.fc(hidden)
        return out

# Example
model = TextRNN(vocab_size=10000, embedding_dim=128, hidden_size=256, num_classes=2)
x = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 2)
```

---

## Limitations of Basic RNNs

### 1. Vanishing Gradients

```
In long sequences, gradients become exponentially small:

∂h₁₀₀/∂h₁ = ∂h₁₀₀/∂h₉₉ · ∂h₉₉/∂h₉₈ · ... · ∂h₂/∂h₁

If each term < 1:
  0.9 × 0.9 × ... × 0.9 (100 times) ≈ 0.000027

Gradient virtually disappears! Early layers barely update.
```

### 2. Short-Term Memory

```
RNNs struggle to remember information from many steps ago:

"The cat, which had been sitting on the windowsill watching
 birds fly by for the past hour, finally ___"

By the time we reach "finally ___", the RNN may have
"forgotten" that the subject is "cat" (many steps ago).
```

### 3. Difficulty with Long-Range Dependencies

```
Example: Language Modeling

"I grew up in France ... [100 words later] ... I speak fluent ___"

The model should predict "French" based on "France"
But that context is 100+ steps in the past
Basic RNNs cannot maintain this information
```

---

## Summary

| Concept          | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| **Hidden State** | Internal memory that carries information through time        |
| **Recurrence**   | Same weights applied at every time step                      |
| **BPTT**         | Backpropagation that traces gradients through all time steps |
| **Variants**     | One-to-Many, Many-to-One, Many-to-Many, Bidirectional        |
| **Limitations**  | Vanishing gradients, short-term memory                       |

---

## What's Next?

The limitations of basic RNNs led to a critical problem that needed solving:

➡️ **Next:** [The Vanishing Gradient Problem](03-Vanishing-Gradient-Problem.md)

---

_RNNs introduced the revolutionary concept of neural memory. While they have limitations, understanding them is essential before learning how LSTM and GRU architectures solve these problems._
