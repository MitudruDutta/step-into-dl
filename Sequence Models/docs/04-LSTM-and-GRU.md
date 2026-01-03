# 🧠 LSTM & GRU Architectures

**Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** are advanced recurrent architectures designed to solve the vanishing gradient problem and capture long-term dependencies in sequential data.

---

## The Problem They Solve

Standard RNNs suffer from short-term memory — they struggle to maintain information across many time steps due to vanishing gradients.

```
RNN Memory Decay:

Step 1: "The" ────────────────────────────→ Strong signal
Step 5: ──────────────────────────────────→ Weakened
Step 20: ─────────────────────────────────→ Almost gone
Step 50: ─────────────────────────────────→ Forgotten

LSTM/GRU Solution: Create a "highway" for information to flow!
```

---

## LSTM Architecture

### The Key Innovation: Cell State

LSTM introduces a **cell state** — a separate memory track that runs through the entire sequence with minimal modifications:

```
           Cell State (Long-term memory highway)
    ──────────────────────────────────────────────→
            ×           +            ×
            ↑           ↑            ↓
    ┌───────┴───────────┴────────────┴───────┐
    │              LSTM Cell                 │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
    │  │ Forget  │ │  Input  │ │ Output  │   │
    │  │  Gate   │ │  Gate   │ │  Gate   │   │
    │  └────┬────┘ └────┬────┘ └────┬────┘   │
    └───────┼───────────┼───────────┼────────┘
            │           │           │
    ────────┴───────────┴───────────┴─────────
            Hidden State (Short-term memory)
```

### The Three Gates

| Gate            | Symbol | Purpose                         | Output Range |
| :-------------- | :----: | :------------------------------ | :----------: |
| **Forget Gate** |   fₜ   | What to discard from cell state |    [0, 1]    |
| **Input Gate**  |   iₜ   | What new info to add            |    [0, 1]    |
| **Output Gate** |   oₜ   | What to output from cell state  |    [0, 1]    |

### LSTM Equations

```
1. FORGET GATE: What to throw away
   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

2. INPUT GATE: What to update
   iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)

3. CANDIDATE VALUES: New potential values
   C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

4. UPDATE CELL STATE: Forget old + add new
   Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

5. OUTPUT GATE: What to output
   oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

6. HIDDEN STATE: Filtered cell state
   hₜ = oₜ ⊙ tanh(Cₜ)
```

### Step-by-Step Gate Operations

```
STEP 1: FORGET GATE
─────────────────────
"Should I forget this information?"

Cₜ₋₁ = [0.8, -0.5, 0.3]  (previous cell state)
fₜ   = [0.1, 0.9, 0.5]   (forget gate output)

fₜ ⊙ Cₜ₋₁ = [0.08, -0.45, 0.15]

→ First element mostly forgotten (×0.1)
→ Second element kept (×0.9)


STEP 2: INPUT GATE + CANDIDATE
───────────────────────────────
"What new information should I add?"

iₜ   = [0.9, 0.2, 0.7]   (input gate)
C̃ₜ   = [0.5, 0.3, -0.8]  (candidate values)

iₜ ⊙ C̃ₜ = [0.45, 0.06, -0.56]

→ Scales how much of each candidate to add


STEP 3: UPDATE CELL STATE
───────────────────────────
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
   = [0.08, -0.45, 0.15] + [0.45, 0.06, -0.56]
   = [0.53, -0.39, -0.41]


STEP 4: OUTPUT GATE
─────────────────────
"What should I output?"

oₜ = [0.8, 0.3, 0.6]
hₜ = oₜ ⊙ tanh(Cₜ)
   = [0.8, 0.3, 0.6] ⊙ tanh([0.53, -0.39, -0.41])
   = [0.38, -0.11, -0.23]
```

---

## Why LSTM Solves Vanishing Gradients

### The Cell State Highway

```
In standard RNN:
  hₜ = tanh(Wh · hₜ₋₁ + Wx · xₜ)

  Gradient: ∂hₜ/∂hₜ₋₁ involves tanh' (can shrink)

In LSTM:
  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

  Gradient: ∂Cₜ/∂Cₜ₋₁ = fₜ (just multiplication by gate!)

If fₜ ≈ 1: Gradient flows through unchanged!
```

### Gradient Flow Comparison

```
Standard RNN:
  ∂h₁₀₀/∂h₁ = ∏ᵢ (tanh'(zᵢ) · Wₕₕ)  →  Vanishes!

LSTM:
  ∂C₁₀₀/∂C₁ = ∏ᵢ fᵢ  →  If f ≈ 1, gradient preserved!

The forget gate literally controls gradient flow:
  f = 1: "Remember everything, gradient flows"
  f = 0: "Forget completely"
```

---

## GRU Architecture

GRU is a simplified version of LSTM, combining the forget and input gates into a single **update gate**.

### GRU Structure

```
GRU has only TWO gates:

    ┌─────────────────────────────────────┐
    │              GRU Cell               │
    │                                     │
    │   ┌───────────┐   ┌───────────┐    │
    │   │   Reset   │   │  Update   │    │
    │   │   Gate    │   │   Gate    │    │
    │   └─────┬─────┘   └─────┬─────┘    │
    │         │               │          │
    └─────────┼───────────────┼──────────┘
              │               │
    ──────────┴───────────────┴───────────
              Hidden State only
              (No separate cell state)
```

### GRU Equations

```
1. RESET GATE: How much past to use in candidate
   rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)

2. UPDATE GATE: Balance between old and new
   zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)

3. CANDIDATE STATE: Potential new hidden state
   h̃ₜ = tanh(Wh · [rₜ ⊙ hₜ₋₁, xₜ] + bh)

4. NEW HIDDEN STATE: Interpolate old and new
   hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

### Understanding GRU Gates

```
UPDATE GATE (zₜ):
─────────────────
Controls the balance between keeping old state and accepting new

z = 1: "Use new candidate completely"
z = 0: "Keep old hidden state completely"

hₜ = (1 - z) · hₜ₋₁ + z · h̃ₜ
     ↑              ↑
     Old           New


RESET GATE (rₜ):
────────────────
Controls how much of the past to "forget" when creating candidate

r = 1: "Use all of previous hidden state"
r = 0: "Ignore previous hidden state completely"

h̃ₜ = tanh(W · [r ⊙ hₜ₋₁, xₜ])
            ↑
            Reset filters what past info to consider
```

---

## LSTM vs GRU Comparison

| Aspect             | LSTM                         | GRU                             |
| :----------------- | :--------------------------- | :------------------------------ |
| **Gates**          | 3 (forget, input, output)    | 2 (reset, update)               |
| **States**         | Cell state + Hidden state    | Hidden state only               |
| **Parameters**     | More                         | ~25% fewer                      |
| **Training Speed** | Slower                       | Faster                          |
| **Memory**         | Excellent long-term          | Good long-term                  |
| **Best For**       | Complex, very long sequences | Simpler sequences, speed needed |

### When to Choose

```
Choose LSTM when:
  ✓ Very long sequences (100+ steps)
  ✓ Complex dependencies
  ✓ Maximum accuracy is priority
  ✓ Memory/compute is not limited

Choose GRU when:
  ✓ Moderate sequence lengths
  ✓ Faster training needed
  ✓ Limited computational resources
  ✓ Fewer training examples
```

---

## PyTorch Implementation

### Basic LSTM

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(hidden)
```

### Basic GRU

```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, h_n = self.gru(embedded)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(hidden)
```

### Sequence-to-Sequence (Many-to-Many)

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        # Encode source sequence
        _, (h, c) = self.encoder(src)

        # Decode target sequence
        dec_out, _ = self.decoder(tgt, (h, c))

        # Project to output space
        return self.fc(dec_out)
```

---

## Practical Tips

### 1. Layer Stacking

```python
# Stack multiple LSTM layers
self.lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=3,      # 3 stacked layers
    dropout=0.3        # Dropout between layers
)
```

### 2. Bidirectional Processing

```python
# Bidirectional LSTM
self.lstm = nn.LSTM(
    hidden_size=256,
    bidirectional=True  # Forward + backward
)
# Output size becomes hidden_size * 2
```

### 3. Gradient Clipping During Training

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

### 4. Attention Mechanism (Modern Enhancement)

```python
class AttentionLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        context = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context
```

---

## Summary

| Concept            | LSTM                    | GRU                                |
| :----------------- | :---------------------- | :--------------------------------- |
| **Key Innovation** | Cell state highway      | Simplified gating                  |
| **Gates**          | Forget, Input, Output   | Reset, Update                      |
| **Gradient Flow**  | Via cell state          | Via update gate                    |
| **Parameters**     | More                    | Fewer                              |
| **Use Case**       | Complex, long sequences | Faster training, shorter sequences |

---

## What You've Learned

✅ How LSTM's gates control information flow  
✅ How the cell state solves vanishing gradients  
✅ GRU's simplified two-gate architecture  
✅ When to choose LSTM vs GRU  
✅ PyTorch implementations for both

---

_LSTM and GRU revolutionized sequence modeling. While Transformers have become dominant for many NLP tasks, these gated RNNs remain excellent choices for time series, embedded systems, and scenarios requiring lower computational cost._
