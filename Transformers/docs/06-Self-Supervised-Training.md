# 📚 Self-Supervised Training

Transformers achieve remarkable capabilities by learning from massive amounts of **unlabeled text**. This is possible through **Self-Supervised Learning**, where the training labels are automatically generated from the data itself.

---

## What is Self-Supervised Learning?

```text
Traditional Supervised Learning:
  Input: "This movie is great!"
  Label: Positive (✋ Human-provided)

  Problem: Expensive, limited scale, requires expertise

Self-Supervised Learning:
  Input: "The cat sat on the ___"
  Label: "mat" (🤖 Auto-generated from data!)

  Advantage: Free labels, unlimited scale, learns general knowledge
```

---

## Why Self-Supervised?

| Aspect          | Supervised         | Self-Supervised       |
| :-------------- | :----------------- | :-------------------- |
| **Labels**      | Human-annotated    | Auto-generated        |
| **Scale**       | Expensive to scale | Unlimited             |
| **Data Source** | Curated datasets   | Raw text (web, books) |
| **Cost**        | High (human labor) | Low (compute only)    |
| **Knowledge**   | Task-specific      | General purpose       |

### The Scale Advantage

```text
GPT-3 Training Data:

  Source                   Tokens (billions)
  ─────────────────────────────────────────
  Common Crawl (filtered)     410
  WebText2                     19
  Books1                       12
  Books2                       55
  Wikipedia                     3
  ─────────────────────────────────────────
  TOTAL                       ~500 billion tokens

Impossible with human labels!
```

---

## Training Approach 1: Causal Language Modeling (CLM)

### Used by: GPT, GPT-2, GPT-3, GPT-4, LLaMA, Claude

Predict the next token given all previous tokens.

### How CLM Works

```text
Training sentence: "The quick brown fox jumps"

Creates these training examples automatically:

  Input                    Target
  ─────────────────────────────────
  [START]               →  The
  [START] The           →  quick
  [START] The quick     →  brown
  [START] The quick brown →  fox
  [START] The quick brown fox → jumps
```

### CLM Visualization

```text
Input:   [START]  The  quick  brown  fox
              \    \     \      \     \
               \    \     \      \     \
                ▼    ▼     ▼      ▼     ▼
Predict:       The  quick brown  fox  jumps

Left-to-right prediction (autoregressive)
```

### Training Objective

```text
Maximize: P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)

Loss = -∑ log P(xᵢ | x₁, ..., xᵢ₋₁)

For each position, predict next token using cross-entropy loss
```

### CLM Characteristics

| Property      | Description                       |
| :------------ | :-------------------------------- |
| **Direction** | Unidirectional (left-to-right)    |
| **Best For**  | Text generation, completion, chat |
| **Context**   | Only past tokens visible          |
| **Inference** | Natural for generation            |

---

## Training Approach 2: Masked Language Modeling (MLM)

### Used by: BERT, RoBERTa, ALBERT, ELECTRA

Randomly mask some tokens and predict them.

### How MLM Works

```text
Original: "The quick brown fox jumps over the lazy dog"

Step 1: Randomly select 15% of tokens for prediction
        Selected: "quick", "fox", "lazy"

Step 2: Apply masking strategy:
        - 80% → [MASK] token
        - 10% → random word
        - 10% → unchanged

        Input: "The [MASK] brown fox jumps over the [MASK] dog"
                     ↑                            ↑
                   predict                     predict
```

### MLM Visualization

```text
Original:   The  quick  brown  fox  jumps  over  the  lazy  dog

Masked:     The  [MASK] brown  fox  jumps  over  the  [MASK] dog
                   ↓                                    ↓
                   ▼                                    ▼
Predict:        "quick"                             "lazy"

Bidirectional context used for prediction!
```

### Why the 80-10-10 Split?

```text
Masking Strategy Rationale:

80% [MASK]:  Standard masking - model learns to predict
10% Random:  Prevents model from only focusing on [MASK]
10% Same:    Model must also verify correct tokens

Without this:
  Model might learn: "If I see [MASK], make a prediction"
  And ignore: "If I don't see [MASK], don't think"

With this:
  Model learns: "Always understand the context"
```

### MLM Characteristics

| Property      | Description                             |
| :------------ | :-------------------------------------- |
| **Direction** | Bidirectional (full context)            |
| **Best For**  | Understanding, classification, NER      |
| **Context**   | Entire sequence visible (except masked) |
| **Inference** | Not directly usable for generation      |

---

## Comparing CLM vs MLM

```text
CLM (GPT-style):           MLM (BERT-style):
←←←←←←←←                   ←→←→←→←→←→

[The] → predict next       [The ████ brown] → fill masks
  Unidirectional              Bidirectional

Good for:                  Good for:
• Generation               • Classification
• Completion               • Question Answering
• Chat                     • Named Entity Recognition
• Creative writing         • Semantic similarity
```

### Side-by-Side Comparison

| Aspect            | CLM (GPT)          | MLM (BERT)             |
| :---------------- | :----------------- | :--------------------- |
| **Training**      | Predict next token | Predict masked tokens  |
| **Context**       | Left-only          | Full bidirectional     |
| **Generation**    | Natural            | Requires modifications |
| **Understanding** | Good               | Excellent              |
| **Typical Use**   | Chatbots, writing  | Classification, search |

---

## Other Self-Supervised Objectives

### Next Sentence Prediction (NSP)

### Used by: Original BERT

```text
Task: Is sentence B the actual next sentence after A?

Example 1 (IsNext = True):
  A: "The cat sat on the mat."
  B: "It was very comfortable."

Example 2 (NotNext = False):
  A: "The cat sat on the mat."
  B: "The stock market rose today."

Training: 50% correct pairs, 50% random pairs
```

### Sentence Order Prediction (SOP)

### Used by: ALBERT

```text
Task: Are sentences in correct order?

Correct:
  A: "He went to the store."
  B: "He bought some milk."

Swapped:
  A: "He bought some milk."
  B: "He went to the store."
```

### Replaced Token Detection (RTD)

### Used by: ELECTRA

```text
Step 1: Small "generator" model fills in [MASK] tokens
Step 2: Larger "discriminator" detects which tokens were replaced

Original:   "The chef cooked dinner"
Generator:  "The chef [MASK] dinner" → "The chef made dinner"
                            ↑ replaced with plausible word

Discriminator task: Identify "made" is not original

More efficient than MLM - learns from all tokens, not just masked!
```

---

## Pre-training → Fine-tuning Paradigm

```text
┌─────────────────────────────────────────────────────────────┐
│                    PRE-TRAINING                              │
│                                                             │
│  Massive unlabeled text (books, web, wikipedia)             │
│                      ↓                                      │
│  Self-supervised learning (CLM or MLM)                      │
│                      ↓                                      │
│  Learn general language understanding                       │
│                                                             │
│  Cost: Millions of dollars, weeks of compute                │
│  Result: Foundation model with general knowledge            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    FINE-TUNING                               │
│                                                             │
│  Small labeled dataset for specific task                    │
│                      ↓                                      │
│  Supervised learning on task (classification, etc.)         │
│                      ↓                                      │
│  Adapt to specific domain/task                              │
│                                                             │
│  Cost: Hours to days, single GPU                            │
│  Result: Task-specific model                                │
└─────────────────────────────────────────────────────────────┘
```

### Fine-tuning Examples

```text
Pre-trained BERT →  Fine-tune on:
                    ├── Sentiment Analysis (movie reviews)
                    ├── Named Entity Recognition (news articles)
                    ├── Question Answering (SQuAD dataset)
                    └── Spam Detection (email dataset)

Pre-trained GPT →   Fine-tune on:
                    ├── Code Generation (GitHub code)
                    ├── Customer Support (chat logs)
                    └── Creative Writing (stories)
```

---

## PyTorch Training Example

### CLM Training Loop

```python
import torch
import torch.nn as nn

def train_clm_step(model, batch, optimizer):
    """One training step for Causal Language Modeling."""

    # batch shape: (batch_size, seq_len)
    input_ids = batch[:, :-1]   # All tokens except last
    labels = batch[:, 1:]        # All tokens except first

    # Forward pass
    logits = model(input_ids)    # (batch, seq_len-1, vocab_size)

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),  # Flatten
        labels.view(-1)
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### MLM Training with Masking

```python
def create_mlm_batch(input_ids, vocab_size, mask_token_id, mask_prob=0.15):
    """Create masked input and labels for MLM."""

    labels = input_ids.clone()

    # Create probability matrix for masking
    prob_matrix = torch.full(input_ids.shape, mask_prob)

    # Decide which tokens to mask
    masked_indices = torch.bernoulli(prob_matrix).bool()

    # Set non-masked labels to -100 (ignored in loss)
    labels[~masked_indices] = -100

    # 80% of time: replace with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8)
    ).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% of time: replace with random token
    indices_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, input_ids.shape)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% of time: keep original (nothing to do)

    return input_ids, labels
```

---

## Training Scale

| Model       | Parameters | Training Tokens | Training Cost |
| :---------- | :--------- | :-------------- | :------------ |
| BERT-Base   | 110M       | 3.3B            | ~$10K         |
| GPT-2       | 1.5B       | 40B             | ~$50K         |
| GPT-3       | 175B       | 500B            | ~$4.6M        |
| LLaMA 2-70B | 70B        | 2T              | ~$2M          |

---

## Summary

| Concept             | Description                               |
| :------------------ | :---------------------------------------- |
| **Self-Supervised** | Labels generated automatically from data  |
| **CLM**             | Predict next token (GPT-style)            |
| **MLM**             | Predict masked tokens (BERT-style)        |
| **Pre-training**    | Learn general knowledge from massive data |
| **Fine-tuning**     | Adapt to specific tasks with small data   |
| **Scale**           | Enables training on billions of tokens    |

---

## What's Next?

Now you understand the complete Transformer pipeline! For hands-on practice:

➡️ **Practice:** Explore the notebooks in this directory:

- `BERT.ipynb` - Encoder-only model
- `GPT2.ipynb` - Decoder-only model
- `spam_classification.ipynb` - Real-world application

---

_Self-supervised learning is what unlocked the era of large language models. By creating training signals from raw text, we can leverage the entirety of human written knowledge._
