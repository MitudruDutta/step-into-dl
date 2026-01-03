# 🏗️ Architecture Overview: Encoder and Decoder

The Transformer architecture consists of two main components: the **Encoder** and the **Decoder**. Understanding how these work is crucial for modern NLP.

---

## The Original Transformer Architecture

```text
                    THE TRANSFORMER

         ENCODER                    DECODER
    ┌─────────────────┐       ┌─────────────────┐
    │ Multi-Head      │       │ Masked Multi-   │
    │ Self-Attention  │       │ Head Attention  │
    │       ↓         │       │       ↓         │
    │ Add & Norm      │       │ Add & Norm      │
    │       ↓         │       │       ↓         │
    │ Feed Forward    │──────►│ Cross-Attention │
    │       ↓         │Context│       ↓         │
    │ Add & Norm      │       │ Add & Norm      │
    │                 │       │       ↓         │
    │     × N         │       │ Feed Forward    │
    └────────┬────────┘       │       ↓         │
             │                │ Add & Norm      │
         Input Tokens         │     × N         │
                              └────────┬────────┘
                                       │
                                  Output Tokens
```

---

## The Encoder: Understanding Context

The encoder creates **contextual representations** of the input. It reads the entire input at once and understands how each token relates to every other.

### Key Properties

| Property                | Description                                               |
| :---------------------- | :-------------------------------------------------------- |
| **Bidirectional**       | Each token attends to all other tokens (left and right)   |
| **Parallel Processing** | All tokens processed simultaneously                       |
| **Contextual Output**   | Same word gets different representations based on context |

### Contextualization Example

```text
"bank" in different contexts:

Sentence 1: "I deposited money at the bank"
  "bank" attends to → "deposited", "money"
  Result: bank → [financial-context embedding]

Sentence 2: "The river bank was muddy"
  "bank" attends to → "river", "muddy"
  Result: bank → [geographical-context embedding]

Same word "bank" → DIFFERENT embeddings!
```

---

## The Decoder: Generating Output

The decoder generates output tokens one at a time, using previous outputs and encoder representations.

### Key Properties

| Property            | Description                   |
| :------------------ | :---------------------------- |
| **Autoregressive**  | Generates one token at a time |
| **Causal Masking**  | Cannot see future tokens      |
| **Cross-Attention** | References encoder output     |

### Autoregressive Generation

```text
Translation: "Hello world" → "Bonjour le monde"

Step 1: <START>              → "Bonjour"
Step 2: <START> Bonjour      → "le"
Step 3: <START> Bonjour le   → "monde"
Step 4: <START> Bonjour le monde → <END>
```

---

## Architecture Variants

### Encoder-Only (BERT, RoBERTa)

```text
Best for: Understanding & classification

    [CLS] Token₁ Token₂ ... TokenN
              ↓
        Encoder Stack
              ↓
    Classification / Token Labels

Use cases: Sentiment analysis, NER, Q&A
```

### Decoder-Only (GPT, LLaMA)

```text
Best for: Text generation

    "The capital of France is"
              ↓
        Decoder Stack
              ↓
    Predict next: "Paris"

Use cases: Chatbots, completion, creative writing
```

### Encoder-Decoder (T5, BART)

```text
Best for: Sequence-to-sequence

    "Translate: Hello world"
              ↓
          ENCODER
              ↓
         Context
              ↓
          DECODER
              ↓
    "Bonjour le monde"

Use cases: Translation, summarization
```

### Comparison

| Architecture        | Attention     | Best For      | Examples      |
| :------------------ | :------------ | :------------ | :------------ |
| **Encoder-Only**    | Bidirectional | Understanding | BERT, RoBERTa |
| **Decoder-Only**    | Causal        | Generation    | GPT, LLaMA    |
| **Encoder-Decoder** | Both          | Seq2Seq       | T5, BART      |

---

## Layer Components

### Multi-Head Attention

```text
Input X
   ├──► Head 1 → Attention₁
   ├──► Head 2 → Attention₂
   └──► Head 8 → Attention₈
              ↓
        Concatenate → Linear → Output
```

### Feed-Forward Network

```text
x → Linear (768→3072) → ReLU → Linear (3072→768) → output
```

### Residual + LayerNorm

```text
output = LayerNorm(x + Sublayer(x))
```

---

## Model Dimensions

| Model     | Layers | d_model | Heads | Parameters |
| :-------- | :----- | :------ | :---- | :--------- |
| BERT-Base | 12     | 768     | 12    | 110M       |
| GPT-2     | 12     | 768     | 12    | 117M       |
| GPT-3     | 96     | 12288   | 96    | 175B       |

---

## Summary

| Component    | Purpose                    | Key Feature              |
| :----------- | :------------------------- | :----------------------- |
| **Encoder**  | Contextual representations | Bidirectional            |
| **Decoder**  | Generate sequence          | Causal + cross-attention |
| **FFN**      | Non-linearity              | Position-wise            |
| **Residual** | Deep networks              | Gradient flow            |

---

## What's Next?

➡️ **Next:** [The Attention Mechanism](03-Attention-Mechanism.md)

---

_The encoder-decoder architecture provides the foundation. The real magic lies in the attention mechanism._
