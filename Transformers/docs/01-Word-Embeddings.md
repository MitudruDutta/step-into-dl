# 📝 Word Embeddings: The Semantic Foundation

Word embeddings are the fundamental building blocks that allow neural networks to understand and process human language. Before a Transformer or any NLP model can work with text, it must convert words into a numerical format that captures their semantic meaning.

---

## What Are Word Embeddings?

Word embeddings are **dense vector representations** of words where semantically similar words are mapped to nearby points in a high-dimensional vector space. Unlike simple one-hot encoding (where each word is independent), embeddings capture the _meaning_ and _relationships_ between words.

### The Problem with One-Hot Encoding

```text
Traditional One-Hot Encoding:

Vocabulary: [cat, dog, king, queen, man, woman]

cat   = [1, 0, 0, 0, 0, 0]
dog   = [0, 1, 0, 0, 0, 0]
king  = [0, 0, 1, 0, 0, 0]
queen = [0, 0, 0, 1, 0, 0]
man   = [0, 0, 0, 0, 1, 0]
woman = [0, 0, 0, 0, 0, 1]

Problems:
❌ Every word is equally distant from every other word
❌ cosine_similarity(cat, dog) = 0  (but they're both animals!)
❌ cosine_similarity(king, queen) = 0  (but they're both royalty!)
❌ Vectors are HUGE (vocabulary size can be 50,000+)
❌ No semantic information captured
```

### The Embedding Solution

```text
Word Embeddings (dense vectors):

cat   = [0.2, -0.4, 0.7, 0.1]     ┐
dog   = [0.3, -0.3, 0.6, 0.2]     ├─ Close in vector space (animals)
                                  ┘
king  = [0.9, 0.1, -0.2, 0.8]    ┐
queen = [0.8, 0.2, -0.1, 0.9]    ├─ Close in vector space (royalty)
                                  ┘

Benefits:
✅ Similar words have similar vectors
✅ Compact representation (100-1024 dimensions vs 50,000+)
✅ Semantic relationships are encoded
✅ Enable mathematical operations on meaning
```

---

## Key Properties of Word Embeddings

### 1. Dimensionality

Embeddings typically have 100-1024 dimensions. Each dimension captures some abstract feature of meaning.

```text
Example: 300-dimensional vector for "king"

Dimension 1:  0.42  → might relate to "royalty"
Dimension 2: -0.15  → might relate to "gender"
Dimension 3:  0.88  → might relate to "power"
...
Dimension 300: 0.23 → some other abstract feature

Note: Dimensions are NOT explicitly labeled—the network learns
      what each dimension represents during training.
```

### 2. Semantic Similarity

Words with similar meanings have vectors that are close together, measured by **cosine similarity**:

```text
Cosine Similarity Formula:

              A · B           Σ(aᵢ × bᵢ)
cos(θ) = ─────────── = ─────────────────────
          ||A|| ||B||   √Σ(aᵢ²) × √Σ(bᵢ²)

Range: -1 to 1 (higher = more similar)

Example Similarities:
┌─────────────────────────────────────┐
│ Word Pair          │ Similarity    │
├─────────────────────────────────────┤
│ king, queen        │ 0.85          │
│ cat, dog           │ 0.76          │
│ happy, joyful      │ 0.89          │
│ king, banana       │ 0.12          │
│ running, walking   │ 0.72          │
└─────────────────────────────────────┘
```

### 3. Vector Arithmetic (Compositionality)

One of the most fascinating properties: semantic relationships are encoded as vector directions!

```text
The Famous Analogy:

    King - Man + Woman ≈ Queen

Visualized:
                    Queen
                     ↑
                     │ Woman
                     │
    King ←───────────┼──────────→ ?
         Man         │
                     │
                     ↓

The vector from "Man" to "King" represents "royalty + male"
Subtracting "Man" removes the male component
Adding "Woman" adds the female component
Result: "royalty + female" = Queen
```

### More Vector Arithmetic Examples

```text
Geographic Relationships:
    Russia - Moscow + Delhi ≈ India
    (Country - Capital + Capital = Country)

    Japan - Tokyo + Paris ≈ France

Verb Tenses:
    walking - walk + swim ≈ swimming
    ran - run + fly ≈ flew

Comparatives:
    bigger - big + small ≈ smaller

Plurals:
    cats - cat + dog ≈ dogs
```

---

## Static Embedding Techniques

### Word2Vec (Google, 2013)

Word2Vec introduced two revolutionary training architectures:

#### Skip-gram: Predict Context from Word

```text
Skip-gram Architecture:

Given center word → Predict surrounding words

Sentence: "The quick brown fox jumps"
Window size: 2

Training pairs (center → context):
  quick → The
  quick → brown
  brown → quick
  brown → fox
  fox → brown
  fox → jumps

Network:
         Center Word
             ↓
    ┌─────────────────┐
    │   Input Layer   │  (one-hot)
    │    (V dims)     │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Hidden Layer   │  (embedding)
    │    (N dims)     │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Output Layer   │  (softmax)
    │    (V dims)     │
    └─────────────────┘
             ↓
      Context Word Probabilities
```

#### CBOW: Predict Word from Context

```text
CBOW (Continuous Bag of Words):

Given surrounding words → Predict center word

Sentence: "The quick brown fox jumps"
Window size: 2

Training example:
  Context: [The, brown] → Center: quick
  Context: [quick, fox] → Center: brown

Network:
    Context Word 1    Context Word 2
          ↓                ↓
    ┌──────────┐     ┌──────────┐
    │  Embed   │     │  Embed   │
    └────┬─────┘     └────┬─────┘
         └────────┬───────┘
                  ↓
            ┌──────────┐
            │ Average  │
            └────┬─────┘
                 ↓
           ┌──────────┐
           │  Output  │
           └────┬─────┘
                ↓
           Center Word Probabilities
```

### GloVe (Stanford, 2014)

**Global Vectors for Word Representation** combines the benefits of matrix factorization and local context methods.

```text
GloVe Key Insight:

Word relationships can be captured through co-occurrence ratios.

Example: Analyzing words related to "ice" vs "steam"

             Co-occurrence with:
             ice    steam
solid       high    low      → ratio >> 1
gas         low     high     → ratio << 1
water       high    high     → ratio ≈ 1
fashion     low     low      → ratio ≈ 1

GloVe learns embeddings such that:
  wᵢᵀwⱼ + bᵢ + bⱼ = log(Xᵢⱼ)

Where Xᵢⱼ = co-occurrence count of word i and j
```

### Comparison: Word2Vec vs GloVe

| Aspect          | Word2Vec                    | GloVe                              |
| :-------------- | :-------------------------- | :--------------------------------- |
| **Approach**    | Predictive (neural network) | Count-based (matrix factorization) |
| **Context**     | Local (sliding window)      | Global (entire corpus statistics)  |
| **Training**    | Stochastic (online)         | Batch (full matrix)                |
| **Speed**       | Faster for small data       | Faster for large data              |
| **Performance** | Similar                     | Similar                            |

---

## The Limitation: Context Blindness

Static embeddings have a critical flaw: **polysemy** (words with multiple meanings).

```text
The Problem:

In static embeddings, "bank" has ONE vector, but:

Sentence 1: "I deposited money at the bank"
Sentence 2: "We had a picnic on the river bank"
Sentence 3: "Don't bank on it happening"

   bank (financial) ≠ bank (river) ≠ bank (rely on)
        ↓                 ↓               ↓
   Same vector!       Same vector!    Same vector!

The embedding averages all meanings, accurately representing NONE.
```

### Other Polysemy Examples

```text
"Apple":
  - "I ate an apple" (fruit)
  - "I bought an Apple" (company)

"Play":
  - "Let's play a game" (activity)
  - "I watched a play" (theater)
  - "Press play to start" (button)

"Spring":
  - "Spring is my favorite season" (time)
  - "The spring in the mattress broke" (coil)
  - "Water from the spring was cold" (water source)
```

---

## From Static to Contextual Embeddings

The limitations of static embeddings led to the development of **contextual embeddings** in Transformers:

```text
Static Embeddings (Word2Vec, GloVe):
┌─────────────────────────────────────┐
│ "bank" → [0.2, -0.3, 0.5, ...]     │
│                                     │
│ Same vector regardless of context   │
└─────────────────────────────────────┘

Contextual Embeddings (BERT, GPT):
┌─────────────────────────────────────────────────────┐
│ "I visited the bank to deposit money"               │
│ "bank" → [0.8, 0.2, -0.1, ...]  (financial)        │
│                                                     │
│ "We walked along the river bank"                    │
│ "bank" → [-0.2, 0.7, 0.3, ...]  (geographical)     │
│                                                     │
│ Different vectors based on context!                 │
└─────────────────────────────────────────────────────┘
```

### How Transformers Create Contextual Embeddings

```text
Processing: "The bank was near the river"

Step 1: Start with static token embeddings
        bank → [initial embedding]

Step 2: Self-attention reads the full context
        bank ← attends to → [The, was, near, the, river]
                                              ↑
                                         HIGH attention
                                         (context clue!)

Step 3: Update embedding based on attention
        bank → [river-influenced embedding]

Result: "bank" representation is now river-specific!
```

---

## Embedding Dimensions in Practice

### Common Configurations

| Model               | Embedding Dim     | Vocabulary Size | Total Parameters       |
| :------------------ | :---------------- | :-------------- | :--------------------- |
| Word2Vec (original) | 300               | ~3M words       | 900M                   |
| GloVe (6B)          | 50, 100, 200, 300 | 400K            | 20M-120M               |
| BERT-Base           | 768               | 30,522          | 23M (embeddings only)  |
| BERT-Large          | 1024              | 30,522          | 31M (embeddings only)  |
| GPT-2               | 768               | 50,257          | 38M (embeddings only)  |
| GPT-3               | 12,288            | 50,257          | 617M (embeddings only) |

### Choosing Embedding Dimensions

```text
Guidelines:

Small (50-100 dims):
  ✓ Fast training and inference
  ✓ Good for small vocabularies
  ✓ Simple tasks

Medium (200-300 dims):
  ✓ Good balance of speed and quality
  ✓ Standard for Word2Vec/GloVe
  ✓ Most NLP tasks

Large (512-1024 dims):
  ✓ Better semantic capture
  ✓ Modern Transformers (BERT, GPT)
  ✓ Complex tasks

Very Large (2048+ dims):
  ✓ State-of-the-art LLMs
  ✓ Maximum expressiveness
  ✗ Expensive to compute/store
```

---

## Using Pre-trained Embeddings in PyTorch

### Loading GloVe Embeddings

```python
import torch
import torch.nn as nn

def load_glove_embeddings(glove_path, word_to_idx, embedding_dim=300):
    """Load GloVe embeddings for your vocabulary."""

    # Initialize random embeddings
    vocab_size = len(word_to_idx)
    embeddings = torch.randn(vocab_size, embedding_dim)

    # Load GloVe vectors
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_idx:
                idx = word_to_idx[word]
                vector = torch.tensor([float(v) for v in values[1:]])
                embeddings[idx] = vector

    return embeddings

# Create embedding layer with pre-trained weights
pretrained_embeddings = load_glove_embeddings(
    'glove.6B.300d.txt',
    word_to_idx,
    embedding_dim=300
)

embedding_layer = nn.Embedding.from_pretrained(
    pretrained_embeddings,
    freeze=False  # Set True to keep embeddings fixed
)
```

### Simple Embedding Example

```python
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 10000
embedding_dim = 128
embedding = nn.Embedding(vocab_size, embedding_dim)

# Convert word indices to embeddings
word_indices = torch.tensor([42, 256, 1024])  # 3 word indices
word_vectors = embedding(word_indices)

print(f"Input shape: {word_indices.shape}")      # (3,)
print(f"Output shape: {word_vectors.shape}")     # (3, 128)

# Each word now has a 128-dimensional representation!
```

---

## Visualizing Embeddings with t-SNE

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(words, embeddings, word_to_idx):
    """Visualize word embeddings in 2D using t-SNE."""

    # Get embeddings for selected words
    indices = [word_to_idx[w] for w in words]
    vectors = embeddings[indices].detach().numpy()

    # Reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.title("Word Embeddings Visualization")
    plt.show()

# Example usage
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'dog', 'cat', 'puppy', 'kitten', 'car', 'truck', 'bus']
visualize_embeddings(words, embeddings, word_to_idx)
```

**Expected Result:**

```text
t-SNE Plot:

                    royalty cluster
                    ┌─────────────┐
         queen •    │             │    • princess
              king • │             │ • prince
                    │    man •    │
                    │  woman •    │
                    └─────────────┘

    dog •  • cat                         • car
                                    • truck
  puppy •    • kitten                    • bus
         ↑                               ↑
    animal cluster              vehicle cluster
```

---

## Summary

| Concept                  | Description                                                  |
| :----------------------- | :----------------------------------------------------------- |
| **Word Embeddings**      | Dense vector representations that capture semantic meaning   |
| **One-Hot Encoding**     | Sparse, high-dimensional, no semantic information            |
| **Dimensionality**       | Typically 100-1024 dimensions per word                       |
| **Semantic Similarity**  | Similar words have similar vectors (cosine similarity)       |
| **Vector Arithmetic**    | Relationships encoded as vector operations                   |
| **Word2Vec**             | Neural predictive model (Skip-gram, CBOW)                    |
| **GloVe**                | Count-based global co-occurrence statistics                  |
| **Static vs Contextual** | Static = one vector per word; Contextual = context-dependent |

---

## What's Next?

Now that you understand how words are represented numerically, let's explore the architecture that processes these embeddings:

➡️ **Next:** [Architecture Overview: Encoder and Decoder](02-Architecture-Overview.md)

---

_Word embeddings were a breakthrough that enabled modern NLP. Understanding how meaning is encoded in vectors is essential for grasping how Transformers and LLMs work at a fundamental level._
