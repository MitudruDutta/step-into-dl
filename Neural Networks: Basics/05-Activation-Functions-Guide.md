# 📊 Comprehensive Guide to Activation Functions

Selecting the correct activation function is vital for network performance and avoiding training issues like vanishing or exploding gradients.

---

## Quick Reference Table

| Function | Output Range | Primary Use Case | Key Features |
|----------|--------------|------------------|--------------|
| **Sigmoid** | 0 to 1 | Binary classification output | Maps to probability; vanishing gradients |
| **Softmax** | 0 to 1 | Multi-class classification output | Probability distribution summing to 1 |
| **Tanh** | -1 to 1 | Hidden layers | Zero-centered; stronger gradients |
| **ReLU** | 0 to ∞ | Default for hidden layers | Fast; avoids vanishing gradients |
| **Leaky ReLU** | -∞ to ∞ | Hidden layers | Solves dying ReLU problem |

---

## Detailed Breakdown

### Sigmoid (Logistic)

**Formula**: σ(x) = 1 / (1 + e⁻ˣ)

**Output Range**: (0, 1)

**Characteristics**:
- Smooth, S-shaped curve
- Output interpretable as probability
- Derivative: σ(x) × (1 - σ(x))

**Pros**:
- Smooth gradient everywhere
- Output bounded and interpretable
- Good for binary classification output

**Cons**:
- Vanishing gradient for large/small inputs (|x| > 4)
- Not zero-centered (outputs always positive)
- Computationally more expensive than ReLU

**Best For**: Binary classification output layers

**Numerical Stability Note**: For extreme inputs, production code often uses numerically-stable variants (conditional formulations or clipping) to avoid overflow.

---

### Softmax

**Formula**: softmax(xᵢ) = eˣⁱ / Σeˣʲ

**Output Range**: (0, 1) for each element, sum = 1

**Characteristics**:
- Generalizes sigmoid to multiple classes
- Outputs form a probability distribution
- Emphasizes the largest values

**Pros**:
- Outputs sum to 1 (valid probability distribution)
- Interpretable as class probabilities
- Differentiable

**Cons**:
- O(n) complexity, can suffer from numerical stability issues (overflow/underflow) with large logits or high class counts
- Use numerically-stable implementations (subtract max logit or log-sum-exp trick)

**Best For**: Multi-class classification output layers (digit recognition, image classification)

**Implementation Tip**: Always subtract the maximum value before exponentiating to prevent overflow:
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

---

### Tanh (Hyperbolic Tangent)

**Formula**: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

**Output Range**: (-1, 1)

**Characteristics**:
- Zero-centered output
- Steeper gradient than sigmoid
- Derivative: 1 - tanh²(x)

**Pros**:
- Zero-centered (helps with gradient updates)
- Stronger gradients than sigmoid
- Bounded output

**Cons**:
- Still suffers from vanishing gradients at extremes
- More computationally expensive than ReLU

**Best For**: 
- Hidden layers when zero-centered output is beneficial
- RNNs and LSTMs (historically)
- When output needs to be bounded

---

### ReLU (Rectified Linear Unit)

**Formula**: ReLU(x) = max(0, x)

**Output Range**: [0, ∞)

**Characteristics**:
- Simple threshold at zero
- Linear for positive values
- Introduces sparsity

**Pros**:
- Computationally efficient (just a comparison)
- Reduces vanishing gradient problem
- Promotes sparse activations
- Converges faster in practice

**Cons**:
- "Dying ReLU" — neurons can become permanently inactive
- Not zero-centered
- Unbounded (can cause exploding activations)

**Best For**: Default choice for hidden layers in most architectures

**The Dying ReLU Problem**: If a neuron's weights update such that it always receives negative input, it will always output 0 and never recover (gradient is 0 for negative inputs).

---

### Leaky ReLU

**Formula**: LeakyReLU(x) = max(αx, x) where α is small (e.g., 0.01)

**Output Range**: (-∞, ∞)

**Characteristics**:
- Small slope for negative values
- Prevents complete neuron death
- α is typically 0.01 or learned

**Pros**:
- Prevents dying neurons
- Maintains gradient flow for negative inputs
- All benefits of ReLU

**Cons**:
- Adds hyperparameter α to tune
- Results can be inconsistent across problems

**Best For**: When experiencing dying ReLU problems

**Variants**:
- **Parametric ReLU (PReLU)**: α is learned during training
- **Exponential Linear Unit (ELU)**: Smooth curve for negative values
- **SELU**: Self-normalizing, maintains mean and variance

---

## Selection Guidelines

### Output Layer

| Task | Activation | Reason |
|------|------------|--------|
| Binary classification | Sigmoid | Output is probability [0, 1] |
| Multi-class classification | Softmax | Outputs are class probabilities |
| Regression | Linear (none) | Unbounded continuous output |
| Regression (positive only) | ReLU | Ensures non-negative output |

### Hidden Layers

| Situation | Recommendation |
|-----------|----------------|
| Default choice | ReLU |
| Dying neurons observed | Leaky ReLU or ELU |
| Need bounded output | Tanh |
| RNNs/LSTMs | Tanh (traditional) or ReLU |
| Very deep networks | ReLU with careful initialization |

### What to Avoid

- Sigmoid/Tanh in deep hidden layers (vanishing gradients)
- ReLU in output layers for classification
- Step functions (not differentiable)

---

## Comparison Visualization

```
Input:  -3   -2   -1    0    1    2    3

Sigmoid: 0.05 0.12 0.27 0.50 0.73 0.88 0.95
Tanh:   -0.99-0.96-0.76 0.00 0.76 0.96 0.99
ReLU:    0    0    0    0    1    2    3
Leaky:  -0.03-0.02-0.01 0    1    2    3
```

---

## Practical Tips

1. **Start with ReLU** for hidden layers — it works well in most cases
2. **Monitor for dead neurons** — if many neurons output 0, try Leaky ReLU
3. **Use appropriate output activation** — match it to your loss function
4. **Consider batch normalization** — it can reduce sensitivity to activation choice
5. **Experiment** — the best activation can be problem-dependent

---

*Choosing the right activation function is part science, part art. Start with the defaults, monitor your training, and adjust based on what you observe.*
