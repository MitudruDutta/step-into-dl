# 🧠 What is a Neuron?

A neuron is the basic building block of neural networks in deep learning. It is a mathematical unit that mimics the biological neuron in the human brain, receiving signals, processing them, and transmitting outputs to other neurons.

---

## The Computational Process

Each neuron follows a specific workflow to process information:

1. **Input Reception**: The neuron receives input values from the data source or neurons in a previous layer. These inputs can be raw features (like pixel values) or processed signals from earlier computations.

2. **Weighting & Summation**: The neuron applies a **weight** to each input and sums them together, adding a **bias** term. This weighted sum determines how much influence each input has on the final output.
   - Formula: `z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b`

3. **Activation**: The weighted sum passes through an **activation function** that introduces non-linearity, allowing the network to learn complex patterns beyond simple linear relationships.

4. **Transmission**: The resulting activated value is passed to neurons in the next layer, continuing the flow of information through the network.

> **Mathematical Insight**: Logistic regression is essentially the simplest form of a neural network, consisting of only a single neuron with a sigmoid activation function.

---

## Biological Inspiration

The artificial neuron draws inspiration from biological neurons:

| Biological Component | Artificial Equivalent |
|---------------------|----------------------|
| **Dendrites** | Input connections (weights) |
| **Cell Body** | Summation and processing |
| **Axon** | Output transmission |
| **Synapses** | Connections to other neurons (network edges) |

---

## How Information Flows

```
Inputs (x₁, x₂, ..., xₙ)
         ↓
    [Weights (w₁, w₂, ..., wₙ)]
         ↓
    Weighted Sum: z = Σ(wᵢ × xᵢ) + b
         ↓
    Activation Function: a = f(z)
         ↓
    Output → Next Layer
```

---

## Key Concepts

### Weights
- Learnable parameters that determine the importance of each input
- Adjusted during training via backpropagation
- Larger weights = stronger influence on the output

### Bias
- An additional learnable parameter added to the weighted sum
- Allows the neuron to shift its activation threshold
- Helps the model fit data that doesn't pass through the origin

### Activation
- Transforms the linear weighted sum into a non-linear output
- Without activation, stacking layers would be equivalent to a single linear transformation
- Different activation functions serve different purposes (covered in detail in the activation functions guide)

---

*Understanding the neuron is fundamental to grasping how neural networks learn. Each neuron is simple, but millions working together can solve incredibly complex problems.*
