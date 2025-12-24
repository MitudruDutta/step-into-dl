# 🎯 What is Hyperparameter Tuning?

Hyperparameter tuning is the practice of selecting the best values for parameters that govern the training process. It's one of the most important steps in building high-performing machine learning models.

---

## Hyperparameters vs. Model Parameters

Understanding the difference between hyperparameters and model parameters is fundamental.

### Model Parameters

Model parameters are **learned from data** during training:
- Weights in neural network layers
- Biases in neurons
- Learned embeddings

```python
# These are MODEL PARAMETERS - learned during training
model = nn.Linear(10, 5)
print(model.weight)  # Learned from data
print(model.bias)    # Learned from data
```

### Hyperparameters

Hyperparameters are **set before training** and control how learning happens:
- Learning rate
- Batch size
- Number of layers
- Dropout rate

```python
# These are HYPERPARAMETERS - set by you
learning_rate = 0.001      # How fast to learn
batch_size = 32            # Samples per update
num_layers = 3             # Architecture choice
dropout_rate = 0.5         # Regularization strength
```

### Comparison Table

| Aspect | Hyperparameters | Model Parameters |
|--------|-----------------|------------------|
| **When set** | Before training begins | During training |
| **How determined** | Manual selection, search algorithms | Gradient descent |
| **Examples** | Learning rate, batch size, layers | Weights, biases |
| **Number** | Typically 5-20 | Thousands to billions |
| **Optimization** | Tuning algorithms | Backpropagation |
| **Affects** | How the model learns | What the model learns |

---

## Categories of Hyperparameters

### 1. Training Hyperparameters

Control the optimization process:

| Hyperparameter | Description | Typical Range |
|----------------|-------------|---------------|
| **Learning rate** | Step size for weight updates | 1e-5 to 0.1 |
| **Batch size** | Samples per gradient update | 8 to 512 |
| **Epochs** | Passes through training data | 10 to 1000 |
| **Optimizer** | Algorithm for updates | Adam, SGD, etc. |
| **LR schedule** | How LR changes over time | Step, cosine, etc. |

### 2. Architecture Hyperparameters

Define the model structure:

| Hyperparameter | Description | Typical Range |
|----------------|-------------|---------------|
| **Number of layers** | Depth of network | 1 to 100+ |
| **Neurons per layer** | Width of layers | 32 to 4096 |
| **Activation function** | Non-linearity type | ReLU, GELU, etc. |
| **Kernel size** | CNN filter dimensions | 3, 5, 7 |
| **Attention heads** | Transformer parallelism | 4 to 16 |

### 3. Regularization Hyperparameters

Prevent overfitting:

| Hyperparameter | Description | Typical Range |
|----------------|-------------|---------------|
| **Dropout rate** | Neuron drop probability | 0.0 to 0.5 |
| **Weight decay** | L2 penalty strength | 1e-6 to 1e-2 |
| **Early stopping patience** | Epochs to wait | 3 to 20 |
| **Data augmentation** | Transform intensity | Task-specific |

### 4. Data Hyperparameters

Affect data processing:

| Hyperparameter | Description | Typical Range |
|----------------|-------------|---------------|
| **Train/val split** | Data division ratio | 0.8/0.2, 0.9/0.1 |
| **Sequence length** | Input size for sequences | 128 to 2048 |
| **Image size** | Input dimensions | 224, 384, 512 |
| **Normalization** | Scaling method | BatchNorm, LayerNorm |

---

## Why Hyperparameter Tuning Matters

### The Impact of Hyperparameters

```
Same model, same data, different hyperparameters:

Configuration A (poor):
  Learning rate: 0.1 (too high)
  → Training diverges, accuracy: 10%

Configuration B (poor):
  Learning rate: 0.00001 (too low)
  → Training too slow, accuracy: 65% after 100 epochs

Configuration C (good):
  Learning rate: 0.001 (just right)
  → Fast convergence, accuracy: 95%
```

### Real-World Example

```python
# Poor hyperparameters
model_poor = train(
    lr=0.1,           # Too high - training unstable
    batch_size=8,     # Too small - very slow
    dropout=0.8,      # Too high - underfitting
    layers=1          # Too few - can't learn complex patterns
)
# Result: 72% accuracy

# Good hyperparameters
model_good = train(
    lr=0.001,         # Stable learning
    batch_size=64,    # Good balance
    dropout=0.3,      # Appropriate regularization
    layers=3          # Sufficient capacity
)
# Result: 94% accuracy
```

### The Difference Good Tuning Makes

| Scenario | Accuracy | Training Time |
|----------|----------|---------------|
| Default hyperparameters | 85% | 2 hours |
| Random guessing | 78% | 3 hours |
| Systematic tuning | 94% | 1.5 hours |
| Expert tuning | 96% | 1 hour |

---

## The Hyperparameter Search Problem

### Why It's Challenging

1. **Large Search Space**
   ```
   5 hyperparameters × 10 values each = 100,000 combinations
   Each evaluation takes 1 hour = 11+ years to try all!
   ```

2. **Expensive Evaluation**
   - Each configuration requires full training
   - Deep learning training can take hours/days
   - GPU resources are costly

3. **Complex Interactions**
   ```
   Learning rate and batch size interact:
   - Large batch + low LR → slow training
   - Large batch + high LR → may work well
   - Small batch + high LR → unstable
   ```

4. **Non-Convex Landscape**
   - Multiple local optima
   - No gradient to follow
   - Global optimum not guaranteed

### Search Space Explosion

```python
# Example search space
search_space = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],  # 4 options
    'batch_size': [16, 32, 64, 128],               # 4 options
    'hidden_layers': [1, 2, 3, 4],                 # 4 options
    'neurons': [64, 128, 256, 512],                # 4 options
    'dropout': [0.0, 0.2, 0.3, 0.5],               # 4 options
    'optimizer': ['adam', 'sgd', 'rmsprop'],       # 3 options
    'activation': ['relu', 'gelu', 'tanh']         # 3 options
}

# Total combinations
total = 4 * 4 * 4 * 4 * 4 * 3 * 3
print(f"Total combinations: {total:,}")  # 9,216 combinations!
```

---

## Hyperparameter Sensitivity

Not all hyperparameters are equally important.

### High Sensitivity (Tune First)

```
Learning Rate:
  0.1    → Training diverges
  0.01   → Slow but stable
  0.001  → Often optimal
  0.0001 → Very slow

Small changes have big impact!
```

### Medium Sensitivity

```
Batch Size:
  16  → More noise, slower
  32  → Good balance
  64  → Faster, less noise
  128 → May need LR adjustment

Moderate impact on results.
```

### Low Sensitivity

```
Activation Function (for most tasks):
  ReLU  → Works well
  GELU  → Works well
  ELU   → Works well

Often doesn't matter much.
```

### Sensitivity Ranking

| Rank | Hyperparameter | Sensitivity |
|------|----------------|-------------|
| 1 | Learning rate | Very High |
| 2 | Architecture (layers/neurons) | High |
| 3 | Batch size | Medium-High |
| 4 | Regularization (dropout, L2) | Medium |
| 5 | Optimizer choice | Medium |
| 6 | Activation function | Low |
| 7 | Weight initialization | Low |

---

## When to Tune Hyperparameters

### Always Tune

- Learning rate (most important)
- Model architecture for your specific task
- Regularization if overfitting

### Sometimes Tune

- Batch size (if training is slow or unstable)
- Optimizer (Adam usually works, but SGD can be better)
- Learning rate schedule

### Rarely Tune

- Activation functions (ReLU/GELU usually fine)
- Weight initialization (defaults are good)
- Optimizer betas (defaults are well-tuned)

---

## Key Takeaways

1. **Hyperparameters control training**, model parameters are learned
2. **Learning rate is most important** — always tune it
3. **Search space grows exponentially** with number of hyperparameters
4. **Not all hyperparameters matter equally** — focus on sensitive ones
5. **Good defaults exist** — start there and tune incrementally
6. **Interactions matter** — some hyperparameters work together

---

*Understanding what hyperparameters are and why they matter is the first step to building better models. The next step is learning how to find good values efficiently.*
