# 📚 Training Fundamentals

Understanding how neural networks learn is crucial for building effective models. This guide covers the core concepts of training.

---

## Loss Functions

Loss functions measure how far off a model's predictions are from the actual values. The goal of training is to minimize this loss.

### Regression Losses

| Loss | Formula | When to Use |
|------|---------|-------------|
| **MSE** | mean((y - ŷ)²) | General regression; penalizes large errors |
| **MAE** | mean(\|y - ŷ\|) | When outliers should have less influence |
| **Huber** | Combination of MSE and MAE | Robust to outliers while still penalizing large errors |

### Classification Losses

| Loss | When to Use |
|------|-------------|
| **Binary Cross-Entropy** | Two classes (spam vs. not spam) |
| **Categorical Cross-Entropy** | Multiple classes, one-hot encoded labels |
| **Sparse Categorical Cross-Entropy** | Multiple classes, integer labels |

### How Loss Guides Learning

```
High Loss → Model predictions are far from targets
           → Large gradients
           → Big weight updates

Low Loss → Model predictions are close to targets
          → Small gradients
          → Small weight updates (fine-tuning)
```

---

## Backpropagation

The algorithm that makes learning possible. It calculates how much each weight contributed to the error.

### The Four Steps

1. **Forward Pass**
   - Input data flows through the network
   - Each layer transforms the data
   - Final layer produces a prediction

2. **Calculate Loss**
   - Compare prediction to actual target
   - Compute a single number representing error

3. **Backward Pass**
   - Compute gradients using the chain rule
   - Calculate how much each weight contributed to the loss
   - Gradients flow backward through the network

4. **Update Weights**
   - Adjust weights in the direction that reduces loss
   - Use an optimizer to determine update magnitude

### The Chain Rule

Backpropagation relies on the chain rule from calculus:

```
If y = f(g(x)), then dy/dx = dy/dg × dg/dx
```

This allows gradients to flow through multiple layers, each layer passing its gradient to the previous one.

---

## Optimizers

Optimizers determine how weights are updated during training. They use gradients to decide the direction and magnitude of updates.

### Common Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| **SGD** | Stochastic Gradient Descent; simple but effective | General use, large datasets |
| **SGD + Momentum** | Adds velocity to updates; helps escape local minima | When SGD is too slow |
| **Adam** | Adaptive learning rates; combines momentum and RMSprop | Most deep learning tasks (default choice) |
| **AdamW** | Adam with proper weight decay | When using regularization |
| **RMSprop** | Adapts learning rate based on recent gradients | RNNs and non-stationary problems |

### How They Differ

**SGD**: Simple update in gradient direction
```
weight = weight - learning_rate × gradient
```

**Momentum**: Accumulates velocity from past gradients
```
velocity = momentum × velocity - learning_rate × gradient
weight = weight + velocity
```

**Adam**: Adapts learning rate per parameter
```
Maintains running averages of gradients and squared gradients
Adjusts learning rate based on these statistics
```

### Choosing an Optimizer

- **Start with Adam** — works well in most cases
- **Try SGD + Momentum** — can generalize better with proper tuning
- **Use AdamW** — when applying weight decay regularization

---

## Hyperparameters

Key settings you'll tune during experimentation. These are not learned by the model—you set them before training.

### Learning Rate

How big of a step to take when updating weights.

| Value | Effect |
|-------|--------|
| **Too High** | Overshoots minimum, loss oscillates or diverges |
| **Too Low** | Very slow convergence, may get stuck |
| **Just Right** | Steady decrease in loss, good convergence |

**Typical Values**: 0.001 to 0.0001 for Adam, 0.01 to 0.1 for SGD

**Learning Rate Schedules**:
- **Step decay**: Reduce LR by factor every N epochs
- **Cosine annealing**: Smoothly decrease LR following cosine curve
- **Warmup**: Start low, increase, then decrease

### Batch Size

Number of samples processed before updating weights.

| Size | Pros | Cons |
|------|------|------|
| **Small (16-32)** | Better generalization, less memory | Noisy gradients, slower |
| **Large (128-512)** | Stable gradients, faster | May generalize worse, more memory |

**Common Values**: 32, 64, 128, 256

### Epochs

Number of complete passes through the training dataset.

- **Too Few**: Underfitting, model hasn't learned enough
- **Too Many**: Overfitting, model memorizes training data
- **Use Early Stopping**: Monitor validation loss, stop when it stops improving

### Dropout Rate

Percentage of neurons randomly "turned off" during training.

- **Purpose**: Prevents overfitting by forcing redundancy
- **Typical Values**: 0.2 to 0.5
- **Where to Apply**: Between fully connected layers
- **During Inference**: Dropout is disabled

---

## The Training Loop

A typical training loop in PyTorch:

```
For each epoch:
    For each batch in training data:
        1. Zero the gradients
        2. Forward pass: predictions = model(inputs)
        3. Calculate loss: loss = criterion(predictions, targets)
        4. Backward pass: loss.backward()
        5. Update weights: optimizer.step()
    
    Evaluate on validation set
    Check for early stopping
    Save checkpoint if best so far
```

---

## Key Takeaways

1. **Loss functions** measure prediction error and guide learning
2. **Backpropagation** calculates gradients using the chain rule
3. **Optimizers** use gradients to update weights (start with Adam)
4. **Hyperparameters** require experimentation to tune
5. **Monitor both training and validation loss** to detect overfitting

---

*Mastering these fundamentals is essential before building complex models. Understanding why training works helps you debug when it doesn't.*
