# 🚧 Common Challenges & Solutions

Every deep learning practitioner encounters these challenges. Understanding them helps you diagnose and fix training issues quickly.

---

## Overfitting

When a model memorizes training data but fails on new data. The model learns noise and specific patterns that don't generalize.

### Symptoms
- High training accuracy, low validation accuracy
- Training loss continues decreasing while validation loss increases
- Model performs well on seen data, poorly on unseen data

### Visual Indicator
```
Loss
  ↑
  │    Training ────────────────
  │                    ╱
  │    Validation ────╱  (diverging = overfitting)
  │                  ╱
  └──────────────────────────→ Epochs
```

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **More training data** | More examples = harder to memorize |
| **Dropout layers** | Forces redundancy, prevents co-adaptation |
| **Data augmentation** | Artificially increases dataset diversity |
| **Early stopping** | Stop training when validation loss stops improving |
| **L1/L2 regularization** | Penalizes large weights, encourages simplicity |
| **Reduce model complexity** | Fewer parameters = less capacity to memorize |

### When to Suspect Overfitting
- Small dataset relative to model size
- Training for too many epochs
- No regularization techniques applied
- Model is very deep/wide

---

## Underfitting

When a model is too simple to capture patterns in the data. The model fails to learn the underlying structure.

### Symptoms
- Low accuracy on both training and validation sets
- Training loss plateaus at a high value
- Model predictions are essentially random or constant

### Visual Indicator
```
Loss
  ↑
  │────────────────────────  (both high = underfitting)
  │    Training ≈ Validation
  │
  │
  └──────────────────────────→ Epochs
```

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **Increase model complexity** | More layers/neurons = more capacity |
| **Train for more epochs** | Give the model more time to learn |
| **Reduce regularization** | Too much regularization constrains learning |
| **Better features** | Engineer more informative input features |
| **Tune learning rate** | Try higher LR or schedules to escape plateaus faster |
| **Check data quality** | Ensure labels are correct and data is clean |

### When to Suspect Underfitting
- Model is very simple (few layers/neurons)
- Training stopped too early
- Heavy regularization applied
- Features don't contain enough information

---

## Vanishing Gradients

When gradients become extremely small during backpropagation, causing early layers to learn very slowly or not at all.

### Symptoms
- Early layers have near-zero gradients
- Training is extremely slow
- Deep networks fail to learn
- Loss decreases very slowly or not at all

### Causes
- **Sigmoid/Tanh activations**: Squash gradients in saturated regions
- **Deep networks**: Gradients multiply through many layers
- **Poor initialization**: Weights start in saturated regions

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **ReLU activation** | Gradient is 1 for positive values (no squashing) |
| **Batch normalization** | Normalizes activations, keeps gradients healthy |
| **Proper initialization** | Xavier for Sigmoid/Tanh, He for ReLU |
| **Residual connections** | Skip connections allow gradients to flow directly |
| **Gradient clipping** | Prevents gradients from becoming too small/large |

---

## Exploding Gradients

When gradients become extremely large, causing unstable training and NaN values.

### Symptoms
- Loss suddenly becomes NaN or infinity
- Weights grow to very large values
- Training is unstable, loss oscillates wildly

### Causes
- **Learning rate too high**: Updates are too large
- **Poor initialization**: Weights start too large
- **Deep networks without normalization**: Gradients compound

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **Gradient clipping** | Caps gradient magnitude |
| **Lower learning rate** | Smaller updates = more stability |
| **Batch normalization** | Keeps activations in reasonable range |
| **Proper initialization** | Prevents large initial gradients |
| **Weight decay** | Penalizes large weights |

### Gradient Clipping Example
```python
# Clip gradients to maximum norm of 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Dead Neurons (ReLU-specific)

When neurons always output zero and stop learning entirely.

### Symptoms
- Many neurons output exactly 0 for all inputs
- Model capacity is effectively reduced
- Training plateaus despite having many parameters

### Causes
- **Large negative bias**: Neuron never activates
- **Large learning rate**: Weights update to always-negative region
- **Poor initialization**: Neurons start in dead region

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **Leaky ReLU** | Small slope for negative values keeps gradients flowing |
| **PReLU/ELU** | Learnable or smooth negative regions |
| **Lower learning rate** | Prevents drastic weight updates |
| **Careful initialization** | He initialization for ReLU |

---

## Debugging Checklist

When training isn't working, check these in order:

### 1. Data Issues
- [ ] Data loaded correctly? (visualize samples)
- [ ] Labels correct? (spot check)
- [ ] Data normalized/standardized?
- [ ] No data leakage between train/val/test?

### 2. Model Issues
- [ ] Architecture appropriate for task?
- [ ] Activation functions present between layers?
- [ ] Output layer matches task (sigmoid for binary, softmax for multi-class)?

### 3. Training Issues
- [ ] Loss function appropriate?
- [ ] Learning rate reasonable? (try 0.001)
- [ ] Gradients flowing? (check for NaN)
- [ ] Batch size appropriate?

### 4. Overfitting/Underfitting
- [ ] Compare training vs. validation loss
- [ ] Add/remove regularization as needed
- [ ] Adjust model complexity

---

## Quick Diagnosis Table

| Symptom | Likely Problem | First Thing to Try |
|---------|---------------|-------------------|
| Loss is NaN | Exploding gradients | Lower learning rate, gradient clipping |
| Loss doesn't decrease | Underfitting or bug | Check data, simplify model, verify gradients |
| Val loss increases while train decreases | Overfitting | Add dropout, early stopping |
| Training very slow | Vanishing gradients | Use ReLU, batch norm |
| Many zero activations | Dead neurons | Use Leaky ReLU |

---

*These challenges are normal—every practitioner faces them. The key is recognizing symptoms quickly and knowing which solutions to try.*
