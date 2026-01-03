# 📉 Loss Functions: Binary Cross Entropy (BCE)

While Mean Squared Error (MSE) is the "go-to" for regression, it is often unsuitable for **Binary Classification** (Yes/No problems). This guide explains why BCE is the correct choice.

---

## The Binary Classification Problem

Binary classification predicts one of two outcomes:
- Spam / Not Spam
- Fraud / Legitimate
- Disease / Healthy
- Click / No Click

The model outputs a probability between 0 and 1, representing confidence in the positive class.

---

## Why MSE Fails in Classification

### Problem 1: Non-Convex Loss Surface

Using MSE for classification creates a "bumpy" cost surface that is not convex:

```
Loss
  ↑
  │    ╱╲    ╱╲
  │   ╱  ╲  ╱  ╲    ← Multiple valleys (local minima)
  │  ╱    ╲╱    ╲
  │ ╱            ╲
  └──────────────────→ Weights
```

**Why this happens:**
- Sigmoid squashes outputs to [0, 1]
- MSE measures squared difference
- The combination creates multiple local minima

### Problem 2: Local Minima Traps

These "bumps" make it easy for the model to get stuck in a **local minimum** (a false "best" point) instead of finding the true global minimum.

The optimizer thinks it found the best solution, but it's actually stuck in a suboptimal valley.

### Problem 3: Vanishing Gradients

MSE gradients become very small when predictions are confident but wrong:

```
If prediction = 0.99, target = 0:
MSE gradient ∝ (0.99 - 0) × sigmoid_derivative(0.99)
            ∝ 0.99 × (very small number)
            ≈ very small gradient
```

Learning slows down exactly when the model needs to correct its biggest mistakes.

---

## The BCE + Sigmoid Solution

### Convex Loss Surface

Combining **Binary Cross Entropy (BCE)** with a **Sigmoid** activation creates a smooth, convex cost surface:

```
Loss
  ↑
  │╲
  │ ╲
  │  ╲
  │   ╲___________    ← Single global minimum
  └──────────────────→ Weights
```

This guarantees that gradient descent will find the global minimum.

### Strong Gradients for Wrong Predictions

BCE heavily penalizes "high confidence errors":

| Prediction | Target | BCE Loss |
|------------|--------|----------|
| 0.99 | 1 | 0.01 (low — correct) |
| 0.01 | 1 | 4.61 (high — wrong) |
| 0.99 | 0 | 4.61 (high — wrong) |
| 0.01 | 0 | 0.01 (low — correct) |

When the model is very sure but wrong, the loss is enormous. This creates strong gradients that quickly correct mistakes.

### Probabilistic Foundation

BCE is mathematically derived from **maximum likelihood estimation** for Bernoulli distributions. When your output represents a probability, BCE is the theoretically correct loss function.

---

## BCE Formula

```
BCE = -[y × log(p) + (1-y) × log(1-p)]

Where:
- y = actual label (0 or 1)
- p = predicted probability (sigmoid output)
```

### Breaking It Down

**When y = 1 (positive class):**
```
BCE = -log(p)
```
- If p = 0.99 → BCE = 0.01 (low loss, correct)
- If p = 0.01 → BCE = 4.61 (high loss, wrong)

**When y = 0 (negative class):**
```
BCE = -log(1-p)
```
- If p = 0.01 → BCE = 0.01 (low loss, correct)
- If p = 0.99 → BCE = 4.61 (high loss, wrong)

### Visualization

```
BCE Loss vs Predicted Probability (when y=1)

Loss
  ↑
4 │╲
  │ ╲
3 │  ╲
  │   ╲
2 │    ╲
  │     ╲
1 │      ╲___
  │          ╲___
0 └──────────────→ p
  0   0.5    1.0
```

The loss approaches infinity as the prediction approaches the wrong answer.

---

## PyTorch Implementation

### Option 1: BCELoss (with Sigmoid)

```python
import torch
import torch.nn as nn

# Model with sigmoid in forward pass
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(x))

model = Model()
criterion = nn.BCELoss()

# Training
output = model(input_data)  # Already passed through sigmoid
loss = criterion(output, target)
```

### Option 2: BCEWithLogitsLoss (Recommended)

```python
# Model WITHOUT sigmoid in forward pass
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)  # Raw logits

model = Model()
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE

# Training
output = model(input_data)  # Raw logits
loss = criterion(output, target)
```

**Why BCEWithLogitsLoss is better:**
- More numerically stable
- Combines sigmoid and BCE in one operation
- Avoids log(0) issues
- Slightly faster

---

## When to Use BCE

| Scenario | Use BCE? | Notes |
|----------|----------|-------|
| Binary classification (spam/not spam) | ✅ Yes | Standard use case |
| Multi-label classification | ✅ Yes | Apply BCE per label |
| Multi-class classification (one of many) | ❌ No | Use CrossEntropyLoss |
| Regression | ❌ No | Use MSE/MAE |
| Ordinal classification | ⚠️ Maybe | Consider specialized losses |

### Multi-Label vs Multi-Class

**Multi-Label** (use BCE):
- Each sample can have multiple labels
- Example: Image tags (sunny, beach, people)
- Output: Multiple sigmoids, one per label

**Multi-Class** (use CrossEntropy):
- Each sample has exactly one label
- Example: Digit recognition (0-9)
- Output: Softmax over all classes

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Data
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000, 1)).float()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# Inference
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()
```

---

## Common Mistakes

### 1. Double Sigmoid

```python
# ❌ Wrong: Sigmoid applied twice
model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
criterion = nn.BCEWithLogitsLoss()  # Also applies sigmoid!

# ✅ Correct: Choose one approach
model = nn.Sequential(nn.Linear(10, 1))
criterion = nn.BCEWithLogitsLoss()
```

### 2. Wrong Target Shape

```python
# ❌ Wrong: Target shape mismatch
output = model(x)  # Shape: [32, 1]
target = labels    # Shape: [32]
loss = criterion(output, target)  # Error!

# ✅ Correct: Match shapes
target = labels.unsqueeze(1)  # Shape: [32, 1]
# or
output = model(x).squeeze()   # Shape: [32]
```

### 3. Integer Targets

```python
# ❌ Wrong: Integer targets
target = torch.tensor([0, 1, 1, 0])  # int64

# ✅ Correct: Float targets
target = torch.tensor([0., 1., 1., 0.])  # float32
```

---

## Key Takeaways

1. **Use BCE for binary classification**, not MSE
2. **BCE + Sigmoid creates convex loss surface** — guaranteed convergence
3. **BCE penalizes confident wrong predictions** — fast error correction
4. **Use BCEWithLogitsLoss** — more stable than separate sigmoid + BCE
5. **Match output and target shapes** — common source of errors

---

*Understanding why BCE works helps you choose the right loss function and debug training issues. This foundation extends to multi-class classification with Cross Entropy.*
