# 🎯 Categorical Cross Entropy

When your model needs to predict more than two classes (e.g., classifying an image as Cat, Dog, or Bird), you switch to **Categorical Cross Entropy** (often just called Cross Entropy in PyTorch).

---

## The Multi-Class Classification Problem

Multi-class classification predicts one of N mutually exclusive classes:
- Digit recognition (0-9) → 10 classes
- Image classification (cat, dog, bird) → 3 classes
- Sentiment (positive, neutral, negative) → 3 classes
- ImageNet classification → 1000 classes

**Key characteristic:** Each sample belongs to exactly one class.

---

## How Cross Entropy Works

### Step 1: Raw Scores (Logits)

The final layer outputs raw scores for each class:

```
Input → Network → [2.1, 0.5, -1.2]  (logits for 3 classes)
```

### Step 2: Softmax Conversion

Softmax converts logits to probabilities that sum to 1:

```
softmax([2.1, 0.5, -1.2]) = [0.72, 0.15, 0.03]
                            ↑
                            Sum = 1.0
```

### Step 3: Cross Entropy Loss

Compare predicted probabilities against true label:

```
True label: Class 0 (one-hot: [1, 0, 0])
Predicted:  [0.72, 0.15, 0.03]

CE Loss = -log(0.72) = 0.33
```

The loss is simply `-log(probability of true class)`.

---

## Cross Entropy Formula

```
CE = -Σ yᵢ × log(pᵢ)

Where:
- yᵢ = 1 for the true class, 0 otherwise
- pᵢ = predicted probability for class i
```

Since only the true class has y=1, this simplifies to:

```
CE = -log(p_true_class)
```

### Loss Values

| Predicted Probability | Cross Entropy Loss |
|----------------------|-------------------|
| 0.99 | 0.01 (very confident, correct) |
| 0.90 | 0.11 |
| 0.50 | 0.69 |
| 0.10 | 2.30 |
| 0.01 | 4.61 (very confident, wrong) |

Like BCE, Cross Entropy heavily penalizes confident wrong predictions.

---

## PyTorch Implementation

### nn.CrossEntropyLoss

```python
import torch
import torch.nn as nn

# Model outputs raw logits (NO softmax)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)  # 10 classes, no activation
)

criterion = nn.CrossEntropyLoss()

# Forward pass
logits = model(input_data)  # Shape: [batch_size, 10]
loss = criterion(logits, targets)  # targets: integer class indices
```

**Important:** `nn.CrossEntropyLoss` combines:
- `nn.LogSoftmax` (log of softmax)
- `nn.NLLLoss` (negative log likelihood)

This is more numerically stable than applying softmax separately.

### Target Format

```python
# Targets are INTEGER class indices, not one-hot
targets = torch.tensor([0, 2, 1, 0])  # Classes for 4 samples

# NOT one-hot encoded
# targets = torch.tensor([[1,0,0], [0,0,1], ...])  # Wrong!
```


---

## BCE vs. Cross Entropy Comparison

| Aspect | Binary Cross Entropy | Categorical Cross Entropy |
|--------|---------------------|--------------------------|
| **Classes** | 2 (binary) | 2+ (multi-class) |
| **Output Neurons** | 1 | N (one per class) |
| **Output Activation** | Sigmoid | Softmax (built into loss) |
| **Output Range** | Single value [0,1] | Vector summing to 1 |
| **Target Format** | Float (0.0 or 1.0) | Integer (class index) |
| **PyTorch Function** | `nn.BCELoss` | `nn.CrossEntropyLoss` |

### The Relationship

Binary Cross Entropy is a special case of Categorical Cross Entropy when there are exactly 2 classes. For binary problems, you can use either:

```python
# Option 1: BCE (1 output neuron)
output = sigmoid(model(x))  # Shape: [batch, 1]
loss = BCELoss(output, target)

# Option 2: CrossEntropy (2 output neurons)
logits = model(x)  # Shape: [batch, 2]
loss = CrossEntropyLoss(logits, target)  # target: 0 or 1
```

BCE with 1 neuron is more common for binary classification.

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model (10 classes for digits 0-9)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)  # No softmax!
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")
```

---

## Getting Predictions

```python
model.eval()
with torch.no_grad():
    logits = model(test_images)
    
    # Get probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # Get predicted class
    _, predicted_class = torch.max(logits, 1)
    # or
    predicted_class = torch.argmax(logits, dim=1)
    
    # Get top-k predictions
    top_probs, top_classes = torch.topk(probabilities, k=3, dim=1)
```

---

## Common Mistakes

### 1. Applying Softmax Before CrossEntropyLoss

```python
# ❌ Wrong: Double softmax
model = nn.Sequential(
    nn.Linear(784, 10),
    nn.Softmax(dim=1)  # Don't do this!
)
criterion = nn.CrossEntropyLoss()  # Already includes softmax

# ✅ Correct: No softmax in model
model = nn.Sequential(
    nn.Linear(784, 10)
)
criterion = nn.CrossEntropyLoss()
```

### 2. Wrong Target Type

```python
# ❌ Wrong: Float targets
targets = torch.tensor([0.0, 1.0, 2.0])

# ✅ Correct: Long (int64) targets
targets = torch.tensor([0, 1, 2])  # or .long()
```

### 3. One-Hot Encoded Targets

```python
# ❌ Wrong: One-hot targets
targets = torch.tensor([[1,0,0], [0,1,0], [0,0,1]])

# ✅ Correct: Class indices
targets = torch.tensor([0, 1, 2])
```

---

## Multi-Label vs Multi-Class

| Aspect | Multi-Class | Multi-Label |
|--------|-------------|-------------|
| **Labels per sample** | Exactly 1 | 0 or more |
| **Example** | Digit (0-9) | Image tags (sunny, beach, people) |
| **Output activation** | Softmax | Sigmoid (per label) |
| **Loss function** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Outputs sum to** | 1 | Independent |

For multi-label, treat each label as independent binary classification.

---

## Key Takeaways

1. **Use CrossEntropyLoss for multi-class** — one class per sample
2. **Don't apply softmax** — it's built into CrossEntropyLoss
3. **Targets are integer indices** — not one-hot encoded
4. **Use argmax for predictions** — get class with highest logit
5. **Apply softmax only for probabilities** — during inference if needed

---

*Cross Entropy is the standard loss for classification. Understanding its relationship to BCE helps you choose correctly for any classification problem.*
