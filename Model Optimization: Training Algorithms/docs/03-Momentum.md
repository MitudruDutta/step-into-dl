# 🏔️ Gradient Descent with Momentum

Standard Gradient Descent can be slow and "zig-zag" wildly in narrow valleys. Momentum solves this by building up "velocity" in directions of consistent gradient, dramatically accelerating convergence.

---

## The Problem with Vanilla Gradient Descent

### Oscillation in Narrow Valleys

Imagine a loss landscape shaped like a narrow valley:
- Steep walls on the sides (large gradients)
- Gentle slope toward the minimum (small gradients)

Vanilla GD will:
1. Take large steps perpendicular to the valley (bouncing off walls)
2. Take tiny steps along the valley (toward the minimum)
3. Result: Slow, zig-zagging path

```
Loss Landscape (top view):

    ←  steep  →
    ___________
   /           \
  /   zig-zag   \    ↓ gentle slope
 /    path →     \
/                 \
\_________________/
      minimum
```

### The Physics Analogy

Think of a ball rolling down a hill:
- Without momentum: Ball stops and changes direction instantly
- With momentum: Ball builds up speed and rolls through small bumps

---

## How Momentum Works

### The Algorithm

```
# Initialize velocity
v = 0

# For each iteration:
v = β × v + (1 - β) × gradient    # Update velocity
weights = weights - lr × v         # Update weights
```

Or equivalently (classical formulation):
```
v = β × v + lr × gradient
weights = weights - v
```

### Breaking It Down

| Term | Meaning |
|------|---------|
| `v` | Velocity (accumulated gradient direction) |
| `β` | Momentum coefficient (typically 0.9) |
| `gradient` | Current gradient |
| `lr` | Learning rate |

### The Two Components

1. **β × v**: Carry forward previous velocity
   - Maintains direction from past gradients
   - Creates "inertia" in the optimization

2. **(1 - β) × gradient**: Add current gradient
   - Incorporates new information
   - Adjusts direction based on current position

---

## Key Benefits

### 1. Acceleration

When gradients consistently point in the same direction, momentum builds up:

```
Iteration 1: v = 0.1 × g
Iteration 2: v = 0.9 × (0.1g) + 0.1g = 0.19g
Iteration 3: v = 0.9 × (0.19g) + 0.1g = 0.27g
...
Eventually: v ≈ g (full gradient)
```

The effective step size increases, accelerating convergence.

### 2. Stability (Dampening Oscillations)

When gradients oscillate (point in opposite directions):

```
Iteration 1: v = +g
Iteration 2: v = 0.9 × (+g) + 0.1 × (-g) = 0.8g
Iteration 3: v = 0.9 × (0.8g) + 0.1 × (+g) = 0.82g
```

Oscillations cancel out, smoothing the path.

### 3. Escaping Local Minima

Momentum provides the "push" needed to:
- Roll through small local minima
- Cross flat regions (saddle points)
- Continue moving when gradients are small

---

## The Momentum Coefficient (β)

### Typical Values

| β Value | Behavior | Use Case |
|---------|----------|----------|
| 0.9 | Standard momentum | Most problems |
| 0.99 | Heavy momentum | Very noisy gradients |
| 0.5 | Light momentum | When you need responsiveness |

### Effect of β

**High β (0.99)**:
- Very smooth updates
- Slow to change direction
- Good for consistent gradients

**Low β (0.5)**:
- More responsive to current gradient
- Less smoothing
- Good for rapidly changing landscapes

### Choosing β

Start with **β = 0.9** (the default). Adjust if:
- Training oscillates → increase β
- Training is too slow to adapt → decrease β

---

## Momentum vs. Vanilla GD: Visual Comparison

### Vanilla Gradient Descent
```
Start → ↗ → ↙ → ↗ → ↙ → ↗ → ↙ → Minimum
        (zig-zag path, many iterations)
```

### With Momentum
```
Start → → → → → → Minimum
        (smooth path, fewer iterations)
```

---

## Implementation

```python
class SGDWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.beta = momentum
        self.velocity = None
    
    def update(self, weights, gradients):
        # Initialize velocity on first call
        if self.velocity is None:
            self.velocity = torch.zeros_like(weights)
        
        # Update velocity
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradients
        
        # Update weights
        weights = weights - self.lr * self.velocity
        
        return weights
```

### PyTorch Usage

```python
import torch.optim as optim

# Create optimizer with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()  # Momentum is applied automatically
```

---

## Nesterov Momentum (NAG)

An improved version that "looks ahead" before computing the gradient:

### Standard Momentum
```
v = β × v + gradient(weights)
weights = weights - lr × v
```

### Nesterov Momentum
```
v = β × v + gradient(weights - lr × β × v)  # Gradient at "lookahead" position
weights = weights - lr × v
```

### Why Nesterov is Better

- Computes gradient at the approximate future position
- Provides a "correction" if momentum is going the wrong way
- Often converges faster than standard momentum

### PyTorch Usage

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## When to Use Momentum

### Good For

- Deep networks with many layers
- Problems with noisy gradients
- Loss landscapes with narrow valleys
- When vanilla SGD oscillates

### Not Ideal For

- Very simple problems (overhead not worth it)
- When you need very precise control
- Problems where Adam works better out-of-box

---

## Key Takeaways

1. **Momentum builds velocity** in consistent gradient directions
2. **Accelerates convergence** by taking larger effective steps
3. **Dampens oscillations** by averaging out zig-zags
4. **β = 0.9** is a good default starting point
5. **Nesterov momentum** often works better than standard momentum

---

*Momentum is a simple but powerful improvement over vanilla gradient descent. It's the foundation for understanding more advanced optimizers like Adam.*
