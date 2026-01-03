# 📊 Exponentially Weighted Moving Average (EWMA)

To understand modern optimizers like Momentum, RMSProp, and Adam, you must first understand **EWMA**. This mathematical technique allows an algorithm to "remember" past data while prioritizing the present.

---

## What is EWMA?

EWMA is a method for computing a running average that gives more weight to recent values and less weight to older values. Unlike a simple average that treats all values equally, EWMA creates a "fading memory" of past observations.

### Why It Matters for Optimization

In gradient descent, we want to:
- Smooth out noisy gradients
- Remember the general direction of descent
- Adapt to changing gradient magnitudes

EWMA provides the mathematical foundation for all of these.

---

## The EWMA Formula

```
V_t = β × V_{t-1} + (1 - β) × θ_t

Where:
- V_t = current weighted average
- V_{t-1} = previous weighted average
- θ_t = current observation (e.g., gradient)
- β = decay factor (0 < β < 1)
```

### Breaking It Down

The formula has two parts:

1. **β × V_{t-1}**: The "memory" term
   - Carries forward a fraction of the previous average
   - Higher β = more memory retained

2. **(1 - β) × θ_t**: The "update" term
   - Incorporates the current observation
   - Higher β = less weight on current value

---

## The Decay Factor (β)

The decay factor β controls how much "memory" the average has:

| β Value | Memory Length | Behavior |
|---------|---------------|----------|
| 0.9 | ~10 values | Moderate smoothing |
| 0.99 | ~100 values | Heavy smoothing |
| 0.999 | ~1000 values | Very heavy smoothing |
| 0.5 | ~2 values | Minimal smoothing |

### Approximate Memory Length

The effective number of values being averaged is approximately:

```
Memory ≈ 1 / (1 - β)

Examples:
- β = 0.9  → Memory ≈ 10 values
- β = 0.99 → Memory ≈ 100 values
- β = 0.5  → Memory ≈ 2 values
```

### Choosing β

| Scenario | Recommended β | Reason |
|----------|---------------|--------|
| Stable gradients | 0.9 - 0.99 | Smooth out minor noise |
| Noisy gradients | 0.5 - 0.9 | Respond faster to changes |
| Very noisy data | 0.99+ | Heavy smoothing needed |

---

## EWMA in Action: A Visual Example

Consider tracking gradients over 10 iterations with β = 0.9:

```
Iteration | Gradient | EWMA (V_t)
----------|----------|------------
    1     |   10     |   1.0
    2     |   12     |   2.1
    3     |   8      |   2.69
    4     |   15     |   3.92
    5     |   11     |   4.63
    6     |   9      |   5.07
    7     |   13     |   5.86
    8     |   10     |   6.27
    9     |   14     |   7.05
   10     |   12     |   7.54
```

Notice how:
- EWMA starts low (bias toward zero initially)
- Gradually approaches the true average (~11.4)
- Smooths out the fluctuations in individual gradients

---

## Bias Correction

### The Problem

EWMA has a "cold start" problem. Early values are biased toward zero because V_0 = 0:

```
V_1 = 0.9 × 0 + 0.1 × 10 = 1.0  (should be closer to 10!)
```

### The Solution

Apply bias correction:

```
V̂_t = V_t / (1 - β^t)

Where t is the iteration number
```

### How It Works

| Iteration | β^t | 1 - β^t | Correction Factor |
|-----------|-----|---------|-------------------|
| 1 | 0.9 | 0.1 | 10× |
| 2 | 0.81 | 0.19 | 5.3× |
| 5 | 0.59 | 0.41 | 2.4× |
| 10 | 0.35 | 0.65 | 1.5× |
| 100 | ~0 | ~1 | ~1× |

Early iterations get large corrections; later iterations need almost none.

---

## EWMA in Optimizers

### In Momentum

EWMA tracks the **mean of gradients**:
```
v_t = β × v_{t-1} + (1 - β) × gradient
```
This creates "velocity" that accelerates training.

### In RMSProp

EWMA tracks the **mean of squared gradients**:
```
s_t = β × s_{t-1} + (1 - β) × gradient²
```
This adapts learning rates per parameter.

### In Adam

EWMA tracks **both**:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × gradient      # First moment
v_t = β₂ × v_{t-1} + (1 - β₂) × gradient²     # Second moment
```
Combining momentum and adaptive learning rates.

---

## Implementation Example

```python
def ewma_update(current_value, new_observation, beta):
    """
    Update EWMA with a new observation.
    
    Args:
        current_value: Previous EWMA value (V_{t-1})
        new_observation: New data point (θ_t)
        beta: Decay factor (0 < β < 1)
    
    Returns:
        Updated EWMA value (V_t)
    """
    return beta * current_value + (1 - beta) * new_observation


def ewma_with_bias_correction(current_value, new_observation, beta, t):
    """
    Update EWMA with bias correction.
    
    Args:
        current_value: Previous EWMA value
        new_observation: New data point
        beta: Decay factor
        t: Current iteration (starting from 1)
    
    Returns:
        Bias-corrected EWMA value
    """
    v_t = beta * current_value + (1 - beta) * new_observation
    v_corrected = v_t / (1 - beta ** t)
    return v_t, v_corrected
```

---

## Key Takeaways

1. **EWMA creates a "fading memory"** of past values
2. **β controls the memory length**: higher β = longer memory
3. **Bias correction fixes the cold start** problem in early iterations
4. **EWMA is the foundation** of Momentum, RMSProp, and Adam
5. **Understanding EWMA** helps you tune optimizer hyperparameters

---

*EWMA is simple but powerful. Once you understand it, the mechanics of modern optimizers become much clearer.*
