# 📐 Padding and Strides

## Overview

**Padding** and **stride** are two crucial hyperparameters that control the spatial dimensions of CNN outputs. Understanding them is essential for designing CNN architectures.

---

## Stride

### What is Stride?

**Stride** determines how many pixels the filter moves at each step. A stride of 1 means the filter moves one pixel at a time; a stride of 2 means it skips every other pixel.

### Stride Visualization

```
Stride = 1 (Default):
┌─────────────┐
│ ■ ■ ■ □ □   │  Step 1: Position (0,0)
│ ■ ■ ■ □ □   │
│ ■ ■ ■ □ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
└─────────────┘

┌─────────────┐
│ □ ■ ■ ■ □   │  Step 2: Position (0,1)
│ □ ■ ■ ■ □   │
│ □ ■ ■ ■ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
└─────────────┘

Stride = 2:
┌─────────────┐
│ ■ ■ ■ □ □   │  Step 1: Position (0,0)
│ ■ ■ ■ □ □   │
│ ■ ■ ■ □ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
└─────────────┘

┌─────────────┐
│ □ □ ■ ■ ■   │  Step 2: Position (0,2) — skipped (0,1)
│ □ □ ■ ■ ■   │
│ □ □ ■ ■ ■   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
└─────────────┘
```

### Effect of Stride on Output Size

```
Input: 6×6, Kernel: 3×3

Stride 1: Output = 4×4 (moves 4 times in each direction)
Stride 2: Output = 2×2 (moves 2 times in each direction)
Stride 3: Output = 2×2 (moves 2 times, with some overlap lost)
```

### When to Use Different Strides

| Stride | Use Case |
|--------|----------|
| **1** | Default, preserves spatial information |
| **2** | Downsampling (alternative to pooling) |
| **>2** | Aggressive downsampling, rarely used |

---

## Padding

### What is Padding?

**Padding** adds extra pixels (usually zeros) around the border of the input image before convolution. This controls the output size and preserves edge information.

### Types of Padding

**No Padding (Valid)**
```
Input: 5×5, Kernel: 3×3
┌─────────────┐
│ □ □ □ □ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
│ □ □ □ □ □   │
└─────────────┘
Output: 3×3 (shrinks)
```

**Same Padding (Zero Padding)**
```
Input: 5×5, Kernel: 3×3, Padding: 1
┌─────────────────┐
│ 0 0 0 0 0 0 0   │  ← Added zeros
│ 0 □ □ □ □ □ 0   │
│ 0 □ □ □ □ □ 0   │
│ 0 □ □ □ □ □ 0   │
│ 0 □ □ □ □ □ 0   │
│ 0 □ □ □ □ □ 0   │
│ 0 0 0 0 0 0 0   │  ← Added zeros
└─────────────────┘
Output: 5×5 (same as input)
```

### Padding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Zero (Constant)** | Pad with zeros | Most common |
| **Reflect** | Mirror the edge pixels | Avoid edge artifacts |
| **Replicate** | Repeat edge pixels | Natural images |
| **Circular** | Wrap around | Periodic signals |

---

## Output Size Formula

The fundamental formula for calculating output dimensions:

```
Output Size = ⌊(Input Size - Kernel Size + 2×Padding) / Stride⌋ + 1
```

### Examples

**Example 1: No Padding, Stride 1**
```
Input: 32×32
Kernel: 3×3
Padding: 0
Stride: 1

Output = (32 - 3 + 0) / 1 + 1 = 30×30
```

**Example 2: Same Padding, Stride 1**
```
Input: 32×32
Kernel: 3×3
Padding: 1
Stride: 1

Output = (32 - 3 + 2) / 1 + 1 = 32×32 (same!)
```

**Example 3: Stride 2 Downsampling**
```
Input: 32×32
Kernel: 3×3
Padding: 1
Stride: 2

Output = (32 - 3 + 2) / 2 + 1 = 16×16 (halved!)
```

**Example 4: 7×7 Kernel (Common in First Layer)**
```
Input: 224×224
Kernel: 7×7
Padding: 3
Stride: 2

Output = (224 - 7 + 6) / 2 + 1 = 112×112
```

---

## Calculating Padding for "Same" Output

To keep output size equal to input size (with stride 1):

```
Padding = (Kernel Size - 1) / 2

For 3×3 kernel: Padding = (3-1)/2 = 1
For 5×5 kernel: Padding = (5-1)/2 = 2
For 7×7 kernel: Padding = (7-1)/2 = 3
```

Note: This only works for odd kernel sizes. Even kernels require asymmetric padding.

---

## Common Configurations

### Configuration 1: Preserve Dimensions
```
Kernel: 3×3
Padding: 1
Stride: 1
Result: Output size = Input size
```

### Configuration 2: Halve Dimensions
```
Kernel: 3×3
Padding: 1
Stride: 2
Result: Output size = Input size / 2
```

### Configuration 3: Aggressive Downsampling (First Layer)
```
Kernel: 7×7
Padding: 3
Stride: 2
Result: Output size = Input size / 2
```

---

## Why Padding Matters

### Problem: Shrinking Feature Maps

Without padding, each convolution shrinks the spatial dimensions:

```
Layer 1: 32×32 → 30×30 (3×3 conv)
Layer 2: 30×30 → 28×28 (3×3 conv)
Layer 3: 28×28 → 26×26 (3×3 conv)
...
After 15 layers: 2×2 (too small!)
```

### Problem: Edge Information Loss

Without padding, edge pixels contribute to fewer output values:

```
Corner pixel: Used in 1 convolution
Edge pixel: Used in 3 convolutions  
Center pixel: Used in 9 convolutions (for 3×3 kernel)

Edge information is underrepresented!
```

### Solution: Use Padding

With padding=1 for 3×3 kernels:
- Spatial dimensions preserved
- All pixels contribute equally
- Can build deeper networks

---

## Stride vs Pooling for Downsampling

Both can reduce spatial dimensions, but they work differently:

### Strided Convolution
```
Pros:
- Learnable downsampling
- Fewer layers needed
- Can capture more information

Cons:
- May miss fine details
- Aliasing possible
```

### Pooling
```
Pros:
- Provides translation invariance
- No additional parameters
- Well-understood behavior

Cons:
- Fixed operation (not learned)
- May lose spatial information
```

### Modern Trend

Many modern architectures (ResNet, EfficientNet) prefer strided convolutions over pooling for downsampling, except for the final global pooling.

---

## Practical Guidelines

### For Feature Extraction Layers
```
Use: kernel=3×3, padding=1, stride=1
Result: Preserves spatial dimensions
When to downsample: Use stride=2 or pooling
```

### For First Layer (Large Images)
```
Use: kernel=7×7, padding=3, stride=2
Result: Quickly reduces dimensions
Why: 224×224 → 112×112 in one layer
```

### For Bottleneck Layers
```
Use: kernel=1×1, padding=0, stride=1
Result: Changes channels, not spatial size
Why: Efficient channel manipulation
```

---

## Dimension Tracking Example

Let's trace dimensions through a simple CNN:

```
Input: 224×224×3

Conv1: 7×7, padding=3, stride=2, 64 filters
  → (224-7+6)/2+1 = 112×112×64

Pool1: 3×3, stride=2
  → 56×56×64

Conv2: 3×3, padding=1, stride=1, 128 filters
  → 56×56×128

Pool2: 2×2, stride=2
  → 28×28×128

Conv3: 3×3, padding=1, stride=1, 256 filters
  → 28×28×256

Pool3: 2×2, stride=2
  → 14×14×256

Conv4: 3×3, padding=1, stride=1, 512 filters
  → 14×14×512

Global Avg Pool:
  → 1×1×512

Flatten:
  → 512

Dense → Output
```

---

## Summary

| Concept | Purpose | Common Values |
|---------|---------|---------------|
| **Stride** | Control step size, downsampling | 1 (preserve), 2 (halve) |
| **Padding** | Control output size, preserve edges | 0 (valid), k//2 (same) |

Key formulas:
- Output = (Input - Kernel + 2×Padding) / Stride + 1
- Same padding = (Kernel - 1) / 2

---

*Next: [04-Pooling-Layers.md](04-Pooling-Layers.md) — Downsampling and invariance*
