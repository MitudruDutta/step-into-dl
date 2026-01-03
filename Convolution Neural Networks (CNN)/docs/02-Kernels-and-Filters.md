# 🔍 Kernels and Filters

## What are Kernels and Filters?

A **kernel** (also called a **filter**) is a small matrix of learnable weights that slides across the input image to detect specific features. The terms "kernel" and "filter" are often used interchangeably, though technically a filter can contain multiple kernels (one per input channel).

---

## How Filters Detect Features

### The Convolution Process

When a filter slides across an image, it performs element-wise multiplication followed by summation at each position:

```
Image Patch        Filter           Result
┌─────────┐      ┌─────────┐
│ 10 20 30│      │ -1  0  1│
│ 40 50 60│  ×   │ -2  0  2│  =  Sum of products
│ 70 80 90│      │ -1  0  1│
└─────────┘      └─────────┘

= (10×-1) + (20×0) + (30×1) + (40×-2) + (50×0) + (60×2) + (70×-1) + (80×0) + (90×1)
= -10 + 0 + 30 - 80 + 0 + 120 - 70 + 0 + 90
= 80
```

The result is high when the image patch matches the pattern encoded in the filter.

---

## Classic Hand-Crafted Filters

Before deep learning, computer vision relied on hand-designed filters. Understanding these helps build intuition for what CNNs learn.

### Edge Detection Filters

**Sobel Filter (Horizontal Edges)**
```
┌────────────┐
│ -1  -2  -1 │
│  0   0   0 │
│  1   2   1 │
└────────────┘
Detects horizontal edges (transitions from dark to light vertically)
```

**Sobel Filter (Vertical Edges)**
```
┌────────────┐
│ -1   0   1 │
│ -2   0   2 │
│ -1   0   1 │
└────────────┘
Detects vertical edges (transitions from dark to light horizontally)
```

### Sharpening Filter
```
┌────────────┐
│  0  -1   0 │
│ -1   5  -1 │
│  0  -1   0 │
└────────────┘
Enhances edges and details
```

### Blur Filter (Box Blur)
```
┌──────────────────┐
│ 1/9  1/9  1/9    │
│ 1/9  1/9  1/9    │
│ 1/9  1/9  1/9    │
└──────────────────┘
Averages neighboring pixels, smoothing the image
```

### Gaussian Blur
```
┌──────────────────────┐
│ 1/16  2/16  1/16     │
│ 2/16  4/16  2/16     │
│ 1/16  2/16  1/16     │
└──────────────────────┘
Weighted average, more natural smoothing
```

---

## Learned Filters in CNNs

Unlike hand-crafted filters, CNN filters are **learned from data** through backpropagation. The network discovers which patterns are useful for the task.

### What CNNs Learn

**Layer 1 Filters** (typically look like Gabor filters):
```
Common patterns learned:
- Horizontal edges at various angles
- Vertical edges at various angles  
- Color blobs (for RGB images)
- Gradient detectors
```

**Deeper Layer Filters**:
- Combinations of earlier features
- Texture patterns
- Object parts
- Abstract concepts

### Visualization of Learned Filters

```
Layer 1 (Edge Detectors):
┌───┐ ┌───┐ ┌───┐ ┌───┐
│ / │ │ \ │ │ — │ │ | │
└───┘ └───┘ └───┘ └───┘

Layer 2 (Corners, Textures):
┌───┐ ┌───┐ ┌───┐
│ ┌ │ │ ┐ │ │ # │
└───┘ └───┘ └───┘

Layer 3+ (Complex Patterns):
┌─────┐ ┌─────┐
│ 👁️  │ │ 🔵  │
└─────┘ └─────┘
```

---

## Filter Dimensions

### For 2D Images

```
Single Channel Input (Grayscale):
  Filter shape: (kernel_height, kernel_width)
  Example: 3×3 filter

Multi-Channel Input (RGB):
  Filter shape: (kernel_height, kernel_width, input_channels)
  Example: 3×3×3 filter for RGB image
  
  The filter has separate weights for each channel,
  but produces a single output value per position.
```

### Multiple Filters

```
Input: H × W × C_in
Filters: K filters, each of size (k × k × C_in)
Output: H' × W' × K

Example:
  Input: 32×32×3 (RGB image)
  32 filters of size 3×3×3
  Output: 30×30×32 (32 feature maps)
```

---

## Common Kernel Sizes

| Size | Use Case | Receptive Field |
|------|----------|-----------------|
| **1×1** | Channel mixing, dimensionality reduction | 1 pixel |
| **3×3** | Most common, good balance | 3×3 pixels |
| **5×5** | Larger features, less common now | 5×5 pixels |
| **7×7** | Often in first layer only | 7×7 pixels |

### Why 3×3 is Preferred

Two 3×3 convolutions have the same receptive field as one 5×5, but with fewer parameters:

```
5×5 convolution: 5 × 5 = 25 parameters
Two 3×3 convolutions: 3 × 3 + 3 × 3 = 18 parameters

Same receptive field, fewer parameters, more non-linearity!
```

---

## 1×1 Convolutions

Despite their small size, 1×1 convolutions are powerful:

### Purpose

1. **Dimensionality Reduction**: Reduce number of channels
2. **Dimensionality Expansion**: Increase number of channels
3. **Add Non-linearity**: When followed by activation
4. **Cross-Channel Interaction**: Mix information across channels

### Example

```
Input: 56×56×256
1×1 Conv with 64 filters
Output: 56×56×64

Reduced channels from 256 to 64 without changing spatial dimensions
```

---

## Depthwise Separable Convolutions

A more efficient alternative to standard convolutions, used in MobileNet and EfficientNet.

### Standard Convolution

```
Input: H × W × C_in
Filter: k × k × C_in × C_out
Operations: H × W × k × k × C_in × C_out
```

### Depthwise Separable Convolution

**Step 1: Depthwise Convolution**
```
Apply one k×k filter per input channel
Operations: H × W × k × k × C_in
```

**Step 2: Pointwise Convolution (1×1)**
```
Mix channels with 1×1 convolutions
Operations: H × W × C_in × C_out
```

### Efficiency Comparison

```
Standard: k² × C_in × C_out multiplications per pixel
Separable: k² × C_in + C_in × C_out multiplications per pixel

For k=3, C_in=256, C_out=256:
  Standard: 9 × 256 × 256 = 589,824
  Separable: 9 × 256 + 256 × 256 = 67,840
  
  ~8.7× fewer operations!
```

---

## Filter Initialization

How filters are initialized affects training:

### Common Initialization Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Xavier/Glorot** | Scaled by fan_in + fan_out | Sigmoid, Tanh |
| **He/Kaiming** | Scaled by fan_in | ReLU, Leaky ReLU |
| **Random Normal** | Small random values | General purpose |

### Why Initialization Matters

- **Too small**: Gradients vanish, slow learning
- **Too large**: Gradients explode, unstable training
- **Just right**: Maintains variance through layers

---

## Visualizing What Filters Learn

### Techniques

1. **Direct Visualization**: Plot filter weights as images (works for first layer)
2. **Activation Maximization**: Generate input that maximally activates a filter
3. **Gradient-based Methods**: See which input regions affect the filter most
4. **Feature Map Visualization**: Show output of each filter for a given input

### Interpreting First Layer Filters

```
For a network trained on natural images:

Filter 1: Responds to horizontal edges
Filter 2: Responds to vertical edges
Filter 3: Responds to diagonal edges (45°)
Filter 4: Responds to diagonal edges (135°)
Filter 5: Responds to red color
Filter 6: Responds to green color
...
```

---

## Key Takeaways

1. **Filters are learnable**: CNNs discover useful patterns automatically
2. **Small filters are efficient**: 3×3 is the sweet spot
3. **Hierarchical features**: Early layers detect simple patterns, deep layers detect complex ones
4. **1×1 convolutions**: Powerful for channel manipulation
5. **Depthwise separable**: Trade-off between accuracy and efficiency

---

*Next: [03-Padding-and-Strides.md](03-Padding-and-Strides.md) — Control output dimensions*
