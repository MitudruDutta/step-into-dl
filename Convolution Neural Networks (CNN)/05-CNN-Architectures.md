# 🏛️ CNN Architectures

## Evolution of CNN Architectures

CNN architectures have evolved dramatically since the 1990s. Each breakthrough introduced new concepts that pushed the boundaries of what's possible in computer vision.

```
Timeline:
1998: LeNet-5      → First successful CNN
2012: AlexNet      → Deep learning revolution
2014: VGGNet       → Deeper is better
2014: GoogLeNet    → Inception modules
2015: ResNet       → Skip connections
2017: DenseNet     → Dense connections
2019: EfficientNet → Compound scaling
```

---

## LeNet-5 (1998)

The **pioneer** of CNNs, designed by Yann LeCun for handwritten digit recognition.

### Architecture

```
Input: 32×32×1 (grayscale)
       ↓
Conv1: 6 filters, 5×5 → 28×28×6
       ↓
Pool1: 2×2, stride 2 → 14×14×6
       ↓
Conv2: 16 filters, 5×5 → 10×10×16
       ↓
Pool2: 2×2, stride 2 → 5×5×16
       ↓
Flatten → 400
       ↓
FC1: 120 neurons
       ↓
FC2: 84 neurons
       ↓
Output: 10 classes
```

### Key Contributions
- Demonstrated that CNNs can learn useful features
- Introduced the Conv → Pool → Conv → Pool pattern
- Used tanh activation (ReLU wasn't popular yet)

### Limitations
- Small by modern standards (~60K parameters)
- Only works on small grayscale images
- Shallow architecture limits feature complexity

---

## AlexNet (2012)

The architecture that **sparked the deep learning revolution** by winning ImageNet 2012 with a huge margin.

### Architecture

```
Input: 227×227×3 (RGB)
       ↓
Conv1: 96 filters, 11×11, stride 4 → 55×55×96
       ↓
MaxPool: 3×3, stride 2 → 27×27×96
       ↓
Conv2: 256 filters, 5×5, padding 2 → 27×27×256
       ↓
MaxPool: 3×3, stride 2 → 13×13×256
       ↓
Conv3: 384 filters, 3×3, padding 1 → 13×13×384
       ↓
Conv4: 384 filters, 3×3, padding 1 → 13×13×384
       ↓
Conv5: 256 filters, 3×3, padding 1 → 13×13×256
       ↓
MaxPool: 3×3, stride 2 → 6×6×256
       ↓
Flatten → 9216
       ↓
FC1: 4096 → FC2: 4096 → Output: 1000
```

### Key Contributions
- **ReLU activation**: Faster training than tanh/sigmoid
- **Dropout**: Regularization to prevent overfitting
- **GPU training**: Split across 2 GPUs
- **Data augmentation**: Image translations, reflections
- **Local Response Normalization** (later replaced by BatchNorm)

### Parameters
~60 million parameters (mostly in FC layers)

---

## VGGNet (2014)

Showed that **depth matters** — using very small (3×3) filters consistently.

### VGG-16 Architecture

```
Input: 224×224×3
       ↓
[Conv3-64] × 2 → MaxPool → 112×112×64
       ↓
[Conv3-128] × 2 → MaxPool → 56×56×128
       ↓
[Conv3-256] × 3 → MaxPool → 28×28×256
       ↓
[Conv3-512] × 3 → MaxPool → 14×14×512
       ↓
[Conv3-512] × 3 → MaxPool → 7×7×512
       ↓
Flatten → FC-4096 → FC-4096 → Output-1000
```

### Key Contributions
- **3×3 filters only**: Two 3×3 convs = one 5×5 receptive field, fewer parameters
- **Uniform architecture**: Easy to understand and implement
- **Deeper networks**: 16-19 layers (vs AlexNet's 8)

### Why 3×3 Filters?
```
5×5 conv: 25 parameters
Two 3×3 convs: 9 + 9 = 18 parameters

Same receptive field, fewer parameters, more non-linearity!
```

### Limitations
- 138 million parameters (very large)
- Slow to train
- FC layers are parameter-heavy

---

## GoogLeNet / Inception (2014)

Introduced the **Inception module** — parallel convolutions at multiple scales.

### Inception Module

```
         Input
           │
    ┌──────┼──────┬──────┐
    │      │      │      │
   1×1    1×1    1×1   3×3
  conv   conv   conv  MaxPool
    │      │      │      │
    │     3×3    5×5    1×1
    │    conv   conv   conv
    │      │      │      │
    └──────┴──────┴──────┘
           │
      Concatenate
           │
        Output
```

### Key Contributions
- **Multi-scale processing**: Capture features at different scales simultaneously
- **1×1 convolutions**: Reduce dimensions before expensive 3×3 and 5×5 convs
- **No FC layers**: Global Average Pooling instead
- **Auxiliary classifiers**: Help gradient flow in deep networks

### Parameters
Only ~5 million parameters (vs VGG's 138M)!

---

## ResNet (2015)

Solved the **degradation problem** with skip connections, enabling extremely deep networks.


### The Degradation Problem

```
Observation: Deeper networks should be at least as good as shallow ones
Reality: After ~20 layers, adding more layers DECREASED accuracy

Why? Not overfitting — training error also increased!
The problem: Optimization difficulty, not capacity
```

### Residual Block

```
        Input (x)
           │
           ├─────────────────┐
           │                 │
        Conv 3×3             │
           │                 │
        BatchNorm            │
           │                 │
          ReLU               │
           │                 │
        Conv 3×3             │
           │                 │
        BatchNorm            │
           │                 │
           +←────────────────┘  (Skip Connection)
           │
          ReLU
           │
        Output: F(x) + x
```

### Why Skip Connections Work

```
Without skip: Network must learn H(x) directly
With skip: Network learns F(x) = H(x) - x (the residual)

If identity mapping is optimal, F(x) = 0 is easier to learn than H(x) = x
```

### ResNet Variants

| Model | Layers | Parameters | Top-1 Accuracy |
|-------|--------|------------|----------------|
| ResNet-18 | 18 | 11.7M | 69.8% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |
| ResNet-152 | 152 | 60.2M | 78.3% |

### Bottleneck Block (ResNet-50+)

```
Input (256 channels)
       │
    1×1 Conv (64) ← Reduce dimensions
       │
    3×3 Conv (64) ← Process
       │
    1×1 Conv (256) ← Restore dimensions
       │
       + ← Skip connection
       │
    Output
```

---

## DenseNet (2017)

Takes skip connections further — **every layer connects to every other layer**.

### Dense Block

```
Layer 1 ──────────────────────────────────┐
    │                                     │
Layer 2 ←─────────────────────────────────┤
    │                                     │
Layer 3 ←─────────────────────────────────┤
    │                                     │
Layer 4 ←─────────────────────────────────┘
```

### Key Contributions
- **Feature reuse**: All previous features available to each layer
- **Gradient flow**: Direct paths from loss to early layers
- **Compact models**: Fewer parameters than ResNet for same accuracy
- **Growth rate**: Each layer adds k feature maps (typically k=32)

---

## EfficientNet (2019)

Introduced **compound scaling** — systematically scale depth, width, and resolution together.

### Scaling Dimensions

```
Width (w): Number of channels
Depth (d): Number of layers
Resolution (r): Input image size

Compound scaling:
  depth = α^φ
  width = β^φ
  resolution = γ^φ

  where α × β² × γ² ≈ 2 (to double FLOPS)
```

### EfficientNet Family

| Model | Resolution | Parameters | Top-1 Accuracy |
|-------|------------|------------|----------------|
| B0 | 224 | 5.3M | 77.1% |
| B1 | 240 | 7.8M | 79.1% |
| B2 | 260 | 9.2M | 80.1% |
| B3 | 300 | 12M | 81.6% |
| B4 | 380 | 19M | 82.9% |
| B5 | 456 | 30M | 83.6% |
| B6 | 528 | 43M | 84.0% |
| B7 | 600 | 66M | 84.3% |

### Key Innovation
- **MBConv blocks**: Mobile inverted bottleneck with squeeze-and-excitation
- **Neural Architecture Search**: Found optimal base architecture
- **Compound scaling**: Balanced scaling outperforms single-dimension scaling

---

## Architecture Comparison

| Architecture | Year | Depth | Parameters | Key Innovation |
|--------------|------|-------|------------|----------------|
| LeNet-5 | 1998 | 5 | 60K | First CNN |
| AlexNet | 2012 | 8 | 60M | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 16 | 138M | 3×3 filters only |
| GoogLeNet | 2014 | 22 | 5M | Inception modules |
| ResNet-50 | 2015 | 50 | 25M | Skip connections |
| DenseNet-121 | 2017 | 121 | 8M | Dense connections |
| EfficientNet-B0 | 2019 | - | 5M | Compound scaling |

---

## Choosing an Architecture

### For Learning/Prototyping
- **VGG-16**: Simple, easy to understand
- **ResNet-18/34**: Good balance of simplicity and performance

### For Production (Accuracy Focus)
- **ResNet-50/101**: Reliable, well-understood
- **EfficientNet-B4/B5**: Best accuracy/efficiency trade-off

### For Mobile/Edge Deployment
- **MobileNet**: Designed for efficiency
- **EfficientNet-B0**: Good accuracy with few parameters

### For Transfer Learning
- **ResNet-50**: Most pre-trained weights available
- **EfficientNet**: Better features for fine-tuning

---

## Summary

The evolution of CNN architectures shows key principles:

1. **Deeper is better** (with proper techniques)
2. **Skip connections** enable very deep networks
3. **Small filters (3×3)** are efficient and effective
4. **Global Average Pooling** replaces FC layers
5. **Compound scaling** balances depth, width, resolution

---

*Next: [06-Data-Augmentation.md](06-Data-Augmentation.md) — Expand your training data*
