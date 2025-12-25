# 🖼️ CNN Fundamentals

## What is a Convolutional Neural Network?

A **Convolutional Neural Network (CNN)** is a specialized type of neural network designed to process data with a grid-like topology, such as images. Unlike traditional fully connected networks that treat input as a flat vector, CNNs preserve and exploit the spatial structure of the data.

CNNs were inspired by the visual cortex of animals, where neurons respond to stimuli only in a restricted region of the visual field known as the receptive field.

---

## Why CNNs for Images?

### The Problem with Fully Connected Networks

Consider a simple 28×28 grayscale image (like MNIST digits):

```
Fully Connected Approach:
  Input: 28 × 28 = 784 pixels
  First hidden layer (256 neurons): 784 × 256 = 200,704 parameters
  
  For a 224×224 RGB image:
  Input: 224 × 224 × 3 = 150,528 pixels
  First hidden layer (256 neurons): 150,528 × 256 = 38,535,168 parameters!
```

This approach has three major problems:

1. **Parameter Explosion**: The number of weights grows rapidly with image size
2. **Loss of Spatial Information**: Flattening destroys the 2D structure
3. **No Translation Invariance**: A cat in the corner looks completely different from a cat in the center

### How CNNs Solve These Problems

| Problem | CNN Solution |
|---------|--------------|
| Too many parameters | **Parameter sharing** — same filter applied across entire image |
| Lost spatial info | **Local connectivity** — neurons connect only to nearby pixels |
| No translation invariance | **Convolution operation** — detects features regardless of position |

---

## The Convolution Operation

The core of CNNs is the **convolution operation**, which slides a small filter (kernel) across the input image to produce a feature map.

### How Convolution Works

```
Input Image (5×5)          Filter (3×3)         Output (3×3)
┌─────────────────┐        ┌─────────┐          ┌─────────┐
│ 1  2  3  4  5   │        │ 1  0  1 │          │ 12  16  │
│ 6  7  8  9  10  │   *    │ 0  1  0 │    =     │ 22  26  │
│ 11 12 13 14 15  │        │ 1  0  1 │          │ ...     │
│ 16 17 18 19 20  │        └─────────┘          └─────────┘
│ 21 22 23 24 25  │
└─────────────────┘

Calculation for top-left output:
(1×1) + (2×0) + (3×1) + (6×0) + (7×1) + (8×0) + (11×1) + (12×0) + (13×1)
= 1 + 0 + 3 + 0 + 7 + 0 + 11 + 0 + 13 = 35
```

### Key Properties of Convolution

1. **Sparse Connectivity**: Each output neuron connects to only a small region of the input
2. **Parameter Sharing**: The same filter weights are used across the entire image
3. **Equivariance to Translation**: If the input shifts, the output shifts by the same amount

---

## Feature Hierarchies

One of the most powerful aspects of CNNs is their ability to learn hierarchical features automatically.

### Layer-by-Layer Feature Learning

```
Layer 1 (Early):     Layer 2 (Middle):    Layer 3 (Deep):
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Edges     │      │   Textures  │      │   Objects   │
│   Corners   │  →   │   Patterns  │  →   │   Parts     │
│   Gradients │      │   Shapes    │      │   Faces     │
└─────────────┘      └─────────────┘      └─────────────┘
   Simple              Combinations         Complex
   Features            of Simple            Concepts
```

### What Each Layer Learns

| Layer Depth | Features Detected | Example |
|-------------|-------------------|---------|
| Layer 1 | Edges, colors, gradients | Horizontal/vertical lines |
| Layer 2 | Textures, simple shapes | Corners, circles |
| Layer 3 | Object parts | Eyes, wheels, windows |
| Layer 4+ | Whole objects, scenes | Faces, cars, buildings |

This hierarchical learning happens automatically through backpropagation — the network discovers which features are useful for the task.

---

## CNN Architecture Overview

A typical CNN consists of two main parts:

### 1. Feature Extraction (Convolutional Base)

```
Input → [Conv → ReLU → Pool] × N → Feature Maps
```

- **Convolutional layers**: Extract features using learnable filters
- **Activation (ReLU)**: Introduce non-linearity
- **Pooling layers**: Reduce spatial dimensions, add invariance

### 2. Classification (Fully Connected Head)

```
Feature Maps → Flatten → [Dense → ReLU] × M → Output
```

- **Flatten**: Convert 2D feature maps to 1D vector
- **Dense layers**: Learn to classify based on extracted features
- **Output layer**: Produce final predictions (softmax for classification)

### Complete Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    CONVOLUTIONAL BASE                        │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐          │
│  │ Conv   │   │ Conv   │   │ Conv   │   │ Conv   │          │
│  │ 32     │ → │ 64     │ → │ 128    │ → │ 256    │          │
│  │ filters│   │ filters│   │ filters│   │ filters│          │
│  └────────┘   └────────┘   └────────┘   └────────┘          │
│      ↓            ↓            ↓            ↓                │
│   Pool 2×2    Pool 2×2    Pool 2×2    Pool 2×2              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION HEAD                        │
│  ┌────────┐   ┌────────┐   ┌────────┐                       │
│  │Flatten │ → │Dense   │ → │Dense   │ → Predictions         │
│  │        │   │512     │   │10      │   (Softmax)           │
│  └────────┘   └────────┘   └────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Receptive Field

The **receptive field** is the region of the input image that influences a particular neuron's output.

### How Receptive Field Grows

```
Layer 1: 3×3 receptive field (direct from kernel)
Layer 2: 5×5 receptive field (sees through layer 1)
Layer 3: 7×7 receptive field (sees through layers 1 and 2)
...
Deep layers: Can "see" the entire image
```

### Why Receptive Field Matters

- **Early layers**: Small receptive field → detect local features (edges)
- **Deep layers**: Large receptive field → detect global features (objects)
- **Design consideration**: Deeper networks or larger kernels increase receptive field

---

## Channels and Feature Maps

### Input Channels

- **Grayscale image**: 1 channel (height × width × 1)
- **RGB image**: 3 channels (height × width × 3)

### Feature Maps (Output Channels)

Each convolutional layer produces multiple feature maps, one per filter:

```
Input: 224×224×3 (RGB image)
       ↓
Conv Layer: 32 filters of size 3×3
       ↓
Output: 222×222×32 (32 feature maps)
```

Each feature map represents a different learned feature (edge detector, color blob detector, etc.).

---

## Key Terminology

| Term | Definition |
|------|------------|
| **Kernel/Filter** | Small matrix of learnable weights that slides across input |
| **Feature Map** | Output of applying a filter to the input |
| **Stride** | Step size when sliding the filter |
| **Padding** | Adding zeros around input to control output size |
| **Receptive Field** | Region of input that affects a neuron's output |
| **Channel** | Depth dimension (3 for RGB, N for N filters) |

---

## Summary

CNNs revolutionized computer vision by:

1. **Preserving spatial structure** through local connectivity
2. **Reducing parameters** through weight sharing
3. **Learning hierarchical features** automatically
4. **Achieving translation invariance** through convolution and pooling

These properties make CNNs the go-to architecture for any task involving images, from classification to object detection to image generation.

---

*Next: [02-Kernels-and-Filters.md](02-Kernels-and-Filters.md) — Learn how filters detect features*
