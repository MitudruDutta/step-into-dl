# 📦 Efficient Data Pipelines: Datasets & DataLoaders

Training an AI model on a large dataset requires sophisticated memory management and data organization. PyTorch provides elegant solutions through `Dataset` and `DataLoader` classes.

---

## Why Data Pipelines Matter

### The Problem
- Datasets can be gigabytes or terabytes in size
- Loading everything into memory is impossible
- Sequential processing is slow
- Random access patterns improve learning

### The Solution
PyTorch's data utilities handle:
- Lazy loading (load data only when needed)
- Batching (process multiple samples together)
- Shuffling (randomize order each epoch)
- Parallel loading (use multiple CPU cores)

---

## Built-in Datasets

PyTorch comes with numerous pre-loaded datasets covering multiple domains:

### Image Datasets (torchvision)

| Dataset | Description | Classes | Size |
|---------|-------------|---------|------|
| **MNIST** | Handwritten digits | 10 | 70,000 |
| **CIFAR-10** | Small color images | 10 | 60,000 |
| **CIFAR-100** | Small color images | 100 | 60,000 |
| **ImageNet** | Large-scale images | 1,000 | 1.2M |
| **FashionMNIST** | Clothing items | 10 | 70,000 |

```python
from torchvision import datasets, transforms

# Download and load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

### Text Datasets (torchtext)

| Dataset | Description |
|---------|-------------|
| **IMDB** | Movie review sentiment |
| **WikiText** | Language modeling |
| **AG_NEWS** | News classification |

### Audio Datasets (torchaudio)

| Dataset | Description |
|---------|-------------|
| **Speech Commands** | Spoken word recognition |
| **LibriSpeech** | Speech recognition |
| **VCTK** | Multi-speaker corpus |

---

## Custom Datasets

For your own data, create a class inheriting from `torch.utils.data.Dataset`.

### Required Methods

| Method | Purpose |
|--------|---------|
| `__init__` | Initialize dataset, load file paths |
| `__len__` | Return total number of samples |
| `__getitem__` | Return one sample by index |

### Basic Example

```python
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Usage
dataset = CustomDataset(X_train, y_train)
print(f"Dataset size: {len(dataset)}")
sample, label = dataset[0]  # Get first sample
```

### CSV File Example

```python
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_file, target_column):
        self.df = pd.read_csv(csv_file)
        self.features = self.df.drop(columns=[target_column]).values
        self.targets = self.df[target_column].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
```

### Image Folder Example

```python
from torchvision import datasets, transforms

# Assumes folder structure:
# data/train/class1/img1.jpg
# data/train/class2/img2.jpg

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(
    root='data/train',
    transform=transform
)
```

---

## The DataLoader

The `DataLoader` wraps your dataset and provides critical training features.

### Basic Usage

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

# Iterate through batches
for batch_x, batch_y in train_loader:
    # batch_x shape: [32, features]
    # batch_y shape: [32]
    pass
```

---

## DataLoader Features

### 1. Batching

Automatically splits datasets into smaller, manageable batches:

```python
# Dataset has 1000 samples, batch_size=32
# Results in 32 batches (31 full + 1 partial)

loader = DataLoader(dataset, batch_size=32)
```

**Why batching matters:**
- Fits in GPU memory
- More frequent weight updates
- Gradient estimates from batch (not single sample)
- Faster training through parallelization

### 2. Shuffling

Reshuffles data at the start of every epoch:

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Why shuffling matters:**
- Prevents learning data order
- Reduces overfitting
- Each epoch sees different batch compositions
- Improves generalization

**Important:** Only shuffle training data, not validation/test data.

### 3. Parallel Loading

Use multiple CPU processes to load data:

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4  # Use 4 CPU processes
)
```

**How it works:**
- CPU prepares next batch while GPU processes current batch
- Eliminates I/O bottleneck
- Dramatically speeds up training for large datasets

**Choosing num_workers:**
- Start with 2-4
- Increase if GPU utilization is low
- Don't exceed CPU core count
- Set to 0 for debugging (single process)

### 4. Pin Memory

Speed up CPU→GPU transfer:

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True  # Use pinned (page-locked) memory
)
```

**When to use:**
- Training on GPU
- Large batches
- I/O-bound training

---

## DataLoader Parameters Reference

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `batch_size` | Samples per batch | 32, 64, 128, 256 |
| `shuffle` | Randomize order each epoch | True for train, False for val/test |
| `num_workers` | Parallel data loading processes | 2-8 (0 for debugging) |
| `drop_last` | Drop incomplete final batch | True for training |
| `pin_memory` | Speed up CPU→GPU transfer | True if using GPU |
| `collate_fn` | Custom batch assembly function | For variable-length data |
| `sampler` | Custom sampling strategy | For imbalanced datasets |

---

## Complete Training Setup

```python
from torch.utils.data import DataLoader, random_split

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,  # Don't shuffle validation
    num_workers=4,
    pin_memory=True
)
```

---

## Handling Imbalanced Data

### Using WeightedRandomSampler

```python
from torch.utils.data import WeightedRandomSampler

# Calculate class weights (inverse frequency)
class_counts = [1000, 100]  # Class 0: 1000, Class 1: 100
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]  # Weight for each sample

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: Can't use shuffle=True with sampler
```

---

## Common Patterns

### Pattern 1: Training Loop with DataLoader

```python
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Pattern 2: Progress Bar with tqdm

```python
from tqdm import tqdm

for epoch in range(num_epochs):
    for data, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
        # Training code
        pass
```

---

## Best Practices

1. **Always shuffle training data** — prevents learning order
2. **Don't shuffle validation/test data** — ensures reproducibility
3. **Use `drop_last=True` for training** — avoids small final batches
4. **Set `num_workers > 0`** — parallelizes data loading
5. **Use `pin_memory=True` on GPU** — speeds up transfers
6. **Start with smaller batch sizes** — increase if GPU memory allows
7. **Use transforms for augmentation** — increases effective dataset size

---

*Efficient data pipelines are crucial for training at scale. Master DataLoaders to eliminate bottlenecks and speed up your experiments.*
