"""
Topic 8: Convolutional Neural Networks (CNNs) - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="08 - Convolutional Neural Networks",
    page_icon="🖼️",
    layout="wide"
)

# Main content
st.markdown("""
# Convolutional Neural Networks (CNNs) 🖼️

## Why CNNs for Images?

Regular fully-connected neural networks have problems with images:

### Problems with Fully-Connected Layers:
1. **Too many parameters**: A 224×224 RGB image = 150,528 input neurons. First hidden layer with 1,000 neurons = **150 million parameters**!
2. **No spatial structure**: Flattening destroys the 2D structure of images
3. **No translation invariance**: A cat in top-left corner is treated completely different from a cat in bottom-right

CNNs solve these problems with two key ideas:

### 1. Local Connectivity (Filters/Kernels)
Instead of connecting every neuron to every pixel, connect small regions:
- A 3×3 filter connects to only 9 pixels
- The same filter slides across the entire image
- This preserves spatial structure!

### 2. Parameter Sharing
The same filter is used across the entire image:
- One 3×3 filter = only 9 parameters (instead of millions!)
- Learns to detect features (edges, textures) anywhere in the image
- This gives **translation invariance** - detect cats anywhere!

**Why this works**: Natural images have local correlations. A small filter can detect edges, corners, textures. Stacking multiple layers builds up from simple to complex features!

---

## The Convolution Operation

A **convolution** slides a small filter across an image and computes dot products:

### Visual Example:

```
Input image (5×5):          Filter (3×3):
┌─────────────┐            ┌─────┐
│ 1 2 3 4 5  │            │ 1 0 -1│
│ 0 1 2 3 4  │            │ 1 0 -1│
│ 5 0 1 2 3  │            │ 1 0 -1│
│ 4 5 0 1 2  │            └─────┘
│ 3 4 5 0 1  │
└─────────────┘

Output (3×3):
┌─────────┐
│ -4  -4  -4│  Each value is a dot product
│  5   5   5│  of the filter with a 3×3 region
│ -4  -4  -4│
└─────────┘
```

### PyTorch Convolution:

```python
import torch
import torch.nn as nn

# Create a 2D convolution layer
conv = nn.Conv2d(
    in_channels=3,      # RGB input (3 channels)
    out_channels=64,    # Number of filters to learn
    kernel_size=3,      # 3×3 filter
    stride=1,           # Move filter 1 pixel at a time
    padding=1           # Add 1 pixel border (keeps size same)
)

# Input: batch of RGB images
x = torch.randn(32, 3, 224, 224)  # [batch, channels, height, width]

# Apply convolution
output = conv(x)
print(output.shape)  # torch.Size([32, 64, 224, 224])
```

### Key Parameters:

| Parameter | Description | Effect on Output Size |
|-----------|-------------|----------------------|
| `in_channels` | Input channels (RGB=3, grayscale=1) | - |
| `out_channels` | Number of filters/feature maps | Increases channels |
| `kernel_size` | Filter size (3, 5, 7, etc.) | Larger = smaller output |
| `stride` | Step size for sliding filter | Larger = smaller output |
| `padding` | Border pixels added | Larger = larger output |

### Output Size Formula:

```
output_size = (input_size - kernel_size + 2*padding) / stride + 1

Example: (224 - 3 + 2*1) / 1 + 1 = 224  (same size with padding=1)
```

---

## Pooling Layers

**Pooling** reduces spatial dimensions while keeping important information:

### Max Pooling (Most Common):
Takes the maximum value in each region:

```
Input (4×4):                MaxPool 2×2:          Output (2×2):
┌───────────┐              ┌─────┐               ┌─────┐
│ 1  3  2  4│              │ max │               │ 3  4│
│ 5  6  7  8│   ──────>    └─────┘    ──────>   │14 16│
│ 9 10 11 12│                                    └─────┘
│13 14 15 16│
└───────────┘
```

### PyTorch Pooling:

```python
import torch.nn as nn

# Max pooling - takes maximum in each region
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling - takes average
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# Example
x = torch.randn(32, 64, 224, 224)  # [batch, channels, height, width]
x = maxpool(x)
print(x.shape)  # torch.Size([32, 64, 112, 112]) - half the spatial size!
```

**Why pooling?**
- Reduces computation (fewer pixels)
- Provides translation invariance (small shifts don't matter)
- Increases receptive field (each neuron sees more of the image)

---

## Building a Complete CNN

The classic pattern: **Conv → Activation → Pool** repeated multiple times:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        # Block 1: 3 channels -> 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64 -> 128 feature maps
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully-connected classification layers
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1: Conv -> ReLU -> Pool
        x = self.conv1(x)           # [batch, 32, 224, 224]
        x = F.relu(x)
        x = self.pool1(x)           # [batch, 32, 112, 112]

        # Block 2: Conv -> ReLU -> Pool
        x = self.conv2(x)           # [batch, 64, 112, 112]
        x = F.relu(x)
        x = self.pool2(x)           # [batch, 64, 56, 56]

        # Block 3: Conv -> ReLU -> Pool
        x = self.conv3(x)           # [batch, 128, 56, 56]
        x = F.relu(x)
        x = self.pool3(x)           # [batch, 128, 28, 28]

        # Flatten for fully-connected layers
        x = x.view(x.size(0), -1)   # [batch, 128*28*28]

        # Fully-connected layers
        x = self.fc1(x)             # [batch, 512]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # [batch, num_classes]

        return x

# Create model
model = SimpleCNN(num_classes=10)

# Test with dummy input
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 10]

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Architecture pattern**:
- **Early layers**: Small filters (3×3), increasing channels (32→64→128)
- **Middle layers**: Pooling reduces spatial size
- **Late layers**: Flatten and fully-connected for classification

---

## Modern CNN Patterns

### 1. Multiple Conv Layers Before Pooling

```python
# VGG-style: 2 convs, then pool
self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

# Forward
x = F.relu(self.conv1_1(x))
x = F.relu(self.conv1_2(x))
x = self.pool1(x)
```

**Why?** Stacking convs increases receptive field without losing resolution!

### 2. 1×1 Convolutions

```python
# 1×1 conv changes channels without affecting spatial size
self.conv_1x1 = nn.Conv2d(128, 64, kernel_size=1)
```

**Why?** Reduce channels (parameters) while mixing information across channels!

### 3. Batch Normalization (After Conv)

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.bn1 = nn.BatchNorm2d(64)  # Normalize feature maps

# Forward
x = self.conv1(x)
x = self.bn1(x)
x = F.relu(x)
```

**Why?** Stabilizes training and allows higher learning rates!

---

## Classic CNN Architectures

### LeNet-5 (1998) - The Original CNN
```python
# 2 conv layers, 2 fully-connected
# Designed for handwritten digits (MNIST)
# ~60,000 parameters
```

### AlexNet (2012) - ImageNet Winner
```python
# 5 conv layers, 3 fully-connected
# ReLU activation, dropout, data augmentation
# ~60 million parameters
# Proved CNNs work for large-scale image recognition!
```

### VGGNet (2014) - Deeper is Better
```python
# 16-19 layers, all 3×3 filters
# Simple architecture: stack 3×3 convs, pool every 2-3 layers
# ~138 million parameters
```

### ResNet (2015) - Skip Connections
```python
# 50-152 layers possible with residual connections
# Skip connections prevent vanishing gradients
# Still widely used today!
```

**Evolution**: Deeper → More efficient → Skip connections → Attention mechanisms

---

## Feature Map Visualization

CNNs learn hierarchical features:

```python
import matplotlib.pyplot as plt

# Visualize what filters detect
def visualize_feature_maps(model, image):
    """
    Show what each filter detects in the first conv layer
    """
    # Get first conv layer
    conv1 = model.conv1

    # Forward pass through first conv only
    with torch.no_grad():
        feature_maps = conv1(image)  # [1, 32, 224, 224]

    # Plot first 16 feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < feature_maps.shape[1]:
            ax.imshow(feature_maps[0, idx].cpu(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {idx}')

    plt.tight_layout()
    plt.show()

# Example usage
model = SimpleCNN()
image = torch.randn(1, 3, 224, 224)
visualize_feature_maps(model, image)
```

**What you'll see**:
- **Layer 1**: Edge detectors (horizontal, vertical, diagonal)
- **Layer 2**: Texture detectors (patterns, corners)
- **Layer 3**: Part detectors (eyes, wheels, windows)
- **Layer 4**: Object detectors (faces, cars, animals)

---

## Why CNNs Work So Well for Images

### 1. Inductive Bias
CNNs have built-in assumptions about images:
- **Local patterns matter**: 3×3 neighborhoods are meaningful
- **Translation invariance**: Features are useful anywhere
- **Hierarchical composition**: Simple features → Complex objects

### 2. Parameter Efficiency
```python
# Fully-connected: 224×224×3 → 1000 = 150 million parameters
# CNN: 3×3 filter × 64 filters = 576 parameters (+ a few more layers)
```

### 3. Computational Efficiency
- Convolutions are highly parallelizable (GPUs love them!)
- Pooling reduces computation exponentially

### 4. Learned Features
CNNs learn features automatically from data (vs hand-crafted features like SIFT, HOG)

---

## Connection to Vision Transformers

Modern vision transformers (ViT) challenge CNNs:

```python
# CNN approach: Convolutions with inductive bias
conv = nn.Conv2d(3, 64, kernel_size=3)

# Vision Transformer approach: Self-attention on image patches
# Splits image into 16×16 patches, treats like tokens
# No inductive bias - learns everything from data!
```

**Trade-offs**:
- **CNNs**: Better with small datasets, faster, inductive bias helps
- **ViTs**: Better with huge datasets, more flexible, needs more data

**Hybrid approaches**: Many modern models combine both (ConvNeXt, Swin Transformer)!

---

## Practical CNN Example

Complete image classification:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define CNN
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Training loop (simplified)
model = ImageClassifier(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train one epoch
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

print("CNN classifier ready for training!")
```

---

## Key Takeaways 💡

✅ **CNNs are designed for images**: Local connectivity and parameter sharing make them efficient
✅ **Convolution operation**: Slide filters across images to detect features
✅ **Pooling**: Reduces spatial size while keeping important information
✅ **Classic pattern**: Conv → ReLU → Pool, repeated with increasing channels
✅ **Hierarchical features**: Early layers detect edges, later layers detect objects
✅ **Translation invariance**: Same filter works anywhere in the image
✅ **Modern evolution**: ResNets (skip connections) → Vision Transformers (attention)
✅ **Best for**: Images, videos, spatial data with local patterns

**Next topic**: Learn about Batch Normalization and Dropout - two powerful techniques for training better neural networks!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "Why do CNNs use parameter sharing (the same filter slides across the entire image)?",
        "options": [
            "To make the model train faster",
            "To achieve translation invariance and reduce parameters",
            "To increase model accuracy",
            "To make the code simpler"
        ],
        "correct": "To achieve translation invariance and reduce parameters",
        "explanation": "Parameter sharing means the same 3×3 filter is used everywhere in the image. This gives translation invariance (detect features anywhere) and dramatically reduces parameters compared to fully-connected layers. A cat detector works whether the cat is in the top-left or bottom-right!"
    },
    {
        "question": "What is the purpose of pooling layers in CNNs?",
        "options": [
            "To increase the number of parameters",
            "To add non-linearity",
            "To reduce spatial dimensions while keeping important features",
            "To normalize the activations"
        ],
        "correct": "To reduce spatial dimensions while keeping important features",
        "explanation": "Pooling (like MaxPool2d) reduces the spatial size of feature maps by taking the maximum (or average) in each region. This reduces computation, provides translation invariance, and increases the receptive field while keeping the most important features."
    },
    {
        "question": "What do early CNN layers (first few convolutions) typically detect?",
        "options": [
            "Complete objects like faces or cars",
            "Simple features like edges, corners, and textures",
            "Complex patterns and entire scenes",
            "Class probabilities"
        ],
        "correct": "Simple features like edges, corners, and textures",
        "explanation": "CNNs learn hierarchical features. Early layers detect simple, low-level features like edges and corners. Middle layers combine these into textures and parts. Late layers detect complex objects. This mirrors how the human visual system works!"
    },
    {
        "question": "In a Conv2d layer with kernel_size=3, stride=1, padding=1 applied to a 224×224 image, what is the output spatial size?",
        "options": [
            "222×222 (smaller)",
            "224×224 (same size)",
            "226×226 (larger)",
            "112×112 (half size)"
        ],
        "correct": "224×224 (same size)",
        "explanation": "Using the formula: output = (input - kernel + 2*padding) / stride + 1 = (224 - 3 + 2*1) / 1 + 1 = 224. With padding=1 and stride=1, the output size stays the same as input. This is called 'same' padding!"
    }
]

for idx, q in enumerate(questions):
    st.markdown(f"### Question {idx + 1}")
    st.markdown(f"**{q['question']}**")

    user_answer = st.radio(
        "Select your answer:",
        options=q["options"],
        key=f"q{idx}",
        index=None
    )

    if st.button(f"Check Answer {idx + 1}", key=f"btn{idx}"):
        if user_answer:
            if user_answer == q["correct"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Incorrect. {q['explanation']}")
        else:
            st.warning("Please select an answer first!")

    st.markdown("---")

# Navigation
st.info("👈 Use the sidebar to navigate to the next topic: **09 - Batch Normalization and Dropout**")
